from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
from lets_plot import (
    LetsPlot,
    aes,
    facet_grid,
    geom_hline,
    geom_line,
    geom_rect,
    ggplot,
    ggsize,
    labs,
    scale_y_continuous,
    theme_light,
)

from .builder import TimeSeriesBuilder
from .functions import normalize
from .generators import GENERATOR_FUNCS, generate_component

LetsPlot.set_theme(theme_light())

# Module constants
DEFAULT_COMPONENTS = ["aggregated", "features", "background"]
DEFAULT_Y_PADDING = 0.05  # 5% padding for y-axis limits
COMPONENT_DISPLAY_ORDER = ["Background", "Features", "Aggregated"]
# as_discrete() does not work in facet_grid() in lets-plot, so we enforce a stable facet
# order with invisible prefix characters and default alpha sorting (hacky, but it works)
COMPONENT_FACET_ORDER_PREFIXES = {
    "Background": "\u200b",  # zero width space
    "Features": "\u200c",  # zero width non-joiner
    "Aggregated": "\u200d",  # zero width joiner
}


def _extract_dim_from_feature_name(feature_name: str) -> Union[int, None]:
    """Extract dimension number from a feature name containing '_dim' suffix.

    Args:
        feature_name: Feature name that may contain '_dimN' suffix.

    Returns:
        The dimension number if found, None otherwise.
    """
    if "_dim" not in str(feature_name):
        return None
    try:
        dim_parts = str(feature_name).split("_dim")
        return int(dim_parts[-1])
    except (ValueError, IndexError):
        return None


def _find_contiguous_regions(mask: np.ndarray) -> List[tuple]:
    """Find contiguous True regions in a boolean mask.

    Args:
        mask: 1D boolean array.

    Returns:
        List of (start, end) tuples for each contiguous True region.
    """
    if not np.any(mask):
        return []

    changes = np.diff(np.concatenate([[False], mask, [False]]).astype(int))
    start_indices = np.where(changes == 1)[0]
    end_indices = np.where(changes == -1)[0]

    return list(zip(start_indices, end_indices))


def _calculate_y_limits(values: pd.Series, padding: float = DEFAULT_Y_PADDING) -> tuple:
    """Calculate y-axis limits with padding.

    Args:
        values: Series of values to calculate limits for.
        padding: Padding as fraction of range (default 0.05 = 5%).

    Returns:
        Tuple of (y_min, y_max) with padding applied.
    """
    y_min = values.min()
    y_max = values.max()
    y_range = y_max - y_min
    if y_range == 0:
        y_range = 1.0  # Avoid zero range
    pad = y_range * padding
    return float(y_min - pad), float(y_max + pad)


def _add_geom_rect(
    plot: Any,
    rectangles: pd.DataFrame,
    rect_fill: str,
    rect_alpha: float,
) -> Any:
    """Add rectangle geometries to a plot.

    Args:
        plot: The lets_plot object to add rectangles to.
        rectangles: DataFrame with xmin, xmax, ymin, ymax columns.
        rect_fill: Fill color for rectangles.
        rect_alpha: Alpha transparency for rectangles.

    Returns:
        The plot with rectangles added.
    """
    plot = plot + geom_rect(
        data=rectangles,
        mapping=aes(xmin="xmin", xmax="xmax", ymin="ymin", ymax="ymax"),
        fill=rect_fill,
        alpha=rect_alpha,
        size=0.2,
        color="grey",
    )
    return plot


def _add_geom_line(
    plot: Any,
    line_color: str,
    line_size: float,
    group_col: str = "line_group",
    data: Union[pd.DataFrame, None] = None,
) -> Any:
    """Add line geometry to a plot.

    Explicit grouping is needed when panels contain repeated x-values from multiple
    feature rows (e.g., multiple features in the same class/component facet).

    Args:
        plot: The lets_plot object to add lines to.
        line_color: Color for the lines.
        line_size: Size of the lines.
        group_col: Column name to use for explicit line grouping.
        data: DataFrame backing the plot, used to check if group_col exists.

    Returns:
        The plot with lines added.
    """
    if data is not None and group_col in data.columns:
        plot = plot + geom_line(
            mapping=aes(group=group_col), size=line_size, color=line_color
        )
    else:
        plot = plot + geom_line(size=line_size, color=line_color)
    return plot


def _add_line_group_column(df: pd.DataFrame) -> pd.DataFrame:
    """Create a deterministic line-group key for stable line rendering.

    lets-plot can connect unrelated points when repeated `time` values exist in one
    facet (common for multiple feature rows). This key ensures each logical series
    is rendered independently.
    """
    result = df.copy()

    group_parts = []
    for col, fallback in [
        ("sample", "all"),
        ("class", "all"),
        ("component", "all"),
        ("dim", "all"),
    ]:
        if col in result.columns:
            group_parts.append(result[col].astype(str))
        else:
            group_parts.append(pd.Series(fallback, index=result.index))

    if "feature" in result.columns:
        feature_part = result["feature"].fillna("__no_feature__").astype(str)
    else:
        feature_part = pd.Series("__no_feature__", index=result.index)
    group_parts.append(feature_part)

    line_group = group_parts[0]
    for part in group_parts[1:]:
        line_group = line_group + "|" + part

    result["line_group"] = line_group
    return result


def _capitalize_components(df: pd.DataFrame) -> pd.DataFrame:
    """Capitalize component names in a DataFrame.

    Args:
        df: DataFrame with a 'component' column.

    Returns:
        DataFrame with capitalized component names as ordered categorical.
    """
    df = df.copy()
    component_display_values = [c.capitalize() for c in df["component"].astype(str)]
    unique_values = pd.Index(component_display_values).unique()
    ordered_defaults = [c for c in COMPONENT_DISPLAY_ORDER if c in unique_values]
    remaining = sorted(c for c in unique_values if c not in COMPONENT_DISPLAY_ORDER)
    component_display_names = ordered_defaults + remaining
    df["component"] = pd.Categorical(
        component_display_values,
        categories=component_display_names,
        ordered=True,
    )
    return df


def _apply_component_facet_order(
    df: Union[pd.DataFrame, None],
) -> Union[pd.DataFrame, None]:
    """Apply an internal sortable representation to component facet labels.

    lets-plot currently fails with `as_discrete()` inside `facet_grid()`, so this helper
    prepends invisible characters to known component names. Facet strip labels stay
    visually unchanged while alphabetical ordering becomes deterministic.
    """
    if df is None or "component" not in df.columns:
        return df

    result = df.copy()
    component_values = result["component"].astype(str)
    result["component"] = (
        component_values.map(COMPONENT_FACET_ORDER_PREFIXES).fillna("")
        + component_values
    )
    return result


def _build_rectangles_from_mask(
    mask: np.ndarray,
    class_label: int,
    feature_name: str,
    dim: Union[int, None] = None,
) -> List[Dict]:
    """Build rectangle dictionaries from a boolean mask.

    Args:
        mask: 1D boolean array indicating feature locations.
        class_label: The class label for these rectangles.
        feature_name: Name of the feature.
        dim: Dimension index if applicable.

    Returns:
        List of rectangle dictionaries with xmin, xmax, class, component, feature keys.
    """
    regions = _find_contiguous_regions(mask)
    rectangles = []

    for start, end in regions:
        rect = {
            "class": f"Class {class_label}",
            "component": "aggregated",
            "feature": feature_name,
            "xmin": float(start),
            "xmax": float(end),
        }
        if dim is not None:
            rect["dim"] = dim
        rectangles.append(rect)

    return rectangles


def _should_skip_dimension(
    dim_match: Union[int, None], dimensions: Union[List[int], None]
) -> bool:
    """Check if a feature should be skipped based on dimension filtering.

    Args:
        dim_match: The dimension extracted from feature name, or None.
        dimensions: List of dimensions to include, or None for all.

    Returns:
        True if the feature should be skipped, False otherwise.
    """
    return (
        dimensions is not None and dim_match is not None and dim_match not in dimensions
    )


def _ensure_visualization_format(dataset: Dict) -> Dict:
    """Ensure the dataset is in the appropriate format for visualization.

    The visualization code expects data in 'channels_last' format.
    This function checks the format and converts if necessary.

    Args:
        dataset (Dict): Dataset containing time series data and components.

    Returns:
        Dict: Dataset with data in the correct format for visualization.
    """
    # Create a shallow copy to avoid modifying the original
    dataset = dataset.copy()

    # Check if the dataset has metadata specifying the format
    if "metadata" in dataset and "data_format" in dataset["metadata"]:
        data_format = dataset["metadata"]["data_format"]

        # If data is in channels_first format, convert to channels_last for visualization
        if data_format == "channels_first" and "X" in dataset:
            # Use the convert_data_format utility from TimeSeriesBuilder
            dataset = TimeSeriesBuilder.convert_data_format(dataset, "channels_last")

    return dataset


def plot_component(
    signal: np.ndarray = None,
    component_type: str = None,
    n_timesteps: int = 100,
    rng: np.random.RandomState = None,
    width: int = 500,
    height: int = 250,
    line_color: str = "black",
    line_size: float = 1.5,
    hline_intercept: float = None,
    normalization: str = "none",
    normalization_kwargs: Dict[str, Any] = None,
    title: str = None,
    **kwargs: Any,
) -> Any:
    """Plot a time series signal generated by one of the generator functions.

    This function can be used in two ways:
    1. Pass a pre-generated signal array directly
    2. Specify a component_type and parameters to generate a new signal

    Args:
        signal (Optional[np.ndarray]): Pre-generated signal array.
        component_type (Optional[str]): Type of component to generate (if signal not provided).
        n_timesteps (int): Length of time series to generate (if signal not provided).
        rng (Optional[np.random.RandomState]): Random number generator.
        width (int): Plot width in pixels.
        height (int): Plot height in pixels.
        line_color (str): Color of the signal line.
        line_size (float): Size of the signal line.
        hline_intercept (Optional[float]): Y-intercept for horizontal line.
            If None, no line is shown.
        normalization (str): Normalization method. Options: "none", "minmax", "zscore".
            "none" means no normalization, "minmax" scales to [0, 1], "zscore" standardizes.
        normalization_kwargs (Optional[Dict[str, Any]]): Additional parameters for normalization methods.
            For "minmax", you can pass feature_range (tuple of min and max).
        title (Optional[str]): Title for the plot.
        **kwargs (Any): Additional parameters for the generator function.

    Returns:
        Any: lets_plot visualization object.
    """
    # Handle default for normalization_kwargs
    if normalization_kwargs is None:
        normalization_kwargs = {}

    # Create default RNG if not provided
    if rng is None:
        rng = np.random.RandomState(None)  # Use system time as seed

    # Generate signal if not provided directly
    if signal is None:
        if component_type is None:
            raise ValueError("Either signal or component_type must be provided")

        if component_type not in GENERATOR_FUNCS:
            raise ValueError(
                f"Unknown component type: {component_type}. "
                f"Valid types are: {', '.join(GENERATOR_FUNCS.keys())}"
            )

        signal = generate_component(component_type, n_timesteps, rng, **kwargs)

    # Ensure signal is numpy array & normalize signal
    signal = np.asarray(signal)
    signal = normalize(signal, method=normalization, **normalization_kwargs)

    # Create dataframe for plotting - using pandas DataFrame to ensure proper types
    n_points = len(signal)
    data = pd.DataFrame({"time": np.arange(n_points), "value": signal})

    # Create the plot
    p = ggplot(data, aes(x="time", y="value")) + geom_line(
        color=line_color, size=line_size
    )

    if hline_intercept is not None:
        p = p + geom_hline(yintercept=hline_intercept, linetype="dashed", color="gray")

    # Get title
    if title is None:
        title = f"{component_type.replace('_', ' ').capitalize() if component_type else 'Precomputed Signal'}"
        title += f" ({normalization} normalized)" if normalization != "none" else ""

    # Add labels and theme
    p = p + labs(title=title, x="Time Steps", y="Value") + ggsize(width, height)

    return p


def prepare_plot_data(
    dataset: Dict,
    sample_indices: Union[Dict[int, int], List[int], int, None] = None,
    components_to_include: List[str] = None,
    dimensions: List[int] = None,
) -> pd.DataFrame:
    """Create data for visualization with lets_plot.

    Args:
        dataset (Dict): Dataset containing time series data and components.
        sample_indices (Union[Dict[int, int], List[int], int, None]): Samples to include.
            Can be provided in several formats:
            - Dict[int, int]: Mapping from class labels to sample indices (e.g., {0: 5, 1: 10})
            - List[int]: List of sample indices to include (e.g., [0, 5, 10])
            - int: Single sample index (e.g., 5)
            - None: Use first sample of each class (default)
        components_to_include (Optional[List[str]]): List of components to include. If None, include all.
            Default components: ["aggregated", "features", "background"].
        dimensions (Optional[List[int]]): List of dimensions to include. If None, include all dimensions.

    Returns:
        pd.DataFrame: Prepared DataFrame for visualization.
    """
    # Ensure dataset is in channels_last format for visualization
    dataset = _ensure_visualization_format(dataset)

    # Process sample_indices to determine which samples to include
    if sample_indices is None:
        # Default: first sample of each class
        indices_dict = {}
        for class_label in np.unique(dataset["y"]):
            indices_dict[class_label] = np.where(dataset["y"] == class_label)[0][0]
        indices = list(indices_dict.values())
    elif isinstance(sample_indices, dict):
        # Dict mapping class -> index
        indices = list(sample_indices.values())
    elif isinstance(sample_indices, int):
        # Single sample index
        indices = [sample_indices]
    elif isinstance(sample_indices, list):
        # List of sample indices
        indices = sample_indices
    else:
        raise TypeError(
            f"sample_indices must be Dict[int, int], List[int], int, or None, "
            f"got {type(sample_indices)}"
        )

    # Use the to_df method to get the data in the right format
    builder = TimeSeriesBuilder()
    df = builder.to_df(
        dataset,
        samples=indices,
        components=components_to_include,
        dimensions=dimensions,
        format_classes=True,
    )

    return df


def prepare_feature_highlights(
    dataset: Dict,
    sample_indices: Dict[int, int] = None,
    components_to_include: List[str] = None,
    dimensions: Union[int, List[int]] = None,
) -> pd.DataFrame:
    """Create rectangle data for feature visualization.

    Args:
        dataset (Dict): Dataset containing time series data and components.
        sample_indices (Optional[Dict[int, int]]): Dictionary mapping class labels to sample indices.
            If None, use first sample of each class.
        components_to_include (Optional[List[str]]): List of components to include. If None, include all.
        dimensions (Optional[Union[int, List[int]]]): List of dimensions to include or an integer for a
            single dimension. If None, include all dimensions.

    Returns:
        Optional[pd.DataFrame]: DataFrame with rectangle coordinates for feature regions,
            or None if no features found.
    """
    # Ensure dataset is in channels_last format for visualization
    dataset = _ensure_visualization_format(dataset)

    # If components_to_include doesn't have aggregated, no need for rectangles
    if components_to_include is not None and "aggregated" not in components_to_include:
        return None

    # Determine sample indices if not provided
    if sample_indices is None:
        sample_indices = {}
        for class_label in np.unique(dataset["y"]):
            sample_indices[class_label] = np.where(dataset["y"] == class_label)[0][0]

    # Convert dimensions to list if it's a single value
    if dimensions is not None and not isinstance(dimensions, list):
        dimensions = [dimensions]

    rectangles = []

    # Process each class and sample
    for class_label, idx in sample_indices.items():
        class_rectangles = []

        # Try dataset-level feature_masks first
        if "feature_masks" in dataset:
            for key, mask in dataset["feature_masks"].items():
                if f"class_{class_label}_" not in key:
                    continue

                dim_match = _extract_dim_from_feature_name(key)
                if _should_skip_dimension(dim_match, dimensions):
                    continue

                feature_name = key.replace(f"class_{class_label}_", "")
                sample_mask = mask[idx]
                class_rectangles.extend(
                    _build_rectangles_from_mask(
                        sample_mask, class_label, feature_name, dim_match
                    )
                )

        # If no rectangles found, try component-level feature_masks
        if not class_rectangles and "components" in dataset:
            comp = dataset["components"][idx]

            if hasattr(comp, "feature_masks") and comp.feature_masks:
                for feature_name, feature_mask in comp.feature_masks.items():
                    dim_match = _extract_dim_from_feature_name(feature_name)
                    if _should_skip_dimension(dim_match, dimensions):
                        continue

                    class_rectangles.extend(
                        _build_rectangles_from_mask(
                            feature_mask, class_label, feature_name, dim_match
                        )
                    )

            # If still no rectangles, try feature values (non-zero regions)
            if not class_rectangles and hasattr(comp, "features") and comp.features:
                for feature_name, feature_values in comp.features.items():
                    dim_match = _extract_dim_from_feature_name(feature_name)
                    if _should_skip_dimension(dim_match, dimensions):
                        continue

                    # Create mask from non-zero values
                    non_zero_mask = np.abs(feature_values) > 1e-6
                    class_rectangles.extend(
                        _build_rectangles_from_mask(
                            non_zero_mask, class_label, feature_name, dim_match
                        )
                    )

        rectangles.extend(class_rectangles)

    if not rectangles:
        return None

    rect_df = pd.DataFrame(rectangles)

    # Ensure correct class ordering
    class_labels = sorted(rect_df["class"].unique(), key=lambda x: int(x.split()[-1]))
    rect_df["class"] = pd.Categorical(
        rect_df["class"], categories=class_labels, ordered=True
    )

    return rect_df


def plot_components(
    dataset: Dict,
    sample_indices: Union[int, List[int], Dict[int, int], None] = None,
    components: List[str] = None,
    dimensions: List[int] = None,
    show_indicators: bool = True,
    line_color: str = "black",
    line_size: float = 1.5,
    hline_intercept: float = 0,
    rect_fill: str = "red",
    rect_alpha: float = 0.25,
    facet_order: Dict[str, str] = {"y": "class", "x": "component"},
    x_order: int = 1,
    y_order: int = 1,
    panel_width: int = 250,
    panel_height: int = 150,
) -> Union[Any, List[Any]]:
    """Create time series visualization with feature indicators as rectangles.

    This function visualizes time series data by showing its components (aggregated series,
    background, features) with options to highlight feature locations.

    Args:
        dataset (Dict): Dataset containing time series data and components.
        sample_indices (Union[int, List[int], Dict[int, int], None]): Specifies which samples to visualize.
            Note: This function shows ONE sample per class for comparison purposes.
            Can be provided in several formats:
            - int: A single sample index (shows that sample and its class)
            - List[int]: Multiple sample indices. These are grouped by class, and if
              multiple samples belong to the same class, only the last one is shown.
            - Dict[int, int]: Mapping from class labels to sample indices
              (e.g., {0: 5, 1: 10} means sample at index 5 for class 0, sample at index 10
              for class 1). The sample at each index must actually belong to the specified class.
            - None: Use first sample from each class (default behavior)
        components (Optional[List[str]]): List of components to include. Can be used to exclude
            certain components. Default: ["aggregated", "features", "background"].
        dimensions (Optional[List[int]]): List of dimensions to include. If None, include all dimensions.
            For multivariate time series, this allows selecting specific dimensions.
        show_indicators (bool): Whether to show feature indicators.
        line_color (str): Color of the time series lines.
        line_size (float): Size of the time series lines.
        hline_intercept (float): Y-intercept for horizontal line at y=hline_intercept.
            If None, no line is shown.
        rect_fill (str): Fill color for feature rectangles.
        rect_alpha (float): Alpha transparency for feature rectangles.
        facet_order (Dict[str, str]): Order of facets, dict with "x" and "y" keys.
            x corresponds to columns, y to rows.
        x_order (int): Order of x-axis facets. 1=ascending, -1=descending, 0=no order.
            Uses names of the column facet variable.
        y_order (int): Order of y-axis facets. 1=ascending, -1=descending, 0=no order.
            Uses names of the row facet variable.
        panel_width (int): Width of each panel in pixels.
        panel_height (int): Height of each panel in pixels.

    Returns:
        Union[Any, List[Any]]: lets_plot visualization or list of visualizations for multivariate data.

    Raises:
        IndexError: If a sample index is out of range.

    Examples:
        # Show first sample of each class (default)
        plot_components(dataset)

        # Show a specific sample by index (shows only the class of that sample)
        plot_components(dataset, sample_indices=5)

        # Show samples from different classes (grouped by class, one sample per class)
        # If indices 0, 10, 20 are from classes 0, 1, 0 respectively, only samples 10 and 20
        # will be shown (last sample for class 0, and sample 10 for class 1)
        plot_components(dataset, sample_indices=[0, 10, 20])

        # Explicit mapping: show sample 5 for class 0, sample 10 for class 1
        # (the sample at index 5 must belong to class 0, and sample at index 10 to class 1)
        plot_components(dataset, sample_indices={0: 5, 1: 10})
    """
    # Ensure dataset is in channels_last format for visualization
    dataset = _ensure_visualization_format(dataset)

    # Process sample_indices parameter to normalize to dict format
    processed_indices = None

    if sample_indices is not None:
        if isinstance(sample_indices, int):
            # Single sample index
            if sample_indices < 0 or sample_indices >= len(dataset["y"]):
                raise IndexError(
                    f"Sample index {sample_indices} is out of range (0-{len(dataset['y']) - 1})"
                )
            class_label = dataset["y"][sample_indices]
            processed_indices = {class_label: sample_indices}

        elif isinstance(sample_indices, list):
            # Multiple sample indices - group by class
            for idx in sample_indices:
                if idx < 0 or idx >= len(dataset["y"]):
                    raise IndexError(
                        f"Sample index {idx} is out of range (0-{len(dataset['y']) - 1})"
                    )
            processed_indices = {}
            for idx in sample_indices:
                class_label = dataset["y"][idx]
                processed_indices[class_label] = idx

        elif isinstance(sample_indices, dict):
            # Validate the indices
            for class_label, idx in sample_indices.items():
                if idx < 0 or idx >= len(dataset["y"]):
                    raise IndexError(
                        f"Sample index {idx} is out of range (0-{len(dataset['y']) - 1})"
                    )
                actual_class = dataset["y"][idx]
                if actual_class != class_label:
                    raise ValueError(
                        f"Sample at index {idx} has class {actual_class}, not {class_label} as specified"
                    )
            processed_indices = sample_indices

    components_to_use = components if components is not None else DEFAULT_COMPONENTS

    # Prepare data for visualization
    df = prepare_plot_data(dataset, processed_indices, components_to_use, dimensions)
    assert len(df) > 0, "No data to display"

    is_multivariate = len(df["dim"].unique()) > 1
    df = _capitalize_components(df)
    df = _apply_component_facet_order(df)
    df = _add_line_group_column(df)

    # Prepare rectangles for feature highlighting
    rectangles = None
    if show_indicators:
        rectangles = prepare_feature_highlights(
            dataset, processed_indices, components_to_use, dimensions
        )
        if rectangles is not None and "component" in rectangles.columns:
            rectangles = rectangles.copy()
            rectangles["component"] = rectangles["component"].str.capitalize()
            rectangles = _apply_component_facet_order(rectangles)

    # For multivariate time series, use specialized dimension-based plotting
    if is_multivariate:
        return _plot_dimensions(
            df,
            rectangles,
            line_color=line_color,
            line_size=line_size,
            hline_intercept=hline_intercept,
            rect_fill=rect_fill,
            rect_alpha=rect_alpha,
            panel_width=panel_width,
            panel_height=panel_height,
        )

    # For univariate case
    y_min, y_max = _calculate_y_limits(df["value"])

    # Calculate plot dimensions
    n_components = len(df["component"].unique())
    n_classes = len(df["class"].unique())
    total_width = panel_width * (
        n_components
        if "x" in facet_order and facet_order["x"] == "component"
        else n_classes
    )
    total_height = panel_height * (
        n_classes
        if "y" in facet_order and facet_order["y"] == "class"
        else n_components
    )

    # Start building the plot
    p = ggplot(df, aes(x="time", y="value"))

    if hline_intercept is not None:
        p = p + geom_hline(yintercept=hline_intercept, linetype="dashed", color="gray")

    # Add rectangles for feature regions
    if show_indicators and rectangles is not None:
        rectangles = rectangles.copy()
        rectangles["ymin"] = y_min
        rectangles["ymax"] = y_max
        p = _add_geom_rect(p, rectangles, rect_fill, rect_alpha)

    p = _add_geom_line(p, line_color, line_size, data=df)

    # Use regular facet grid with proper ordering
    p = p + facet_grid(**facet_order, scales="fixed", x_order=x_order, y_order=y_order)
    p = p + scale_y_continuous(limits=[y_min, y_max])

    p = p + labs(x="Time Steps", y="Value")
    p = p + ggsize(total_width, total_height)

    return p


def _plot_dimensions(
    df: pd.DataFrame,
    rectangles: pd.DataFrame = None,
    line_color: str = "black",
    line_size: float = 1.5,
    hline_intercept: float = 0,
    rect_fill: str = "red",
    rect_alpha: float = 0.25,
    panel_width: int = 250,
    panel_height: int = 150,
) -> Union[Any, List[Any]]:
    """Plot multivariate time series with dimensions as separate facet grids.

    Internal helper function that creates separate plots for each dimension.

    Args:
        df (pd.DataFrame): DataFrame with time series data, already processed by prepare_plot_data.
        rectangles (Optional[pd.DataFrame]): DataFrame with rectangle data for feature indicators.
        line_color (str): Color of the time series lines.
        line_size (float): Size of the time series lines.
        hline_intercept (float): Y-intercept for horizontal line.
        rect_fill (str): Fill color for feature rectangles.
        rect_alpha (float): Alpha transparency for feature rectangles.
        panel_width (int): Width of each panel in pixels.
        panel_height (int): Height of each panel in pixels.

    Returns:
        Union[Any, List[Any]]: lets_plot visualization organized by dimensions.
    """
    df = _add_line_group_column(df)
    dims = sorted(df["dim"].unique())
    plots = []

    for dim in dims:
        dim_df = df[df["dim"] == dim].copy()
        y_min, y_max = _calculate_y_limits(dim_df["value"])

        # Filter rectangles for this dimension
        dim_rects = None
        if rectangles is not None and len(rectangles) > 0:
            if "dim" in rectangles.columns:
                dim_rects = rectangles[rectangles["dim"] == dim].copy()
            else:
                # Try to extract dimension from feature names
                dim_rects = rectangles[
                    rectangles["feature"].apply(
                        lambda f: f"_dim{dim}" in str(f)
                        or not any(f"_dim{d}" in str(f) for d in dims)
                    )
                ].copy()

            if len(dim_rects) > 0:
                dim_rects["ymin"] = y_min
                dim_rects["ymax"] = y_max
            else:
                dim_rects = None

        # Build the plot
        p = ggplot(dim_df, aes(x="time", y="value"))

        if hline_intercept is not None:
            p = p + geom_hline(
                yintercept=hline_intercept, linetype="dashed", color="gray"
            )

        if dim_rects is not None:
            p = _add_geom_rect(p, dim_rects, rect_fill, rect_alpha)

        p = _add_geom_line(p, line_color, line_size, data=dim_df)

        p = p + facet_grid(y="class", x="component", scales="fixed")
        p = p + scale_y_continuous(limits=[y_min, y_max])

        p = p + labs(title=f"Dimension {dim}", x="Time steps", y="Value")

        n_components = len(dim_df["component"].unique())
        n_classes = len(dim_df["class"].unique())
        p = p + ggsize(panel_width * n_components, panel_height * n_classes)

        plots.append(p)

    return plots[0] if len(plots) == 1 else plots


def plot_sample(
    X: np.ndarray,
    y: np.ndarray = None,
    feature_masks: Dict[str, np.ndarray] = None,
    components: List[Any] = None,
    sample_idx: int = 0,
    components_to_include: List[str] = None,
    line_color: str = "black",
    rect_fill: str = "red",
    panel_width: int = 225,
    panel_height: int = 175,
) -> Any:
    """Plot a time series sample with its components and feature masks using lets_plot.

    Args:
        X (np.ndarray): Time series data.
        y (Optional[np.ndarray]): Class labels.
        feature_masks (Optional[Dict[str, np.ndarray]]): Dictionary of feature masks.
        components (Optional[List[Any]]): List of TimeSeriesComponents objects.
        sample_idx (int): Index of the sample to plot.
        components_to_include (Optional[List[str]]): List of components to include.
        line_color (str): Color of the time series lines.
        rect_fill (str): Fill color for feature rectangles.
        panel_width (int): Width of each panel in pixels.
        panel_height (int): Height of each panel in pixels.

    Returns:
        Any: lets_plot visualization.
    """
    # Prepare dataset in the expected format
    dataset = {"X": X, "components": components}

    if y is not None:
        dataset["y"] = y
    else:
        # Create dummy labels if not provided (must be int for class label formatting)
        dataset["y"] = np.zeros(len(X), dtype=int)

    if feature_masks is not None:
        dataset["feature_masks"] = feature_masks

    # Set metadata for format detection - assume channels_first if shape is correct
    if len(X.shape) == 3 and X.shape[1] < X.shape[2]:  # Likely channels_first
        dataset["metadata"] = {"data_format": "channels_first"}

    # Set up sample indices
    class_label = dataset["y"][sample_idx]
    sample_indices = {class_label: sample_idx}

    # Create and return plot
    return plot_components(
        dataset,
        sample_indices=sample_indices,
        components=components_to_include,
        line_color=line_color,
        rect_fill=rect_fill,
        panel_width=panel_width,
        panel_height=panel_height,
    )
