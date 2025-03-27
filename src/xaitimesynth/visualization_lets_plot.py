import numpy as np
import pandas as pd
from lets_plot import *


def create_visualization_data(dataset, sample_indices=None, components_to_include=None):
    """Create data for visualization with lets_plot.

    Args:
        dataset: Dataset containing time series data and components.
        sample_indices: Dictionary mapping class labels to sample indices.
            If None, use first sample of each class.
        components_to_include: List of components to include. If None, include all.

    Returns:
        Prepared DataFrame for visualization.
    """
    # Determine sample indices if not provided
    if sample_indices is None:
        sample_indices = {}
        for class_label in np.unique(dataset["y"]):
            sample_indices[class_label] = np.where(dataset["y"] == class_label)[0][0]

    # Create main time series data
    data_rows = []

    # Define default components and their order
    default_components = ["Full Series", "Features", "Base Structure", "Noise"]

    # Use specified components if provided, otherwise use defaults
    components = (
        components_to_include
        if components_to_include is not None
        else default_components
    )

    # Process each class
    for class_label, idx in sample_indices.items():
        # Get component for this sample
        comp = dataset["components"][idx]

        # Get time series length
        n_timesteps = len(dataset["X"][idx])

        # Add data for each component in the specified order
        for component_name in components:
            if component_name == "Full Series" and idx < len(dataset["X"]):
                for t, val in enumerate(dataset["X"][idx]):
                    data_rows.append(
                        {
                            "time": float(t),
                            "value": float(val),
                            "class": f"Class {class_label}",
                            "component": component_name,
                        }
                    )

            elif component_name == "Features" and hasattr(comp, "features"):
                if not comp.features:
                    # Empty features
                    for t in range(n_timesteps):
                        data_rows.append(
                            {
                                "time": float(t),
                                "value": 0.0,
                                "class": f"Class {class_label}",
                                "component": component_name,
                            }
                        )
                else:
                    # Combine all features
                    combined_features = np.zeros(n_timesteps)
                    for name, feature in comp.features.items():
                        combined_features += feature

                    for t, val in enumerate(combined_features):
                        data_rows.append(
                            {
                                "time": float(t),
                                "value": float(val),
                                "class": f"Class {class_label}",
                                "component": component_name,
                            }
                        )

            elif component_name == "Base Structure" and hasattr(comp, "base_structure"):
                for t, val in enumerate(comp.base_structure):
                    data_rows.append(
                        {
                            "time": float(t),
                            "value": float(val),
                            "class": f"Class {class_label}",
                            "component": component_name,
                        }
                    )

            elif (
                component_name == "Noise"
                and hasattr(comp, "noise")
                and comp.noise is not None
            ):
                for t, val in enumerate(comp.noise):
                    data_rows.append(
                        {
                            "time": float(t),
                            "value": float(val),
                            "class": f"Class {class_label}",
                            "component": component_name,
                        }
                    )

    # Create DataFrame
    df = pd.DataFrame(data_rows)

    # No data after filtering
    if len(df) == 0:
        return df

    # Ensure component order matches the input order
    # Only include components that actually exist in the data
    available_components = [c for c in components if c in df["component"].unique()]
    df["component"] = pd.Categorical(
        df["component"], categories=available_components, ordered=True
    )

    # Ensure class order is numeric
    class_labels = sorted(df["class"].unique(), key=lambda x: int(x.split()[-1]))
    df["class"] = pd.Categorical(df["class"], categories=class_labels, ordered=True)

    return df


def create_feature_rectangles(dataset, sample_indices=None, components_to_include=None):
    """Create rectangle data for feature visualization.

    Args:
        dataset: Dataset containing time series data and components.
        sample_indices: Dictionary mapping class labels to sample indices.
            If None, use first sample of each class.
        components_to_include: List of components to include. If None, include all.

    Returns:
        DataFrame with rectangle coordinates for feature regions.
    """
    # If components_to_include doesn't have Full Series, no need for rectangles
    if components_to_include is not None and "Full Series" not in components_to_include:
        return None

    # Determine sample indices if not provided
    if sample_indices is None:
        sample_indices = {}
        for class_label in np.unique(dataset["y"]):
            sample_indices[class_label] = np.where(dataset["y"] == class_label)[0][0]

    rectangles = []

    # Process each class and sample
    for class_label, idx in sample_indices.items():
        # Extract feature masks for this sample
        if "feature_masks" in dataset:
            for key, mask in dataset["feature_masks"].items():
                # Check if mask is for this class
                if f"class_{class_label}_" in key:
                    sample_mask = mask[idx]

                    # Find contiguous regions in the mask
                    if np.any(sample_mask):
                        changes = np.diff(
                            np.concatenate([[False], sample_mask, [False]]).astype(int)
                        )
                        start_indices = np.where(changes == 1)[0]
                        end_indices = np.where(changes == -1)[0]

                        # Extract feature name from key
                        feature_name = key.replace(f"class_{class_label}_", "")

                        # Create rectangle for each region
                        for start, end in zip(start_indices, end_indices):
                            rectangles.append(
                                {
                                    "class": f"Class {class_label}",
                                    "component": "Full Series",
                                    "feature": feature_name,
                                    "xmin": float(start),
                                    "xmax": float(end),
                                }
                            )

        # If no rectangles found, try to extract from components
        if not any(r["class"] == f"Class {class_label}" for r in rectangles):
            if "components" in dataset:
                comp = dataset["components"][idx]

                # Try feature masks first
                if hasattr(comp, "feature_masks") and comp.feature_masks:
                    for feature_name, feature_mask in comp.feature_masks.items():
                        if np.any(feature_mask):
                            changes = np.diff(
                                np.concatenate([[False], feature_mask, [False]]).astype(
                                    int
                                )
                            )
                            start_indices = np.where(changes == 1)[0]
                            end_indices = np.where(changes == -1)[0]

                            for start, end in zip(start_indices, end_indices):
                                rectangles.append(
                                    {
                                        "class": f"Class {class_label}",
                                        "component": "Full Series",
                                        "feature": feature_name,
                                        "xmin": float(start),
                                        "xmax": float(end),
                                    }
                                )

                # If still no rectangles, try feature values
                if not any(r["class"] == f"Class {class_label}" for r in rectangles):
                    if hasattr(comp, "features") and comp.features:
                        for feature_name, feature_values in comp.features.items():
                            # Find where the feature has non-zero values
                            non_zero = np.abs(feature_values) > 1e-6
                            if np.any(non_zero):
                                changes = np.diff(
                                    np.concatenate([[False], non_zero, [False]]).astype(
                                        int
                                    )
                                )
                                start_indices = np.where(changes == 1)[0]
                                end_indices = np.where(changes == -1)[0]

                                for start, end in zip(start_indices, end_indices):
                                    rectangles.append(
                                        {
                                            "class": f"Class {class_label}",
                                            "component": "Full Series",
                                            "feature": feature_name,
                                            "xmin": float(start),
                                            "xmax": float(end),
                                        }
                                    )

    if not rectangles:
        return None

    rect_df = pd.DataFrame(rectangles)

    # Ensure correct class ordering
    class_labels = sorted(rect_df["class"].unique(), key=lambda x: int(x.split()[-1]))
    rect_df["class"] = pd.Categorical(
        rect_df["class"], categories=class_labels, ordered=True
    )

    return rect_df


def create_ts_visualization(
    dataset,
    sample_indices=None,
    components=None,
    show_indicators=True,
    line_color="black",
    line_size=1.5,
    rect_fill="red",
    rect_alpha=0.25,
    facet_order={"y": "class", "x": "component"},
    free_y=False,
    single_row=False,
    panel_width=225,
    panel_height=175,
):
    """Create time series visualization with feature indicators as rectangles.

    Args:
        dataset: Dataset containing time series data and components.
        sample_indices: Dictionary mapping class labels to sample indices.
            If None, use first sample of each class.
        components: List of components to include and their order.
            Default: ["Full Series", "Features", "Base Structure", "Noise"]
        show_indicators: Whether to show feature indicators.
        line_color: Color of the time series lines ("black" or "auto" for colored by class).
        line_size: Size of the time series lines.
        rect_fill: Fill color for feature rectangles.
        rect_alpha: Alpha transparency for feature rectangles.
        facet_order: Order of facets, dict with "x" and "y" keys.
        free_y: Whether to use free y scales in facets (default=False).
        single_row: Display all classes in a single row, using colors (default=False).
        panel_width: Width of each panel in pixels.
        panel_height: Height of each panel in pixels.

    Returns:
        lets_plot visualization.
    """
    # Default component order
    default_components = ["Full Series", "Features", "Base Structure", "Noise"]
    components_to_use = components if components is not None else default_components

    # Prepare data for visualization
    df = create_visualization_data(dataset, sample_indices, components_to_use)

    # If no data, return empty plot
    if len(df) == 0:
        return (
            ggplot()
            + ggsize(panel_width * 4, panel_height * 2)
            + labs(title="No data to display")
        )

    # Create feature rectangles if needed
    rectangles = None
    if show_indicators:
        rectangles = create_feature_rectangles(
            dataset, sample_indices, components_to_use
        )

    # Calculate global y-limits
    y_min = df["value"].min()
    y_max = df["value"].max()
    padding = (y_max - y_min) * 0.15
    y_min = float(y_min - padding)
    y_max = float(y_max + padding)

    # Calculate plot dimensions based on the number of panels
    if single_row:
        n_components = len(df["component"].unique())
        total_width = panel_width * n_components
        total_height = panel_height
    else:
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
    if single_row:
        # In single_row mode, always use colors by class
        p = ggplot(df, aes(x="time", y="value", color="class"))
        line_color = "auto"  # Force auto color
    elif line_color == "auto":
        # Use colors by class
        p = ggplot(df, aes(x="time", y="value", color="class"))
    else:
        # Use specified line color
        p = ggplot(df, aes(x="time", y="value"))

    # Add the rectangles for feature regions
    if show_indicators and rectangles is not None:
        # Set rectangle height to full panel height
        rectangles = rectangles.copy()
        rectangles["ymin"] = y_min
        rectangles["ymax"] = y_max

        # Add rectangles to the plot
        if line_color == "auto":
            # Color rectangles by feature
            p = p + geom_rect(
                data=rectangles,
                mapping=aes(
                    xmin="xmin", xmax="xmax", ymin="ymin", ymax="ymax", fill="feature"
                ),
                alpha=rect_alpha,
            )
        else:
            # Use specified rectangle color
            p = p + geom_rect(
                data=rectangles,
                mapping=aes(xmin="xmin", xmax="xmax", ymin="ymin", ymax="ymax"),
                fill=rect_fill,
                alpha=rect_alpha,
            )

    # Complete the plot
    if line_color == "auto":
        p = p + geom_line(size=line_size)
    else:
        p = p + geom_line(size=line_size, color=line_color)

    p = p + geom_hline(yintercept=0, linetype="dashed", color="gray")

    # Handle single row mode
    if single_row:
        # In single row mode, only facet by component
        p = p + facet_wrap("component", scales="free_y" if free_y else "fixed")
    else:
        # Use regular facet grid with proper ordering
        scales = "free_y" if free_y else "fixed"

        # Ensure we use proper ordering for components and classes
        # The x_order and y_order parameters force the order of factors
        p = p + facet_grid(**facet_order, scales=scales)

    if not free_y:
        p = p + scale_y_continuous(limits=[y_min, y_max])

    p = p + labs(x="Time", y="Value")
    p = p + theme_bw()
    p = p + ggsize(total_width, total_height)

    return p


def plot_sample_lets_plot(
    X,
    y=None,
    feature_masks=None,
    components=None,
    sample_idx=0,
    components_to_include=None,
    line_color="black",
    rect_fill="red",
    free_y=False,
    panel_width=225,
    panel_height=175,
):
    """Plot a time series sample with its components and feature masks using lets_plot.

    Args:
        X: Time series data.
        y: Class labels.
        feature_masks: Dictionary of feature masks.
        components: List of TimeSeriesComponents objects.
        sample_idx: Index of the sample to plot.
        components_to_include: List of components to include.
        line_color: Color of the time series lines ("black" or "auto" for colored by class).
        rect_fill: Fill color for feature rectangles.
        free_y: Whether to use free y scales in facets.
        panel_width: Width of each panel in pixels.
        panel_height: Height of each panel in pixels.

    Returns:
        lets_plot visualization.
    """
    # Prepare dataset in the expected format
    dataset = {"X": X, "components": components}

    if y is not None:
        dataset["y"] = y
    else:
        # Create dummy labels if not provided
        dataset["y"] = np.zeros(len(X))

    if feature_masks is not None:
        dataset["feature_masks"] = feature_masks

    # Set up sample indices
    class_label = dataset["y"][sample_idx]
    sample_indices = {class_label: sample_idx}

    # Create and return plot
    return create_ts_visualization(
        dataset,
        sample_indices=sample_indices,
        components=components_to_include,
        line_color=line_color,
        rect_fill=rect_fill,
        free_y=free_y,
        panel_width=panel_width,
        panel_height=panel_height,
    )


def plot_class_comparison(
    dataset,
    components_to_include=None,
    line_color="auto",
    rect_fill="red",
    free_y=False,
    single_row=False,
    panel_width=225,
    panel_height=175,
):
    """Plot comparison of classes with feature indicators using lets_plot.

    Args:
        dataset: Dataset containing time series data and components.
        components_to_include: List of components to include.
        line_color: Color of the time series lines ("black" or "auto" for colored by class).
        rect_fill: Fill color for feature rectangles.
        free_y: Whether to use free y scales in facets.
        single_row: Display all classes in a single row, using colors.
        panel_width: Width of each panel in pixels.
        panel_height: Height of each panel in pixels.

    Returns:
        lets_plot visualization of class comparison.
    """
    # Get first sample from each class
    sample_indices = {}
    for class_label in np.unique(dataset["y"]):
        sample_indices[class_label] = np.where(dataset["y"] == class_label)[0][0]

    # Default to only showing "Full Series" for class comparison
    if components_to_include is None:
        components_to_include = ["Full Series"]

    # Create and return visualization
    return create_ts_visualization(
        dataset,
        sample_indices=sample_indices,
        components=components_to_include,
        show_indicators=True,
        line_color=line_color,
        rect_fill=rect_fill,
        free_y=free_y,
        single_row=single_row,
        panel_width=panel_width,
        panel_height=panel_height,
    )
