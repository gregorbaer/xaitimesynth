from typing import Any, Dict, Optional, Tuple

import numpy as np


class SignalAdder:
    """Helper class for adding signal components to a TimeSeriesBuilder.

    This class encapsulates the parameters needed to add a signal component
    to a time series being built using the TimeSeriesBuilder. It allows for
    specifying the component and its temporal placement within the time series.
    """

    def __init__(
        self,
        component: Dict[str, Any],
        start_pct: Optional[float] = None,
        end_pct: Optional[float] = None,
        length_pct: Optional[float] = None,
        random_location: bool = False,
        shared_location: bool = True,
    ):
        """Initialize the SignalAdder.

        Args:
            component (Dict[str, Any]): Component definition dictionary.
            start_pct (Optional[float]): Start position as percentage of time series length (0-1). Defaults to None.
            end_pct (Optional[float]): End position as percentage of time series length (0-1). Defaults to None.
            length_pct (Optional[float]): Length of feature as percentage of time series length (0-1). Defaults to None.
            random_location (bool): Whether to place the signal at a random location. Defaults to False.
            shared_location (bool): If True and random_location is True, the same random
                location will be used across all dimensions. Default is True.
        """
        self.component = component
        self.start_pct = start_pct
        self.end_pct = end_pct
        self.length_pct = length_pct
        self.random_location = random_location
        self.shared_location = shared_location

    def __call__(self, builder):
        """Add the component to the builder.

        This method is called when the SignalAdder instance is called as a function.
        It adds the signal to the TimeSeriesBuilder instance.

        Args:
            builder: TimeSeriesBuilder instance.

        Returns:
            The builder for method chaining.
        """
        return builder.add_signal(
            self.component,
            start_pct=self.start_pct,
            end_pct=self.end_pct,
            length_pct=self.length_pct,
            random_location=self.random_location,
            shared_location=self.shared_location,
        )


class FeatureAdder:
    """Helper class for adding feature components to a TimeSeriesBuilder.

    This class encapsulates the parameters needed to add a feature component
    to a time series being built using the TimeSeriesBuilder. It allows for
    specifying the component and its temporal placement within the time series.
    """

    def __init__(
        self,
        component: Dict[str, Any],
        start_pct: Optional[float] = None,
        end_pct: Optional[float] = None,
        length_pct: Optional[float] = None,
        random_location: bool = False,
    ):
        """Initialize the FeatureAdder.

        Args:
            component (Dict[str, Any]): Component definition dictionary.
            start_pct (Optional[float]): Start position as percentage of time series length (0-1). Defaults to None.
            end_pct (Optional[float]): End position as percentage of time series length (0-1). Defaults to None.
            length_pct (Optional[float]): Length of feature as percentage of time series length (0-1). Defaults to None.
            random_location (bool): Whether to place the feature at a random location. Defaults to False.
        """
        self.component = component
        self.start_pct = start_pct
        self.end_pct = end_pct
        self.length_pct = length_pct
        self.random_location = random_location

    def __call__(self, builder):
        """Add the component to the builder.

        This method is called when the FeatureAdder instance is called as a function.
        It adds the feature component to the TimeSeriesBuilder instance.

        Args:
            builder: TimeSeriesBuilder instance.

        Returns:
            The builder for method chaining.
        """
        return builder.add_feature_component(
            self.component,
            start_pct=self.start_pct,
            end_pct=self.end_pct,
            length_pct=self.length_pct,
            random_location=self.random_location,
        )


def add_signal(
    component: Dict[str, Any],
    start_pct: Optional[float] = None,
    end_pct: Optional[float] = None,
    length_pct: Optional[float] = None,
    random_location: bool = False,
    shared_location: bool = True,
) -> SignalAdder:
    """Add a component as a global signal using a SignalAdder.

    This function creates a SignalAdder instance with the specified parameters,
    which can then be used to add a signal component to a TimeSeriesBuilder.
    By default, signals are applied to the entire time series. Optional time range
    parameters can be specified to limit where the signal appears.

    Args:
        component (Dict[str, Any]): Component definition dictionary.
        start_pct (Optional[float]): Start position as percentage of time series length (0-1). Defaults to None.
        end_pct (Optional[float]): End position as percentage of time series length (0-1). Defaults to None.
        length_pct (Optional[float]): Length of signal as percentage of time series length (0-1). Defaults to None.
        random_location (bool): Whether to place the signal at a random location. Defaults to False.
        shared_location (bool): If True and random_location is True, the same random
            location will be used across all dimensions. Default is True.

    Returns:
        SignalAdder: SignalAdder instance.
    """
    return SignalAdder(
        component,
        start_pct=start_pct,
        end_pct=end_pct,
        length_pct=length_pct,
        random_location=random_location,
        shared_location=shared_location,
    )


def add_feature(
    component: Dict[str, Any],
    start_pct: Optional[float] = None,
    end_pct: Optional[float] = None,
    length_pct: Optional[float] = None,
    random_location: bool = False,
) -> FeatureAdder:
    """Add a component as a localized feature using a FeatureAdder.

    This function creates a FeatureAdder instance with the specified parameters,
    which can then be used to add a feature component to a TimeSeriesBuilder.

    Args:
        component (Dict[str, Any]): Component definition dictionary.
        start_pct (Optional[float]): Start position as percentage of time series length (0-1). Defaults to None.
        end_pct (Optional[float]): End position as percentage of time series length (0-1). Defaults to None.
        length_pct (Optional[float]): Length of feature as percentage of time series length (0-1). Defaults to None.
        random_location (bool): Whether to place the feature at a random location. Defaults to False.

    Returns:
        FeatureAdder: FeatureAdder instance.
    """
    return FeatureAdder(
        component,
        start_pct=start_pct,
        end_pct=end_pct,
        length_pct=length_pct,
        random_location=random_location,
    )


def minmax_normalize(
    data: np.ndarray, feature_range: Tuple[float, float] = (0, 1)
) -> np.ndarray:
    """Apply min-max scaling to data.

    Scales the input data to a specified range using the min-max normalization method.

    Args:
        data (np.ndarray): Input array to normalize.
        feature_range (Tuple[float, float]): The desired range of transformed data, defaults to (0, 1).

    Returns:
        np.ndarray: Normalized data with values scaled to given range.
    """
    min_val = np.min(data)
    max_val = np.max(data)
    if max_val > min_val:
        return (data - min_val) / (max_val - min_val) * (
            feature_range[1] - feature_range[0]
        ) + feature_range[0]
    return data  # Return original if max == min (constant array)


def zscore_normalize(data: np.ndarray, epsilon: float = 1e-10) -> np.ndarray:
    """Apply z-score normalization to data.

    Standardizes the input data by subtracting the mean and dividing by the standard deviation.

    Args:
        data (np.ndarray): Input array to normalize.
        epsilon (float): Small constant to avoid division by zero. Defaults to 1e-10.

    Returns:
        np.ndarray: Normalized data with zero mean and unit variance.
    """
    mean = np.mean(data)
    std = np.std(data)
    if std > epsilon:
        return (data - mean) / std
    return data


def normalize(data: np.ndarray, method: str = "zscore", **kwargs) -> np.ndarray:
    """Normalize data using specified method.

    Applies a normalization method to the input data based on the specified method.
    Supports 'zscore' (standardization), 'minmax' (min-max scaling), and 'none' (no normalization).

    Args:
        data (np.ndarray): Input array to normalize.
        method (str): Normalization method ("zscore", "minmax", or "none"). Defaults to "zscore".
        **kwargs: Additional parameters for specific normalization methods.

    Returns:
        np.ndarray: Normalized data according to specified method.

    Raises:
        ValueError: If an invalid normalization method is specified.
    """
    if method == "minmax":
        feature_range = kwargs.get("feature_range", (0, 1))
        return minmax_normalize(data, feature_range)
    elif method == "zscore":
        epsilon = kwargs.get("epsilon", 1e-10)
        return zscore_normalize(data, epsilon)
    elif method == "none":
        return data
    else:
        raise ValueError(
            f"Invalid normalization method: {method}. "
            "Choose 'zscore', 'minmax', or 'none'."
        )
