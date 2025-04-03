from typing import Any, Dict, Optional, Tuple

import numpy as np


class SignalAdder:
    """Helper class for adding signal components."""

    def __init__(
        self,
        component: Dict[str, Any],
        role: str = "foundation",
        start_pct: Optional[float] = None,
        end_pct: Optional[float] = None,
        length_pct: Optional[float] = None,
        random_location: bool = False,
        shared_location: bool = True,
    ):
        """Initialize the signal adder.

        Args:
            component: Component definition dictionary.
            role: Role of the component (foundation, noise).
            start_pct: Start position as percentage of time series length (0-1).
            end_pct: End position as percentage of time series length (0-1).
            length_pct: Length of feature as percentage of time series length (0-1).
            random_location: Whether to place the signal at a random location.
            shared_location: If True and random_location is True, the same random
                location will be used across all dimensions. Default is True.
        """
        self.component = component
        self.role = role
        self.start_pct = start_pct
        self.end_pct = end_pct
        self.length_pct = length_pct
        self.random_location = random_location
        self.shared_location = shared_location

    def __call__(self, builder):
        """Add the component to the builder.

        Args:
            builder: TimeSeriesBuilder instance.

        Returns:
            The builder for method chaining.
        """
        return builder.add_signal_component(
            self.component,
            role=self.role,
            start_pct=self.start_pct,
            end_pct=self.end_pct,
            length_pct=self.length_pct,
            random_location=self.random_location,
            shared_location=self.shared_location,
        )


class FeatureAdder:
    """Helper class for adding feature components."""

    def __init__(
        self,
        component: Dict[str, Any],
        start_pct: Optional[float] = None,
        end_pct: Optional[float] = None,
        length_pct: Optional[float] = None,
        random_location: bool = False,
    ):
        """Initialize the feature adder.

        Args:
            component: Component definition dictionary.
            start_pct: Start position as percentage of time series length (0-1).
            end_pct: End position as percentage of time series length (0-1).
            length_pct: Length of feature as percentage of time series length (0-1).
            random_location: Whether to place the feature at a random location.
        """
        self.component = component
        self.start_pct = start_pct
        self.end_pct = end_pct
        self.length_pct = length_pct
        self.random_location = random_location

    def __call__(self, builder):
        """Add the component to the builder.

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
    role: str = "foundation",
    start_pct: Optional[float] = None,
    end_pct: Optional[float] = None,
    length_pct: Optional[float] = None,
    random_location: bool = False,
    shared_location: bool = True,
) -> SignalAdder:
    """Add a component as a global signal.

    By default, signals are applied to the entire time series. Optional time range
    parameters can be specified to limit where the signal appears.

    Args:
        component: Component definition dictionary.
        role: Role of the component (foundation, noise).
        start_pct: Start position as percentage of time series length (0-1).
        end_pct: End position as percentage of time series length (0-1).
        length_pct: Length of signal as percentage of time series length (0-1).
        random_location: Whether to place the signal at a random location.
        shared_location: If True and random_location is True, the same random
            location will be used across all dimensions. Default is True.

    Returns:
        SignalAdder instance.
    """
    return SignalAdder(
        component,
        role=role,
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
    """Add a component as a localized feature.

    Args:
        component: Component definition dictionary.
        start_pct: Start position as percentage of time series length (0-1).
        end_pct: End position as percentage of time series length (0-1).
        length_pct: Length of feature as percentage of time series length (0-1).
        random_location: Whether to place the feature at a random location.

    Returns:
        FeatureAdder instance.
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

    Args:
        data: Input array to normalize
        feature_range: The desired range of transformed data, defaults to (0, 1)

    Returns:
        Normalized data with values scaled to given range
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

    Args:
        data: Input array to normalize
        epsilon: Small constant to avoid division by zero

    Returns:
        Normalized data with zero mean and unit variance
    """
    mean = np.mean(data)
    std = np.std(data)
    if std > epsilon:
        return (data - mean) / std
    return data


def normalize(data: np.ndarray, method: str = "zscore", **kwargs) -> np.ndarray:
    """Normalize data using specified method.

    Args:
        data: Input array to normalize
        method: Normalization method ("zscore", "minmax", or "none")
        **kwargs: Additional parameters for specific normalization methods

    Returns:
        Normalized data according to specified method

    Raises:
        ValueError: If an invalid normalization method is specified
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
