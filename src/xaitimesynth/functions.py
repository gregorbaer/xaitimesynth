from typing import Any, Dict, Optional


class SignalAdder:
    """Helper class for adding signal components."""

    def __init__(self, component: Dict[str, Any], role: str = "base_structure"):
        """Initialize the signal adder.

        Args:
            component: Component definition dictionary.
            role: Role of the component (base_structure, noise).
        """
        self.component = component
        self.role = role

    def __call__(self, builder):
        """Add the component to the builder.

        Args:
            builder: TimeSeriesBuilder instance.

        Returns:
            The builder for method chaining.
        """
        return builder.add_signal_component(self.component, role=self.role)


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


def add_signal(component: Dict[str, Any], role: str = "base_structure") -> SignalAdder:
    """Add a component as a global signal.

    Args:
        component: Component definition dictionary.
        role: Role of the component (base_structure, noise).

    Returns:
        SignalAdder instance.
    """
    return SignalAdder(component, role)


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
