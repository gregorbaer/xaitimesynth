"""
XAITimeSynth - A package for generating synthetic time series with ground truth for XAI evaluation.

This package provides a fluent, composable API for creating time series datasets
with precise control over signal components, noise, and discriminative features.

Example usage:
    ```python
    TODO: Add example usage here once API is stable.
    ```
"""

from .builder import TimeSeriesBuilder
from .components import (
    constant,
    ecg_like,
    gaussian,
    gaussian_pulse,
    manual,
    peak,
    random_walk,
    red_noise,
    seasonal,
    trend,
    trough,
    uniform,
)
from .data_structures import TimeSeriesComponents
from .functions import normalize
from .parser import load_builders_from_config
from .registry import (
    get_component_parameters,
    list_components,
    list_feature_components,
    list_signal_components,
    register_component,
    register_component_generator,
)
from .visualization import (
    plot_class_comparison,
    plot_component,
    plot_components,
    plot_sample,
)

# Export all the imported names
__all__ = [
    # Classes and Functions
    "TimeSeriesBuilder",
    "load_builders_from_config",
    "TimeSeriesComponents",
    "normalize",
    "get_component_parameters",
    "list_components",
    "list_feature_components",
    "list_signal_components",
    "register_component",
    "register_component_generator",
    # Visualization Functions
    "plot_class_comparison",
    "plot_component",
    "plot_components",
    "plot_sample",
    "plot_timeseries",
    # Data Generation Components
    "constant",
    "ecg_like",
    "gaussian",
    "gaussian_pulse",
    "manual",
    "peak",
    "random_walk",
    "red_noise",
    "seasonal",
    "trend",
    "trough",
    "uniform",
]

# Register standard components
register_component(constant, "signal")
register_component(random_walk, "signal")
register_component(gaussian, "signal")
register_component(uniform, "signal")
register_component(seasonal, "signal")
register_component(ecg_like, "signal")
register_component(red_noise, "signal")

register_component(trend, "feature")
register_component(peak, "feature")
register_component(trough, "feature")
register_component(gaussian_pulse, "feature")

register_component(manual, "both")

# Version
__version__ = "0.1.0"
