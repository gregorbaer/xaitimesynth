"""
XAITimeSynth - A package for generating synthetic time series with ground truth for XAI evaluation.

This package provides a fluent, composable API for creating time series datasets
with precise control over signal components, noise, and discriminative features.

Example usage:
    ```python
    from xaitimesynth import (
        TimeSeriesBuilder,
        constant, random_walk, autoregressive, gaussian, uniform, seasonal,
        shapelet, level_change, trend, peak, trough, time_frequency, manual
    )

    # Create a dataset with two classes
    dataset = (
        TimeSeriesBuilder(n_timesteps=100, n_samples=1000)
        .for_class(0)
        .add_signal(random_walk(step_size=0.2))
        .add_signal(gaussian(sigma=0.1), role="noise")
        .for_class(1)
        .add_signal(random_walk(step_size=0.2))
        .add_signal(gaussian(sigma=0.1), role="noise")
        .add_feature(shapelet(scale=1.0), start_pct=0.4, end_pct=0.6)
        .add_feature(level_change(amplitude=0.5), start_pct=0.7, end_pct=0.9)
        .build(train_test_split=0.7)
    )
    ```
"""

from .builder import TimeSeriesBuilder
from .components import (
    autoregressive,
    constant,
    ecg_like,
    gaussian,
    level_change,
    manual,
    peak,
    random_walk,
    seasonal,
    shapelet,
    time_frequency,
    trend,
    trough,
    uniform,
)
from .data_structures import TimeSeriesComponents
from .functions import normalize
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

__all__ = [  # Export all the imported names
    "TimeSeriesBuilder",
    "autoregressive",
    "constant",
    "ecg_like",
    "gaussian",
    "level_change",
    "manual",
    "peak",
    "random_walk",
    "seasonal",
    "shapelet",
    "time_frequency",
    "trend",
    "trough",
    "uniform",
    "TimeSeriesComponents",
    "normalize",
    "get_component_parameters",
    "list_components",
    "list_feature_components",
    "list_signal_components",
    "register_component",
    "register_component_generator",
    "plot_class_comparison",
    "plot_component",
    "plot_components",
    "plot_sample",
    "plot_timeseries",
]

# Register standard components
register_component(constant, "signal")
register_component(random_walk, "signal")
register_component(autoregressive, "signal")
register_component(gaussian, "signal")
register_component(uniform, "signal")
register_component(seasonal, "signal")
register_component(ecg_like, "signal")

register_component(shapelet, "feature")
register_component(level_change, "feature")
register_component(trend, "feature")
register_component(peak, "feature")
register_component(trough, "feature")
register_component(time_frequency, "feature")

register_component(manual, "both")

# Version
__version__ = "0.1.0"
