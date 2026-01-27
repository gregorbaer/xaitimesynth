# Usage Guide

This guide provides a detailed walkthrough of the xaitimesynth API with examples for common use cases.

## Table of Contents

1. [Basic Usage](#basic-usage)
2. [Builder Parameters](#builder-parameters)
3. [Adding Signals](#adding-signals)
4. [Adding Features](#adding-features)
5. [Positioning Features and Signals](#positioning-features-and-signals)
6. [Multivariate Time Series](#multivariate-time-series)
7. [Creating Data Splits](#creating-data-splits)
8. [YAML Configuration](#yaml-configuration)
9. [Custom Components](#custom-components)
10. [Accessing Component Data](#accessing-component-data)

## Basic Usage

The core workflow is: create a builder, define classes with signals and features, then build.

```python
from xaitimesynth import TimeSeriesBuilder, gaussian, peak

# Create a simple two-class dataset
dataset = (
    TimeSeriesBuilder(n_timesteps=100, n_samples=200)
    .for_class(0)
    .add_signal(gaussian(sigma=0.5))
    .for_class(1)
    .add_signal(gaussian(sigma=0.5))
    .add_feature(peak(amplitude=2.0, width=10), start_pct=0.3, end_pct=0.7)
    .build()
)

# Access the results
X = dataset["X"]              # (200, 1, 100) - samples x dims x timesteps
y = dataset["y"]              # (200,) - class labels
masks = dataset["feature_masks"]  # (200, 1, 100) - ground truth locations
components = dataset["components"]  # List of TimeSeriesComponents objects
```

## Builder Parameters

`TimeSeriesBuilder` accepts these parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_timesteps` | 100 | Length of each time series |
| `n_samples` | 1000 | Number of time series to generate |
| `n_dimensions` | 1 | Number of channels (1 = univariate) |
| `random_state` | None | Seed for reproducibility |
| `normalization` | "zscore" | Normalization method: "zscore", "minmax", or "none" |
| `normalization_kwargs` | {} | Extra args for normalization (e.g., feature_range for minmax) |
| `data_format` | "channels_first" | Output format: "channels_first" (N,D,T) or "channels_last" (N,T,D) |
| `feature_fill_value` | np.nan | Fill value for feature component outside feature window |
| `foundation_fill_value` | 0.0 | Fill value for foundation component |
| `noise_fill_value` | np.nan | Fill value for noise component |

### Normalization Options

```python
# Z-score normalization (default): mean=0, std=1
builder = TimeSeriesBuilder(normalization="zscore")

# Min-max normalization to [0, 1]
builder = TimeSeriesBuilder(normalization="minmax")

# Min-max to custom range
builder = TimeSeriesBuilder(
    normalization="minmax",
    normalization_kwargs={"feature_range": (-1, 1)}
)

# No normalization
builder = TimeSeriesBuilder(normalization="none")
```

### Data Format

```python
# Default: channels first (N, D, T)
dataset = TimeSeriesBuilder(n_dimensions=3, data_format="channels_first").build()
print(dataset["X"].shape)  # (n_samples, 3, n_timesteps)

# Alternative: channels last (N, T, D)
dataset = TimeSeriesBuilder(n_dimensions=3, data_format="channels_last").build()
print(dataset["X"].shape)  # (n_samples, n_timesteps, 3)
```

## Adding Signals

Signals are full-length background patterns. Use `add_signal()` to add them.

```python
from xaitimesynth import random_walk, gaussian, seasonal, trend, red_noise

builder = (
    TimeSeriesBuilder(n_timesteps=200)
    .for_class(0)
    # Foundation signal: the base pattern
    .add_signal(random_walk(step_size=0.2), role="foundation")
    # Noise signal: stochastic variation
    .add_signal(gaussian(sigma=0.1), role="noise")
    # Multiple signals are additive
    .add_signal(seasonal(period=20, amplitude=0.5), role="foundation")
)
```

### Available Signal Components

| Component | Parameters | Description |
|-----------|------------|-------------|
| `random_walk` | `step_size=0.1` | Cumulative sum of random steps |
| `gaussian` | `mu=0, sigma=1` | White noise from normal distribution |
| `uniform` | `low=-1, high=1` | White noise from uniform distribution |
| `seasonal` | `period=10, amplitude=1.0, phase=0` | Periodic sinusoidal pattern |
| `trend` | `endpoints=(0, 1)` | Linear trend from start to end value |
| `red_noise` | `phi=0.9, sigma=1.0` | AR(1) correlated noise |
| `ecg_like` | `num_heartbeats=5` | Simulated ECG pattern |
| `constant` | `value=1.0` | Constant value |

### Signal Roles

The `role` parameter determines how the signal is categorized internally:

```python
# Foundation: the base pattern the time series is built on
builder.add_signal(random_walk(step_size=0.2), role="foundation")

# Noise: stochastic variation (default if not specified is "foundation")
builder.add_signal(gaussian(sigma=0.1), role="noise")
```

This categorization is stored in `TimeSeriesComponents` and can be useful for visualization and analysis.

## Adding Features

Features are localized discriminative patterns. Use `add_feature()` to add them.

```python
from xaitimesynth import peak, trough, constant, gaussian_pulse

builder = (
    TimeSeriesBuilder(n_timesteps=100)
    .for_class(1)
    .add_signal(gaussian(sigma=0.5))
    # Fixed position: 30% to 60% of the time series (30 timesteps)
    .add_feature(peak(amplitude=2.0, width=5), start_pct=0.3, end_pct=0.6)
    # Random position: feature takes up 15% of length, placed randomly
    .add_feature(constant(value=1.0), length_pct=0.15, random_location=True)
)
```

### Available Feature Components

| Component | Parameters | Description |
|-----------|------------|-------------|
| `peak` | `amplitude=1.0, width=10` | Gaussian-shaped peak |
| `trough` | `amplitude=1.0, width=10` | Inverted peak (downward) |
| `gaussian_pulse` | `amplitude=1.0, width_ratio=0.5, center=0.5` | Sharp pulse |
| `constant` | `value=1.0` | Level shift |
| `trend` | `endpoints=(0, 1)` | Linear ramp |

## Positioning Features and Signals

### Fixed Position

Use `start_pct` and `end_pct` together to place a component at a fixed location:

```python
# Feature from 30% to 50% of the time series (20% of total length)
builder.add_feature(constant(value=1.0), start_pct=0.3, end_pct=0.5)

# Another feature from 60% to 80%
builder.add_feature(constant(value=-1.0), start_pct=0.6, end_pct=0.8)
```

### Random Position

Use `random_location=True` with `length_pct` to place features at random positions:

```python
# Feature takes up 20% of the time series, placed randomly for each sample
builder.add_feature(constant(value=1.0), length_pct=0.2, random_location=True)
```

### Signals with Position (Segment Mode)

Signals can also be positioned like features using the same parameters:

```python
# A trend that only appears in the first half
builder.add_signal(trend(endpoints=(0, 1)), start_pct=0.0, end_pct=0.5)

# A burst of noise at a random location
builder.add_signal(gaussian(sigma=2.0), length_pct=0.2, random_location=True, role="noise")
```

## Multivariate Time Series

For multi-channel data, specify `n_dimensions` and use the `dim` parameter.

```python
builder = (
    TimeSeriesBuilder(n_timesteps=100, n_samples=50, n_dimensions=3)
    .for_class(0)
    # Apply to all dimensions
    .add_signal(random_walk(step_size=0.2), dim=[0, 1, 2])
    .add_signal(gaussian(sigma=0.1), role="noise", dim=[0, 1, 2])
    .for_class(1)
    .add_signal(random_walk(step_size=0.2), dim=[0, 1, 2])
    .add_signal(gaussian(sigma=0.1), role="noise", dim=[0, 1, 2])
    # Feature only in dimensions 0 and 1
    .add_feature(
        constant(value=1.0),
        dim=[0, 1],
        length_pct=0.15,
        random_location=True,
        shared_location=True,  # Same position in both dimensions
    )
    # Feature in dimension 2 only
    .add_feature(
        peak(amplitude=2.0, width=5),
        dim=[2],
        start_pct=0.4,
        end_pct=0.6,
    )
)
```

### Shared Location vs Independent Location

When using `random_location=True` with multiple dimensions:

```python
# shared_location=True (default): Feature appears at same position in all specified dims
builder.add_feature(peak(), dim=[0, 1], random_location=True, shared_location=True)

# shared_location=False: Feature appears at different random positions in each dim
builder.add_feature(peak(), dim=[0, 1], random_location=True, shared_location=False)
```

### Shared Randomness

For stochastic components, `shared_randomness` controls whether the same random values are used across dimensions:

```python
# shared_randomness=True: Same noise pattern in all dimensions
builder.add_signal(gaussian(sigma=0.1), dim=[0, 1, 2], shared_randomness=True)

# shared_randomness=False (default): Independent noise in each dimension
builder.add_signal(gaussian(sigma=0.1), dim=[0, 1, 2], shared_randomness=False)
```

## Creating Data Splits

Use `clone()` to create train/test/validation splits from the same data distribution:

```python
# Define the data structure once
base_builder = (
    TimeSeriesBuilder(n_timesteps=100)
    .for_class(0)
    .add_signal(gaussian(sigma=0.5))
    .for_class(1)
    .add_signal(gaussian(sigma=0.5))
    .add_feature(peak(amplitude=2.0), length_pct=0.2, random_location=True)
)

# Create splits with different random seeds
train = base_builder.clone(n_samples=1000, random_state=42).build()
test = base_builder.clone(n_samples=200, random_state=43).build()
val = base_builder.clone(n_samples=100, random_state=44).build()

print(f"Train: {train['X'].shape}")  # (1000, 1, 100)
print(f"Test: {test['X'].shape}")    # (200, 1, 100)
print(f"Val: {val['X'].shape}")      # (100, 1, 100)
```

### Clone Parameters

`clone()` accepts any builder parameter to override:

```python
# Change multiple parameters
variant = base_builder.clone(
    n_samples=500,
    n_timesteps=200,  # Different length
    random_state=99,
    normalization="minmax",
).build()
```

## YAML Configuration

For reproducible experiments, define datasets in YAML files.

### Basic YAML Structure

```yaml
# config.yaml
my_dataset:
  n_timesteps: 100
  n_samples: 200
  n_dimensions: 1
  random_state: 42
  normalization: zscore
  classes:
    - id: 0
      signals:
        - function: gaussian
          params: { sigma: 0.5 }
          role: noise
    - id: 1
      weight: 1.5  # Sample this class 1.5x as often
      signals:
        - function: gaussian
          params: { sigma: 0.5 }
          role: noise
      features:
        - function: peak
          params: { amplitude: 2.0, width: 10 }
          start_pct: 0.3
          end_pct: 0.7
```

### Loading from YAML

```python
from xaitimesynth.parser import load_builders_from_config

# Load all datasets from file
builders = load_builders_from_config(config_path="config.yaml")
dataset = builders["my_dataset"].build()

# Load specific datasets
builders = load_builders_from_config(
    config_path="config.yaml",
    dataset_names=["my_dataset"]
)

# Load from nested path in config
builders = load_builders_from_config(
    config_path="config.yaml",
    path_key="experiments/datasets"
)
```

### YAML Anchors for Reuse

Use YAML anchors to avoid repetition:

```yaml
# Define common settings
common: &common_settings
  n_timesteps: 100
  random_state: 42
  normalization: zscore

# Reuse with aliases
dataset_a:
  <<: *common_settings
  n_samples: 500
  classes:
    - id: 0
      signals: [{ function: gaussian, params: { sigma: 0.5 } }]

dataset_b:
  <<: *common_settings
  n_samples: 1000
  classes:
    - id: 0
      signals: [{ function: gaussian, params: { sigma: 0.5 } }]
```

### Loading from Dictionary or String

```python
# From dictionary
config = {
    "my_dataset": {
        "n_timesteps": 100,
        "n_samples": 50,
        "classes": [
            {"id": 0, "signals": [{"function": "gaussian", "params": {"sigma": 0.5}}]}
        ]
    }
}
builders = load_builders_from_config(config_dict=config)

# From YAML string
yaml_str = """
my_dataset:
  n_timesteps: 100
  classes:
    - id: 0
      signals:
        - function: gaussian
          params: { sigma: 0.5 }
"""
builders = load_builders_from_config(config_str=yaml_str)
```

## Custom Components

### Using the `manual()` Component

For one-off custom patterns, use `manual()` with either values or a generator function:

```python
from xaitimesynth import TimeSeriesBuilder, manual, gaussian
import numpy as np

# With pre-computed values (must match n_timesteps or use as feature with matching length)
values = np.sin(np.linspace(0, 4 * np.pi, 100))  # 100 points to match n_timesteps
builder = TimeSeriesBuilder(n_timesteps=100, n_samples=5).for_class(0)
builder.add_signal(manual(values=values))

# With a generator function (more flexible - automatically handles length)
def damped_sine(n_timesteps, rng, frequency=0.1, decay=0.02, length=None, **kwargs):
    output_length = length if length is not None else n_timesteps
    t = np.arange(output_length)
    return np.exp(-decay * t) * np.sin(2 * np.pi * frequency * t)

builder.add_signal(manual(generator=damped_sine, frequency=0.05, decay=0.01))
```

### Generator Function Signature

Custom generators must follow this signature:

```python
def my_generator(
    n_timesteps: int,           # Total time series length (context)
    # Your custom parameters here
    param1: float = default,
    param2: float = default,
    # Standard parameters (required)
    rng: np.random.RandomState = None,  # Random state for reproducibility
    length: int = None,         # Actual output length (may differ from n_timesteps)
    **kwargs,                   # Catch extra parameters
) -> np.ndarray:
    output_length = length if length is not None else n_timesteps
    # ... your implementation ...
    return result  # 1D array of shape (output_length,)
```

### Registering Custom Components

For reusable components, register them with the package:

```python
from xaitimesynth.registry import register_component

def my_custom_component(param1=1.0, param2=0.5, **kwargs):
    return {"type": "my_custom", "param1": param1, "param2": param2, **kwargs}

# Register as a signal, feature, or both
register_component(my_custom_component, "both")

# Now you can use it like built-in components
builder.add_signal(my_custom_component(param1=2.0))
```

See [Adding Generators](adding_generators.md) for the complete guide on creating new components.

## Accessing Component Data

Each sample's data is broken down in the `components` list:

```python
dataset = builder.build()

# Get the first sample's components
sample = dataset["components"][0]

print(sample.foundation.shape)    # (n_timesteps, n_dims) - foundation signals
print(sample.noise.shape)         # (n_timesteps, n_dims) - noise signals
print(sample.aggregated.shape)    # (n_timesteps, n_dims) - final combined signal
print(sample.features)            # Dict of feature name -> array of feature values
print(sample.feature_masks)       # Dict of feature name -> boolean mask array
```

### Visualization

```python
from xaitimesynth import plot_components, plot_sample

# Plot multiple samples showing component breakdown
plot_components(dataset).show()

# Plot a specific sample
plot_sample(dataset, sample_idx=0).show()

# For multivariate data, specify dimensions
plot_components(dataset, dimensions=[0, 1]).show()
```

### Converting to DataFrame

```python
# Convert to pandas DataFrame for analysis
df = TimeSeriesBuilder().to_df(dataset)
print(df.head())

# Specify which dimensions to include
df = TimeSeriesBuilder().to_df(dataset, dimensions=[0, 1])
```

For detailed documentation of the data structure including shapes, keys, and common access patterns, see the [Data Structure Reference](data_structure.md).
