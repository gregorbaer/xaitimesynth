# Usage Guide

This guide provides a detailed walkthrough of the xaitimesynth API. For a quick overview, see the [Quick Start](../index.md).

## Table of Contents

1. [Builder Parameters](#builder-parameters)
2. [Discovering Available Components](#discovering-available-components)
3. [Adding Signals](#adding-signals)
4. [Adding Features](#adding-features)
5. [Positioning Features and Signals](#positioning-features-and-signals)
6. [Multivariate Time Series](#multivariate-time-series)
7. [Creating Data Splits](#creating-data-splits)
8. [YAML Configuration](#yaml-configuration)
9. [Custom Components](#custom-components)
10. [Accessing Component Data](#accessing-component-data)

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
| `background_fill_value` | 0.0 | Fill value for background component |


### Data Format

```python
# Default: channels first (N, D, T)
dataset = TimeSeriesBuilder(n_dimensions=3, data_format="channels_first").build()
print(dataset["X"].shape)  # (n_samples, 3, n_timesteps)

# Alternative: channels last (N, T, D)
dataset = TimeSeriesBuilder(n_dimensions=3, data_format="channels_last").build()
print(dataset["X"].shape)  # (n_samples, n_timesteps, 3)
```

## Discovering Available Components

xaitimesynth provides functions to discover available signal and feature components programmatically. Each function returns a dictionary mapping component names to their definition functions:

```python
from xaitimesynth import list_components, list_signal_components, list_feature_components

# List all registered components
all_components = list_components()
print(all_components.keys())
# dict_keys(['constant', 'random_walk', 'gaussian_noise', 'uniform', 'seasonal', ...])

# List components designed for use as signals (background patterns)
signals = list_signal_components()
print(signals.keys())
# dict_keys(['constant', 'random_walk', 'gaussian_noise', 'uniform', 'seasonal', ...])

# List components designed for use as features (discriminative patterns)
features = list_feature_components()
print(features.keys())
# dict_keys(['constant', 'peak', 'trough', 'gaussian_pulse', 'trend', ...])
```


## Adding Signals

Signals are full-length background patterns. They are used for the general shape of the time series. Use `add_signal()` to add them. All signals are combined additively into the background component.

```python
from xaitimesynth import TimeSeriesBuilder, random_walk, gaussian_noise, seasonal, trend, red_noise

builder = (
    TimeSeriesBuilder(n_timesteps=200)
    .for_class(0)
    # Multiple signals are additive
    .add_signal(random_walk(step_size=0.2))
    .add_signal(gaussian_noise(sigma=0.1))
    .add_signal(seasonal(period=20, amplitude=0.5))
)
```


## Adding Features

Features are localized discriminative patterns. Use `add_feature()` to add them.

```python
from xaitimesynth import TimeSeriesBuilder, peak, trough, constant, gaussian_pulse, gaussian_noise

builder = (
    TimeSeriesBuilder(n_timesteps=100)
    .for_class(1)
    .add_signal(gaussian_noise(sigma=0.5))
    # Fixed position: 30% to 60% of the time series (30 timesteps)
    .add_feature(peak(amplitude=2.0, width=5), start_pct=0.3, end_pct=0.6)
    # Random position: feature takes up 15% of length, placed randomly
    .add_feature(constant(value=1.0), length_pct=0.15, random_location=True)
)
```

## Positioning Features and Signals

### Fixed Position

Use `start_pct` and `end_pct` together to place a component at a fixed location:

```python
# Feature from 30% to 50% of the time series (20% of total length)
builder.add_feature(constant(value=1.0), start_pct=0.3, end_pct=0.5)
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

# A burst of signal at a random location
builder.add_signal(gaussian_noise(sigma=2.0), length_pct=0.2, random_location=True)
```

## Multivariate Time Series

See the [Multivariate Time Series example notebook](../examples/multivariate.ipynb) for
runnable examples covering `n_dimensions`, `dim`, `shared_location`, and `shared_randomness`.

## Creating Data Splits

Use `clone()` to create train/test/validation splits from the same data distribution:

```python
# Define the data structure once
base_builder = (
    TimeSeriesBuilder(n_timesteps=100)
    .for_class(0)
    .add_signal(gaussian_noise(sigma=0.5))
    .for_class(1)
    .add_signal(gaussian_noise(sigma=0.5))
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

For reproducible experiments, define datasets in YAML files. The YAML structure mirrors the builder API:

```yaml
# config.yaml
my_dataset:
  n_timesteps: 100
  n_samples: 200
  random_state: 42
  classes:
    - id: 0
      signals:
        - function: gaussian_noise
          params: { sigma: 0.5 }
    - id: 1
      signals:
        - function: gaussian_noise
          params: { sigma: 0.5 }
      features:
        - function: peak
          params: { amplitude: 2.0, width: 10 }
          start_pct: 0.3
          end_pct: 0.7
```

```python
from xaitimesynth.parser import load_builders_from_config

builders = load_builders_from_config(config_path="config.yaml")
dataset = builders["my_dataset"].build()
```

For the full reference (config options, stochastic lengths, YAML anchors, export), see the [YAML Configuration Guide](yaml_config.md).

## Custom Components

### Using the `manual()` Component

For one-off custom patterns, use `manual()` with either values or a generator function:

```python
from xaitimesynth import TimeSeriesBuilder, manual, gaussian_noise
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

`builder.build()` returns a dictionary with these top-level keys:

```python
dataset["X"]               # (n_samples, n_dims, n_timesteps) — time series data
dataset["y"]               # (n_samples,) — class labels
dataset["feature_masks"]   # Dict[str, (n_samples, n_timesteps)] — ground truth locations
dataset["components"]      # List[TimeSeriesComponents] — per-sample breakdown
dataset["metadata"]        # Dict — configuration info
```

See the [Data Structure Reference](data_structure.md) for shapes, key naming conventions, and access patterns.

### Visualization

```python
from xaitimesynth import plot_components

# Plot background, features, and aggregated signal for a sample
plot_components(dataset).show()

# For multivariate data, specify dimensions
plot_components(dataset, dimensions=[0, 1]).show()
```
