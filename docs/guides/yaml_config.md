# YAML Configuration Guide

If you want to save dataset definitions in configuration files rather than Python code, you can use YAML. This is useful for:

- **Reproducibility**: Share exact dataset configurations with collaborators
- **Experiment management**: Define multiple dataset variants in one file
- **Version control**: Track configuration changes in git
- **Avoid code duplication**: Keep data generation settings separate from analysis code and avoid code duplication for multiple datasets

## Quick Start

Create a YAML file with your dataset definition, for example:

```yaml
# config.yaml
my_dataset:
  n_timesteps: 100
  n_samples: 200
  random_state: 42
  classes:
    - id: 0
      signals:
        - function: gaussian
          params: { sigma: 0.5 }
      features:
        - function: constant
          params: { value: -1.0 }
          start_pct: 0.4
          end_pct: 0.6
    - id: 1
      signals:
        - function: gaussian
          params: { sigma: 0.5 }
      features:
        - function: constant
          params: { value: 1.0 }
          start_pct: 0.4
          end_pct: 0.6
```

Then load and build in Python:

```python
from xaitimesynth.parser import load_builders_from_config

builders = load_builders_from_config(config_path="config.yaml")
dataset = builders["my_dataset"].build()
```

## Basic Structure

Each dataset in your YAML file needs a name (the top-level key) and a configuration that mirrors the `TimeSeriesBuilder` API:

```yaml
dataset_name:
  # Builder parameters
  n_timesteps: 100
  n_samples: 200
  n_dimensions: 1
  random_state: 42

  # Class definitions (required)
  classes:
    - id: 0
      signals:
        - function: random_walk
          params: { step_size: 0.2 }
        - function: gaussian
          params: { sigma: 0.1 }
      features:
        - function: constant
          params: { value: -1.0 }
          start_pct: 0.4
          end_pct: 0.6
    - id: 1
      # ... class 1 definition
```

You can define multiple datasets in the same file - each top-level key becomes a separate builder.

## Configuration Reference

### Builder Parameters

These go at the top level of each dataset definition:

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `n_timesteps` | int | 100 | Length of each time series |
| `n_samples` | int | 1000 | Total number of samples to generate |
| `n_dimensions` | int | 1 | Number of channels (for multivariate) |
| `random_state` | int | None | Random seed for reproducibility |
| `normalization` | str | "zscore" | Normalization method ("zscore", "minmax", "none") |
| `data_format` | str | "channels_first" | Output shape format |

### Class Definition

Each class in the `classes` list defines one class label and its components:

| Key | Type | Required | Description |
|-----|------|----------|-------------|
| `id` | int | Yes | Class label (0, 1, 2, ...) |
| `weight` | float | No | Sampling weight for class balance (default: 1.0) |
| `signals` | list | No | Background signal components |
| `features` | list | No | Discriminative feature components |

### Signal Configuration

Signals define the background patterns in your time series. Each signal in the `signals` list:

| Key | Type | Required | Description |
|-----|------|----------|-------------|
| `function` | str | Yes | Generator name (e.g., "random_walk", "gaussian") |
| `params` | dict | No | Parameters passed to the generator |
| `dimensions` | list | No | Which dimensions to apply to (null = all) |
| `start_pct`, `end_pct` | float | No | Position (0-1) for partial coverage |
| `length_pct` | float | No | Length as fraction for random placement (scalar only; stochastic forms not supported for signals) |
| `random_location` | bool | No | Place at random position each sample |

### Feature Configuration

Features are the class-discriminating patterns. Each feature in the `features` list:

| Key | Type | Required | Description |
|-----|------|----------|-------------|
| `function` | str | Yes | Generator name (e.g., "peak", "constant") |
| `params` | dict | No | Parameters passed to the generator |
| `start_pct`, `end_pct` | float | No* | Fixed position (0-1) |
| `length_pct` | see below | No* | Length for random placement |
| `random_location` | bool | No | Place randomly (requires `length_pct`) |
| `dimensions` | list | No | Which dimensions to apply to |

*You must specify either `start_pct`/`end_pct` for fixed position, or `length_pct` with `random_location: true`.

#### Stochastic feature lengths with `length_pct`

`length_pct` controls the feature window size. It accepts three forms, both in Python and YAML:

| Form | Python API | YAML syntax | Effect |
|------|-----------|-------------|--------|
| Fixed | `length_pct=0.5` | `length_pct: 0.5` | Same length every sample |
| Discrete choices | `length_pct=[0.25, 0.5]` | `length_pct: [0.25, 0.5]` | Randomly pick one value per sample |
| Uniform range | `length_pct=(0.25, 0.75)` | `length_pct: {range: [0.25, 0.75]}` | Sample uniformly per sample |

> **YAML note:** YAML has no tuple type, so a plain list like `[0.25, 0.75]` is always treated as **discrete choices**, not a range. Use the `{range: [...]}` dict form to express a uniform range in YAML.

```yaml
features:
  # Fixed length — always 30% of the series
  - function: peak
    params: { amplitude: 1.5 }
    random_location: true
    length_pct: 0.3

  # Discrete choices — randomly pick 25% or 50% per sample
  - function: constant
    params: { value: 1.0 }
    random_location: true
    length_pct: [0.25, 0.5]

  # Uniform range — sample any length between 25% and 75% per sample
  - function: trend
    params: { slope: 0.05 }
    random_location: true
    length_pct: {range: [0.25, 0.75]}
```

The Python API uses a tuple for ranges:

```python
# Python equivalents of the three YAML forms above
.add_feature(peak(amplitude=1.5),   random_location=True, length_pct=0.3)
.add_feature(constant(value=1.0),   random_location=True, length_pct=[0.25, 0.5])
.add_feature(trend(slope=0.05),     random_location=True, length_pct=(0.25, 0.75))
```

`to_config()` serializes tuples as `{range: [...]}` so configurations round-trip faithfully through YAML.

### Available Functions

The `function` field must match the name of a component function in the package (e.g., `gaussian`, `peak`, `random_walk`). Use `list_signal_components()` and `list_feature_components()` to discover available functions programmatically. See the [Usage Guide](usage.md#discovering-available-components) for details.

## Loading Configurations

The `load_builders_from_config()` function provides several ways to load configurations:

```python
from xaitimesynth.parser import load_builders_from_config

# Load all datasets from a file
builders = load_builders_from_config(config_path="config.yaml")

# Load from a nested path within the file
# Useful if you organize datasets under categories
builders = load_builders_from_config(
    config_path="config.yaml",
    path_key="experiments/ablation_study"
)

# Load only specific datasets by name
builders = load_builders_from_config(
    config_path="config.yaml",
    dataset_names=["dataset_a", "dataset_b"]
)

# Load from a Python dictionary (useful for testing)
builders = load_builders_from_config(config_dict=my_config_dict)

# Load from a YAML string
builders = load_builders_from_config(config_str=yaml_string)
```

The function returns a dictionary mapping dataset names to `TimeSeriesBuilder` instances.

## Reusing Configuration with YAML Anchors

YAML has built-in support for reusing configuration blocks. This is helpful when multiple datasets share common settings.

Use `&name` to define an anchor and `*name` to reference it. Use `<<:` to merge an anchor's contents:

```yaml
# Define common settings once
common: &common
  n_timesteps: 100
  n_samples: 500
  random_state: 42

# Define reusable signal configurations
gaussian_background: &gaussian_background
  function: gaussian
  params: { sigma: 1.0 }

# Use anchors in dataset definitions
dataset_a:
  <<: *common                    # Merge common settings
  classes:
    - id: 0
      signals: [*gaussian_background]

dataset_b:
  <<: *common
  n_samples: 1000                # Override specific values
  classes:
    - id: 0
      signals: [*gaussian_background]
```

## Creating Train/Test Splits

A common pattern is to load a configuration once and create multiple splits with different random seeds:

```python
builders = load_builders_from_config(config_path="config.yaml")

# Same data distribution, different random realizations
train = builders["my_dataset"].clone(n_samples=1000, random_state=42).build()
test = builders["my_dataset"].clone(n_samples=200, random_state=43).build()
val = builders["my_dataset"].clone(n_samples=100, random_state=44).build()
```

## Exporting Python Configurations to YAML

If you've built a dataset programmatically and want to save its configuration for later use, you can export it with `to_config()`:

```python
import yaml
from xaitimesynth import TimeSeriesBuilder, gaussian, peak

# Define dataset in Python
builder = (
    TimeSeriesBuilder(n_timesteps=100, n_samples=200)
    .for_class(0)
    .add_signal(gaussian(sigma=0.1))
    .for_class(1)
    .add_signal(gaussian(sigma=0.1))
    .add_feature(peak(amplitude=1.0), start_pct=0.3, end_pct=0.6)
)

# Export to dictionary
config = builder.to_config()

# Save to YAML file
with open("config.yaml", "w") as f:
    yaml.dump({"my_dataset": config}, f)
```

This enables round-trip conversion: define in Python, save to YAML, reload later.
