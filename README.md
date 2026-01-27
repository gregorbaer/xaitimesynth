# xaitimesynth

A Python package for benchmarking explainable AI (XAI) algorithms on time series classification tasks using synthetic data with known ground truth feature locations.

## Why xaitimesynth?

Evaluating XAI methods for time series is challenging because we rarely know which time points truly matter for classification. xaitimesynth solves this by generating synthetic data where **you control exactly where the class-discriminating features are located**.

Each synthetic time series follows a simple additive model:

```
x = n + f
```

Where `n` is background signal (noise, trends, etc.) and `f` contains the class-specific feature in a known window. This lets you directly measure whether attribution methods correctly identify the important time points.

## Key Features

- **Known ground truth**: Feature locations are tracked internally, enabling direct evaluation of attribution correctness
- **Flexible data generation**: Combine signals (random walks, seasonal patterns, noise) with localized features (peaks, level shifts)
- **Univariate and multivariate**: Generate single-channel or multi-channel time series
- **Fluent builder API**: Chain methods to define complex datasets concisely
- **YAML configuration**: Define datasets in config files for reproducibility
- **Built-in metrics**: AUC-PR, AUC-ROC, Relevance Mass Accuracy, Relevance Rank Accuracy, and more
- **Train/test splits**: Clone builders with different random seeds to create splits from the same distribution

## Installation

```bash
pip install xaitimesynth
```

## Quick Start

```python
from xaitimesynth import TimeSeriesBuilder, gaussian, random_walk, constant

# Create a dataset where both classes have features (level shifts in opposite directions)
dataset = (
    TimeSeriesBuilder(n_timesteps=100, n_samples=200, random_state=42)
    .for_class(0)
    .add_signal(random_walk(step_size=0.2))
    .add_signal(gaussian(sigma=0.1))
    .add_feature(constant(value=-1.0), start_pct=0.4, end_pct=0.6)  # Downward shift
    .for_class(1)
    .add_signal(random_walk(step_size=0.2))
    .add_signal(gaussian(sigma=0.1))
    .add_feature(constant(value=1.0), start_pct=0.4, end_pct=0.6)   # Upward shift
    .build()
)

# Access the data
X = dataset["X"]                  # Shape: (200, 1, 100) - samples x dims x timesteps
y = dataset["y"]                  # Shape: (200,) - class labels
feature_masks = dataset["feature_masks"]  # Dict of ground truth masks per feature
components = dataset["components"]        # List of TimeSeriesComponents with full breakdown
```

## Evaluating Attributions

```python
import numpy as np
from xaitimesynth.metrics import auc_pr_score, relevance_mass_accuracy

# Get indices for class 1 (the class with features)
class1_indices = np.where(y == 1)[0].tolist()

# Your attribution method produces scores for each timestep
# Shape: (n_samples, n_timesteps, n_dims)
attributions = your_explainer.explain(X[class1_indices])

# Evaluate: do high attributions align with true feature locations?
auc = auc_pr_score(attributions, dataset, sample_indices=class1_indices, normalize=True)
rma = relevance_mass_accuracy(attributions, dataset, sample_indices=class1_indices)

print(f"Normalized AUC-PR: {auc:.3f}")  # 0 = random, 1 = perfect
print(f"Relevance Mass Accuracy: {rma:.3f}")  # Fraction of attribution in ground truth
```

## Creating Train/Test Splits

```python
from xaitimesynth import TimeSeriesBuilder, gaussian, random_walk, constant

# Define the data structure once (both classes have features)
base_builder = (
    TimeSeriesBuilder(n_timesteps=100, n_dimensions=3)
    .for_class(0)
    .add_signal(random_walk(step_size=0.2), dim=[0, 1, 2])
    .add_signal(gaussian(sigma=0.1), dim=[0, 1, 2])
    .add_feature(constant(-1.0), start_pct=0.4, end_pct=0.6, dim=[0])
    .for_class(1)
    .add_signal(random_walk(step_size=0.2), dim=[0, 1, 2])
    .add_signal(gaussian(sigma=0.1), dim=[0, 1, 2])
    .add_feature(constant(1.0), start_pct=0.4, end_pct=0.6, dim=[0])
)

# Generate splits with different seeds
train = base_builder.clone(n_samples=500, random_state=42).build()
test = base_builder.clone(n_samples=100, random_state=43).build()
val = base_builder.clone(n_samples=50, random_state=44).build()
```

## Loading from YAML Configuration

```yaml
# config.yaml
my_dataset:
  n_timesteps: 100
  n_samples: 200
  n_dimensions: 1
  random_state: 42
  classes:
    - id: 0
      signals:
        - { function: random_walk, params: { step_size: 0.2 } }
        - { function: gaussian, params: { sigma: 0.1 } }
      features:
        - { function: constant, params: { value: -1.0 }, start_pct: 0.4, end_pct: 0.6 }
    - id: 1
      signals:
        - { function: random_walk, params: { step_size: 0.2 } }
        - { function: gaussian, params: { sigma: 0.1 } }
      features:
        - { function: constant, params: { value: 1.0 }, start_pct: 0.4, end_pct: 0.6 }
```

```python
from xaitimesynth.parser import load_builders_from_config

builders = load_builders_from_config(config_path="config.yaml")
dataset = builders["my_dataset"].build()
```

## Available Components

Discover available components programmatically:

```python
from xaitimesynth import list_signal_components, list_feature_components

# Signal components (background patterns): random_walk, gaussian, seasonal, etc.
print(list_signal_components().keys())

# Feature components (discriminative patterns): peak, trough, constant, etc.
print(list_feature_components().keys())
```

Use `manual()` to integrate your own generator functions. See the [Usage Guide](docs/guides/usage.md) for details.

## Documentation

- [Package Overview](docs/guides/overview.md) - Design philosophy and architecture
- [Usage Guide](docs/guides/usage.md) - Detailed API walkthrough
- [Data Structure Reference](docs/guides/data_structure.md) - Internal data format for ML and XAI
- [YAML Configuration](docs/guides/yaml_config.md) - Define datasets in config files
- [Metrics Guide](docs/guides/metrics.md) - Evaluation metrics explained
- [Adding Generators](docs/guides/adding_generators.md) - Extend with custom components

## Citation

If you use xaitimesynth in your research, please cite:

```bibtex
@software{xaitimesynth,
  title = {xaitimesynth: Synthetic Time Series for XAI Evaluation},
  author = {TODO},
  year = {TODO},
  url = {https://github.com/YOUR_USERNAME/xaitimesynth}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.
