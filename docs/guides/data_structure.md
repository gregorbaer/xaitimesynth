# Data Structure Reference

This guide documents the internal data structures returned by `TimeSeriesBuilder.build()`. Understanding these structures is essential for:
- Extracting data for ML model training (`X`, `y`)
- Accessing ground truth for XAI evaluation (`feature_masks`)
- Visualizing individual components (`components`)

## Quick Reference

```python
dataset = builder.build()

# Top-level keys
dataset["X"]              # Time series data: (n_samples, n_dims, n_timesteps)
dataset["y"]              # Class labels: (n_samples,)
dataset["feature_masks"]  # Ground truth masks: Dict[str, (n_samples, n_timesteps)]
dataset["components"]     # Per-sample breakdown: List[TimeSeriesComponents]
dataset["metadata"]       # Configuration info: Dict
```

## Dataset Dictionary

When you call `builder.build()`, it returns a dictionary with these keys:

### `X` - Time Series Data

The main time series array for ML training.

| Property | Value |
|----------|-------|
| Type | `np.ndarray` |
| Shape | `(n_samples, n_dimensions, n_timesteps)` in `channels_first` format |
| Dtype | `float64` |

```python
X = dataset["X"]
print(X.shape)  # (200, 3, 100) = 200 samples, 3 channels, 100 timesteps

# Access a single time series
sample_idx, dim_idx = 0, 0
single_ts = X[sample_idx, dim_idx, :]  # Shape: (100,)

# Filter by class
class_1_mask = dataset["y"] == 1
X_class_1 = X[class_1_mask]  # All samples from class 1
```

**Data Format Options:**

The default format is `channels_first` (PyTorch/tsai compatible). To use `channels_last`:

```python
# Option 1: Set at build time
builder = TimeSeriesBuilder(data_format="channels_last")

# Option 2: Convert existing dataset
from xaitimesynth import TimeSeriesBuilder
dataset_cl = TimeSeriesBuilder.convert_data_format(dataset, "channels_last")
# Shape changes: (n_samples, n_dims, n_timesteps) -> (n_samples, n_timesteps, n_dims)
```

### `y` - Class Labels

Integer class labels for each sample.

| Property | Value |
|----------|-------|
| Type | `np.ndarray` |
| Shape | `(n_samples,)` |
| Dtype | `int64` |

```python
y = dataset["y"]
print(np.unique(y))  # [0, 1] for binary classification

# Get sample indices for each class
class_0_indices = np.where(y == 0)[0]
class_1_indices = np.where(y == 1)[0]
```

### `feature_masks` - Ground Truth

Dictionary of boolean masks indicating where features are located. This is the ground truth for XAI evaluation.

| Property | Value |
|----------|-------|
| Type | `Dict[str, np.ndarray]` |
| Key format | `"class_{label}_feature_{idx}_{type}_dim{dim}"` |
| Value shape | `(n_samples, n_timesteps)` |
| Value dtype | `bool` |

```python
feature_masks = dataset["feature_masks"]

# List all feature mask keys
print(feature_masks.keys())
# Example output:
# - 'class_0_feature_0_constant_dim0'
# - 'class_0_feature_1_peak_dim1'
# - 'class_1_feature_0_constant_dim0'

# Access a specific mask
mask = feature_masks["class_0_feature_0_constant_dim0"]
print(mask.shape)  # (n_samples, n_timesteps)

# Find feature location for a specific sample
sample_idx = 5
feature_timesteps = np.where(mask[sample_idx])[0]
print(f"Feature at timesteps: {feature_timesteps[0]} to {feature_timesteps[-1]}")
```

**Key Naming Convention:**

| Part | Meaning |
|------|---------|
| `class_{label}` | Which class this feature belongs to |
| `feature_{idx}` | Feature index (order added via `add_feature()`) |
| `{type}` | Component type (e.g., `constant`, `peak`) |
| `dim{dim}` | Dimension index |

**Combining Masks:**

To get a combined mask for all features in a class:

```python
# All feature masks for class 0
class_0_masks = [v for k, v in feature_masks.items() if k.startswith("class_0")]

# Combine with OR (any feature present)
combined_mask = np.any(class_0_masks, axis=0)  # Shape: (n_samples, n_timesteps)
```

### `components` - Per-Sample Breakdown

List of `TimeSeriesComponents` objects, one per sample. Contains the individual signal components before combination.

| Property | Value |
|----------|-------|
| Type | `List[TimeSeriesComponents]` |
| Length | `n_samples` |

```python
components = dataset["components"]
comp = components[sample_idx]  # Get one sample's breakdown
```

Each `TimeSeriesComponents` has these attributes:

#### `foundation`

Base signal pattern (random walks, seasonal patterns, etc.).

| Property | Value |
|----------|-------|
| Shape | `(n_timesteps, n_dimensions)` |
| Note | Shape is (T, D), not (D, T) |

```python
foundation = comp.foundation
print(foundation.shape)  # (100, 3)

# Get foundation for dimension 0
foundation_dim0 = foundation[:, 0]  # Shape: (100,)
```

#### `aggregated`

Final combined signal (foundation + features). This matches the corresponding row in `X` but with `(T, D)` shape instead of `(D, T)`.

| Property | Value |
|----------|-------|
| Shape | `(n_timesteps, n_dimensions)` |

```python
aggregated = comp.aggregated

# Verify it matches X (accounting for shape difference)
# X is (D, T), aggregated is (T, D)
assert np.allclose(dataset["X"][sample_idx], aggregated.T)
```

#### `features`

Dictionary mapping feature names to their values. Values are `NaN` outside the feature region.

| Property | Value |
|----------|-------|
| Type | `Dict[str, np.ndarray]` |
| Key format | `"feature_{idx}_{type}_dim{dim}"` |
| Value shape | `(n_timesteps,)` |

```python
features = comp.features
print(features.keys())
# Example: ['feature_0_constant_dim0', 'feature_1_peak_dim1']

# Access feature values
feat_values = features["feature_0_constant_dim0"]
print(feat_values.shape)  # (100,)

# Get only the non-NaN values (where feature exists)
valid_values = feat_values[~np.isnan(feat_values)]
```

#### `feature_masks`

Per-sample boolean masks (same keys as `features`).

| Property | Value |
|----------|-------|
| Type | `Dict[str, np.ndarray]` |
| Value shape | `(n_timesteps,)` |
| Value dtype | `bool` |

```python
masks = comp.feature_masks
mask = masks["feature_0_constant_dim0"]
feature_range = np.where(mask)[0]
print(f"Feature spans timesteps {feature_range[0]} to {feature_range[-1]}")
```

### `metadata` - Configuration

Dictionary with dataset configuration information.

```python
metadata = dataset["metadata"]
print(metadata.keys())
# ['n_samples', 'n_timesteps', 'n_dimensions', 'class_definitions',
#  'normalize', 'normalization_kwargs', 'random_state', 'data_format', 'shuffled']
```

| Key | Description |
|-----|-------------|
| `n_samples` | Number of samples generated |
| `n_timesteps` | Length of each time series |
| `n_dimensions` | Number of channels |
| `class_definitions` | List of class configuration dicts |
| `normalize` | Normalization method used |
| `normalization_kwargs` | Additional normalization parameters |
| `random_state` | Random seed used |
| `data_format` | `"channels_first"` or `"channels_last"` |
| `shuffled` | Whether samples were shuffled |

## Common Use Cases

### ML Training

```python
from sklearn.model_selection import train_test_split

dataset = builder.build()

# Extract data
X, y = dataset["X"], dataset["y"]

# Split for training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
```

### XAI Evaluation

```python
from xaitimesynth.metrics import auc_pr_score

# Your XAI method produces attributions
attributions = explainer.explain(X)  # Shape: (n_samples, n_timesteps, n_dims)

# Evaluate against ground truth
score = auc_pr_score(attributions, dataset)
```

### Visualization of Components

```python
from xaitimesynth import plot_components, plot_component

# Plot all components (foundation, features, aggregated) for sample 0, dimension 0
plot_components(dataset, sample_indices=0, dimensions=[0])

# Or plot individual arrays using plot_component
sample_idx = 0
dim_idx = 0
comp = dataset["components"][sample_idx]

plot_component(signal=comp.foundation[:, dim_idx], title="Foundation")
plot_component(signal=comp.aggregated[:, dim_idx], title="Aggregated")
```

### Get Samples by Class

```python
# Get all class 0 samples
class_0_mask = dataset["y"] == 0
X_class_0 = dataset["X"][class_0_mask]
components_class_0 = [c for i, c in enumerate(dataset["components"]) if class_0_mask[i]]

# Get indices for iteration
class_0_indices = np.where(dataset["y"] == 0)[0].tolist()
```

### Access Feature Location

```python
# From top-level feature_masks (for all samples)
mask_key = "class_0_feature_0_constant_dim0"
mask = dataset["feature_masks"][mask_key]

# For a specific sample
sample_idx = 5
start = np.where(mask[sample_idx])[0][0]
end = np.where(mask[sample_idx])[0][-1]
print(f"Sample {sample_idx}: feature at timesteps {start}-{end}")

# Or from per-sample components
comp = dataset["components"][sample_idx]
sample_mask = comp.feature_masks["feature_0_constant_dim0"]
```

## Shape Summary

| Array | Shape | Notes |
|-------|-------|-------|
| `dataset["X"]` | `(N, D, T)` | `channels_first` format |
| `dataset["y"]` | `(N,)` | |
| `dataset["feature_masks"][key]` | `(N, T)` | Boolean |
| `comp.foundation` | `(T, D)` | Per-sample |
| `comp.aggregated` | `(T, D)` | Per-sample |
| `comp.features[key]` | `(T,)` | 1D per feature |
| `comp.feature_masks[key]` | `(T,)` | 1D per feature |

Where: `N` = n_samples, `D` = n_dimensions, `T` = n_timesteps

## Exploration Script

For hands-on exploration, run the provided script:

```bash
python scripts/explore_data_structure.py
```

This script creates a multivariate dataset and prints detailed information about each part of the data structure.
