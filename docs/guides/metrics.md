# Evaluation Metrics for Feature Attributions

This guide explains how to evaluate XAI attribution methods using xaitimesynth's metrics module. The metrics compare attribution values against ground truth feature masks to quantify how well an explainer identifies the true discriminative regions.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Available Metrics](#available-metrics)
3. [Input Format](#input-format)
4. [Aggregation Modes](#aggregation-modes)
5. [Working with External XAI Packages](#working-with-external-xai-packages)

## Quick Start

```python
import numpy as np
from xaitimesynth import TimeSeriesBuilder, constant, gaussian, random_walk
from xaitimesynth.metrics import (
    relevance_mass_accuracy,
    relevance_rank_accuracy,
    auc_roc_score,
    auc_pr_score,
    nac_score,
    pointing_game,
    mean_absolute_error,
    mean_squared_error,
)

# Create a dataset with known feature locations
# Both classes have a level shift feature, but in opposite directions
dataset = (
    TimeSeriesBuilder(n_timesteps=100, n_samples=40, random_state=42)
    .for_class(0)
    .add_signal(random_walk(step_size=0.2))
    .add_signal(gaussian(sigma=0.1), role="noise")
    .add_feature(constant(value=-1.0), start_pct=0.4, end_pct=0.6)  # Downward shift
    .for_class(1)
    .add_signal(random_walk(step_size=0.2))
    .add_signal(gaussian(sigma=0.1), role="noise")
    .add_feature(constant(value=1.0), start_pct=0.4, end_pct=0.6)   # Upward shift
    .build()
)

# Evaluate attributions for class-1 samples
class1_indices = np.where(dataset["y"] == 1)[0].tolist()

# Simulate attributions (in practice, these come from your XAI method)
# Shape: (n_samples, n_timesteps, n_dims)
attributions = np.random.rand(len(class1_indices), 100, 1)

# Evaluate
score = relevance_mass_accuracy(attributions, dataset, sample_indices=class1_indices)
print(f"RMA score: {score:.3f}")
```

## Available Metrics

### Relevance Mass Accuracy (RMA)

**What it measures:** Fraction of total attribution "mass" that falls inside the ground truth region.

**Formula:** `sum(attr[mask]) / sum(attr)`

**Range:** [0, 1], higher is better

**Use when:** You want to know if high attributions concentrate on the true features.

```python
score = relevance_mass_accuracy(attributions, dataset, sample_indices=class1_indices)
```

### Relevance Rank Accuracy (RRA)

**What it measures:** Fraction of the top-K attributed timesteps that fall inside the ground truth (where K = size of ground truth region).

**Range:** [0, 1], higher is better

**Use when:** You care about whether the highest-ranked attributions are correct.

```python
score = relevance_rank_accuracy(attributions, dataset, sample_indices=class1_indices)
```

### Pointing Game

**What it measures:** Whether the single highest attribution falls inside the ground truth.

**Range:** 0 or 1 per sample (aggregated score in [0, 1])

**Use when:** You want a simple binary check of the maximum attribution location.

```python
score = pointing_game(attributions, dataset, sample_indices=class1_indices)
```

### AUC-ROC

**What it measures:** Area under the ROC curve, treating attribution values as predictions and the mask as ground truth.

**Range:** [0, 1], 0.5 = random, 1.0 = perfect

**Use when:** You want a threshold-independent ranking metric.

```python
# Standard AUC-ROC
score = auc_roc_score(attributions, dataset, sample_indices=class1_indices)

# Normalized to [-1, 1] where 0 = random
score = auc_roc_score(attributions, dataset, sample_indices=class1_indices, normalize=True)
```

### AUC-PR

**What it measures:** Area under the Precision-Recall curve.

**Range:** [0, 1], baseline = prevalence (feature proportion)

**Use when:** Ground truth is sparse (small features relative to series length). More sensitive than AUC-ROC for imbalanced cases.

```python
# Standard AUC-PR
score = auc_pr_score(attributions, dataset, sample_indices=class1_indices)

# Normalized: (AUC - prevalence) / (1 - prevalence)
score = auc_pr_score(attributions, dataset, sample_indices=class1_indices, normalize=True)
```

### Normalized Attribution Correspondence (NAC)

**What it measures:** Mean z-scored attribution at ground truth locations. Positive = attributions elevated at features.

**Range:** Unbounded (typically -3 to +3 for reasonable attributions)

**Use when:** You want to measure relative elevation of attributions at features compared to background.

```python
# Evaluate at feature locations (default)
score = nac_score(attributions, dataset, sample_indices=class1_indices)

# Evaluate at non-feature locations (should be negative for good attributions)
score = nac_score(attributions, dataset, sample_indices=class1_indices, ground_truth_only=False)
```

### Mean Absolute Error (MAE)

**What it measures:** Average absolute difference between attributions and binary mask.

**Range:** [0, inf), lower is better (0 = perfect)

**Use when:** You want attributions to match the mask exactly (1 at features, 0 elsewhere).

**Note:** Attributions should be normalized to [0, 1] for meaningful results.

```python
score = mean_absolute_error(attributions, dataset, sample_indices=class1_indices)
```

### Mean Squared Error (MSE)

**What it measures:** Average squared difference between attributions and binary mask.

**Range:** [0, inf), lower is better (0 = perfect)

**Use when:** You want to penalize large deviations more heavily than MAE.

**Note:** Attributions should be normalized to [0, 1] for meaningful results.

```python
score = mean_squared_error(attributions, dataset, sample_indices=class1_indices)
```

## Input Format

Attributions can be provided in several shapes:

| Shape | Description | Auto-reshape |
|-------|-------------|--------------|
| `(n_timesteps,)` | Single sample, single dimension | → `(1, T, 1)` |
| `(n_timesteps, n_dims)` | Single sample, multiple dimensions | → `(1, T, D)` |
| `(n_samples, n_timesteps, n_dims)` | Multiple samples and dimensions | Canonical format |
| `(n_samples, n_dims, n_timesteps)` | Alternative ordering | Auto-detected and transposed |

### Example with different shapes

```python
import numpy as np
from xaitimesynth.metrics import relevance_mass_accuracy

# 1D: single sample, single dimension
attr_1d = np.random.rand(100)
score = relevance_mass_accuracy(attr_1d, dataset, sample_indices=[class1_indices[0]])

# 2D: single sample, 3 dimensions
attr_2d = np.random.rand(100, 3)
score = relevance_mass_accuracy(attr_2d, dataset, sample_indices=[class1_indices[0]])

# 3D: 5 samples, 100 timesteps, 3 dimensions
attr_3d = np.random.rand(5, 100, 3)
score = relevance_mass_accuracy(attr_3d, dataset, sample_indices=class1_indices[:5])
```

## Aggregation Modes

All metrics support flexible aggregation via the `average` parameter:

| Mode | Return Type | Description |
|------|-------------|-------------|
| `'macro'` | `float` | Mean across all samples and dimensions (default) |
| `'per_sample'` | `Dict[int, float]` | Mean per sample across dimensions |
| `'per_dimension'` | `Dict[int, float]` | Mean per dimension across samples |
| `None` | `Dict[Tuple[int,int], float]` | No aggregation, raw per-(sample, dimension) scores |

### Example

```python
from xaitimesynth.metrics import relevance_mass_accuracy

# Single score (default)
macro_score = relevance_mass_accuracy(attr, dataset, sample_indices=indices, average='macro')
# Returns: 0.75

# Score per sample
per_sample = relevance_mass_accuracy(attr, dataset, sample_indices=indices, average='per_sample')
# Returns: {10: 0.8, 11: 0.7, 12: 0.75, ...}

# Score per dimension (for multivariate data)
per_dim = relevance_mass_accuracy(attr, dataset, sample_indices=indices, average='per_dimension')
# Returns: {0: 0.72, 1: 0.78}

# Raw scores
raw = relevance_mass_accuracy(attr, dataset, sample_indices=indices, average=None)
# Returns: {(10, 0): 0.8, (10, 1): 0.85, (11, 0): 0.65, ...}
```

## Working with External XAI Packages

Attribution methods from packages like Captum, SHAP, or lime may return arrays in different formats. Here's how to prepare them for xaitimesynth metrics.

### General Approach

1. **Check the output shape** of your XAI method
2. **Reshape to** `(n_samples, n_timesteps, n_dims)` if needed
3. **Match sample indices** between attributions and dataset

### Example Workflow with Train/Test/Validation Splits

This example shows a realistic workflow: define a base builder, create train/test/val splits with different random seeds, train a model, and evaluate attributions on the test set.

```python
import numpy as np
from xaitimesynth import TimeSeriesBuilder, gaussian, constant, random_walk
from xaitimesynth.metrics import relevance_mass_accuracy, auc_pr_score

# 1. Define a base builder with class definitions
#    Both classes have level shift features in opposite directions
base_builder = (
    TimeSeriesBuilder(n_timesteps=100, n_dimensions=3)
    .for_class(0)
    .add_signal(random_walk(step_size=0.2), dim=[0, 1, 2])
    .add_signal(gaussian(sigma=0.1), role="noise", dim=[0, 1, 2])
    .add_feature(constant(-1.0), start_pct=0.4, end_pct=0.6, dim=[0])  # Downward shift
    .for_class(1)
    .add_signal(random_walk(step_size=0.2), dim=[0, 1, 2])
    .add_signal(gaussian(sigma=0.1), role="noise", dim=[0, 1, 2])
    .add_feature(constant(1.0), start_pct=0.4, end_pct=0.6, dim=[0])   # Upward shift
)

# 2. Create train/test/val splits with different random seeds
train_dataset = base_builder.clone(n_samples=200, random_state=42).build()
test_dataset = base_builder.clone(n_samples=50, random_state=43).build()
val_dataset = base_builder.clone(n_samples=30, random_state=44).build()

X_train, y_train = train_dataset["X"], train_dataset["y"]
X_test, y_test = test_dataset["X"], test_dataset["y"]

# 3. Train your model on train set (placeholder)
# model.fit(X_train, y_train)

# 4. Get attributions for test set from your XAI method
#    (placeholder - replace with your actual explainer)
#    Suppose your explainer returns shape (batch, dims, timesteps)
class1_test_mask = y_test == 1
class1_test_indices = np.where(class1_test_mask)[0].tolist()
X_test_class1 = X_test[class1_test_mask]

attributions_raw = np.random.rand(*X_test_class1.shape)  # (n, 3, 100)

# 5. Reshape to (samples, timesteps, dims) if needed
attributions = np.transpose(attributions_raw, (0, 2, 1))  # (n, 100, 3)

# 6. Evaluate on test set
rma = relevance_mass_accuracy(attributions, test_dataset, sample_indices=class1_test_indices)
auc = auc_pr_score(attributions, test_dataset, sample_indices=class1_test_indices, normalize=True)

print(f"Test set RMA: {rma:.3f}")
print(f"Test set normalized AUC-PR: {auc:.3f}")

# 7. Per-dimension analysis
per_dim = relevance_mass_accuracy(
    attributions, test_dataset, sample_indices=class1_test_indices, average='per_dimension'
)
for dim, score in per_dim.items():
    print(f"  Dimension {dim}: {score:.3f}")
```

### Tips

- **Normalize attributions** to [0, 1] before using MAE/MSE metrics
- **Use `dim_indices`** to evaluate specific dimensions in multivariate data
- **Check for NaN/Inf** in attributions before evaluation - these can cause unexpected results
- **Use `.clone()`** to create multiple datasets with the same structure but different samples/seeds
