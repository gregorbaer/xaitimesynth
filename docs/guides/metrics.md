# Evaluation Metrics for Feature Attributions

This guide explains how to evaluate XAI attribution methods using xaitimesynth's metrics module. The metrics compare attribution values against ground truth feature masks to quantify how well an explainer identifies the true discriminative regions.

## Table of Contents

1. [Available Metrics](#available-metrics)
2. [Input Format](#input-format)
3. [Aggregation Modes](#aggregation-modes)
4. [Working with External XAI Packages](#working-with-external-xai-packages)


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

Attribution methods from packages like Captum, SHAP, or lime may return arrays in different formats. The canonical input format is `(n_samples, n_timesteps, n_dims)` — see the [Input Format](#input-format) table above. If your method returns `(n_samples, n_dims, n_timesteps)`, transpose before passing:

```python
import numpy as np
from xaitimesynth import TimeSeriesBuilder, gaussian_noise, gaussian_pulse, seasonal
from xaitimesynth.metrics import relevance_mass_accuracy, auc_pr_score

base_builder = (
    TimeSeriesBuilder(n_timesteps=100, normalization="zscore")
    .for_class(0)
    .add_signal(gaussian_noise(sigma=1.0))
    .add_feature(gaussian_pulse(amplitude=3.0), random_location=True, length_pct=0.3)
    .for_class(1)
    .add_signal(gaussian_noise(sigma=1.0))
    .add_feature(seasonal(period=10, amplitude=3.0), random_location=True, length_pct=0.3)
)
test_dataset = base_builder.clone(n_samples=50, random_state=43).build()

# attributions_raw shape: (n_samples, n_dims, n_timesteps) from your XAI method
attributions_raw = np.random.rand(*test_dataset["X"].shape)

# Transpose to (n_samples, n_timesteps, n_dims)
attributions = np.transpose(attributions_raw, (0, 2, 1))

# Evaluate (optionally filter to specific class)
class1_indices = np.where(test_dataset["y"] == 1)[0].tolist()
rma = relevance_mass_accuracy(attributions, test_dataset, sample_indices=class1_indices)
auc = auc_pr_score(attributions, test_dataset, sample_indices=class1_indices, normalize=True)
```

**Tips:**
- Normalize attributions to [0, 1] before using MAE/MSE metrics
- Check for NaN/Inf in attributions before evaluation
