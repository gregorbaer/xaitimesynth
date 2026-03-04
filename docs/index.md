# XAITimeSynth

A Python package for benchmarking explainable AI (XAI) algorithms on time series classification tasks using synthetic data with known ground truth feature locations.

## Purpose

Generate synthetic time series data where the location of class-discriminating features is known, enabling direct evaluation of whether attribution methods correctly identify important time points.

## Quick Start

```python
from xaitimesynth import TimeSeriesBuilder, gaussian_noise, gaussian_pulse, seasonal
from xaitimesynth.metrics import auc_pr_score, relevance_mass_accuracy
import numpy as np

# Define dataset structure once
base_builder = (
    TimeSeriesBuilder(n_timesteps=100, normalization="zscore")
    .for_class(0)
    .add_signal(gaussian_noise(sigma=1.0))
    .add_feature(gaussian_pulse(amplitude=3.0), random_location=True, length_pct=0.3)
    .for_class(1)
    .add_signal(gaussian_noise(sigma=1.0))
    .add_feature(seasonal(period=10, amplitude=3.0), random_location=True, length_pct=0.3)
)

# Generate train and test sets with different seeds
train = base_builder.clone(n_samples=200, random_state=42).build()
test  = base_builder.clone(n_samples=50,  random_state=43).build()

# Replace with your XAI method; shape must be (n_samples, n_dims, n_timesteps)
attributions = np.random.rand(*test["X"].shape)

# Evaluate against ground truth feature locations
auc = auc_pr_score(attributions, test, normalize=True)
rma = relevance_mass_accuracy(attributions, test)
```

## Installation

```bash
pip install xaitimesynth
```

## Where to Go Next

1. **[Usage Guide](guides/usage.md)** — Full API: signals, features, positioning, multivariate, splits
2. **[Metrics Guide](guides/metrics.md)** — Evaluate XAI attributions against ground truth
3. **[Data Structure Reference](guides/data_structure.md)** — Output shapes, keys, and access patterns
4. **[YAML Configuration](guides/yaml_config.md)** — Define datasets in config files
5. **[Adding Generators](guides/adding_generators.md)** — Create custom signal and feature components
