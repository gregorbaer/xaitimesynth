# XAITimeSynth

A Python package for benchmarking explainable AI (XAI) algorithms on time series classification tasks using synthetic data with known ground truth feature locations.

## Purpose

Generate synthetic time series data where the location of class-discriminating features is known, enabling direct evaluation of whether attribution methods correctly identify important time points.

**Core concept:** `x = n + f` (background signal + localized feature)

## Quick Start

```python
from xaitimesynth import TimeSeriesBuilder, gaussian_noise, peak

dataset = (
    TimeSeriesBuilder(n_timesteps=100, n_samples=50)
    .for_class(0)
    .add_signal(gaussian_noise(sigma=0.1))
    .for_class(1)
    .add_signal(gaussian_noise(sigma=0.1))
    .add_feature(peak(amplitude=1.0, width=10), start_pct=0.3, end_pct=0.7)
    .build()
)

X, y, feature_masks = dataset["X"], dataset["y"], dataset["feature_masks"]
```

## Installation

```bash
pip install xaitimesynth
```

## Documentation

- **[Overview](guides/overview.md)** - Introduction and core concepts
- **[Usage Guide](guides/usage.md)** - Detailed usage examples
- **[Data Structure](guides/data_structure.md)** - Understanding the output format
- **[YAML Configuration](guides/yaml_config.md)** - Define datasets in config files
- **[Metrics](guides/metrics.md)** - XAI evaluation metrics
- **[API Reference](api/builder.md)** - Full API documentation
