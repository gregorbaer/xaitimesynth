  <img src="./assets/logo/xaitimesynth_logo.svg" width="500">
</p>

<p align="center">
  <a href="https://pypi.org/project/xaitimesynth/"><img src="https://img.shields.io/pypi/v/xaitimesynth"></a>
  <a href="https://pypi.org/project/xaitimesynth/"><img src="https://img.shields.io/pypi/pyversions/xaitimesynth"></a>
</p>

# xaitimesynth

A Python package for benchmarking explainable AI (XAI) algorithms (mostly feature attributions) on time series classification tasks using synthetic data with known ground truth feature locations.

## Installation

```bash
pip install xaitimesynth
```

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

# Replace with your XAI method output; shape must be (n_samples, n_dims, n_timesteps)
attributions = np.random.rand(*test["X"].shape)

# Evaluate against ground truth feature locations
auc = auc_pr_score(attributions, test, normalize=True)
rma = relevance_mass_accuracy(attributions, test)
```

## Citation

If you use xaitimesynth in your work, please consider citing:

```bibtex
TODO: Add reference after DOI is issued
```

## License

This project is licensed under the [MIT License](https://github.com/gregorbaer/xaitimesynth/blob/main/LICENSE).
