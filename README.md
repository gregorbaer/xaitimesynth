<p align="center">
  <img src="https://raw.githubusercontent.com/gregorbaer/xaitimesynth/main/docs/assets/logo/xaitimesynth_logo.svg" width="500">
</p>

<p align="center">
  <a href="https://pypi.org/project/xaitimesynth/"><img src="https://img.shields.io/pypi/v/xaitimesynth"></a>
  <a href="https://pypi.org/project/xaitimesynth/"><img src="https://img.shields.io/pypi/pyversions/xaitimesynth"></a>
  <a href="https://github.com/gregorbaer/xaitimesynth/actions/workflows/pytest.yml"><img src="https://github.com/gregorbaer/xaitimesynth/actions/workflows/pytest.yml/badge.svg"></a>
  <a href="https://gregorbaer.github.io/xaitimesynth/"><img src="https://img.shields.io/badge/docs-GitHub%20Pages-blue"></a>
  <a href="https://github.com/gregorbaer/xaitimesynth/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-green"></a>
  <a href="https://doi.org/10.5281/zenodo.18888778"><img src="https://zenodo.org/badge/954013569.svg" alt="DOI"></a>
</p>

# xaitimesynth

A Python package for benchmarking explainable AI (XAI) algorithms (mostly feature attributions) on time series classification tasks using synthetic data with known ground truth feature locations.

## Why xaitimesynth?

Evaluating XAI methods for time series is challenging because we rarely know which time points truly matter for classification. xaitimesynth solves this by generating synthetic data where **you control exactly where the class-discriminating features are located**.

Each synthetic time series follows a simple additive model: `x = background + feature`, where the feature is placed at a known window. This lets you directly measure whether attribution methods correctly identify the important time points.

## Key Features

- **Known ground truth**: Feature locations are tracked internally, enabling direct evaluation of attribution correctness
- **Flexible data generation**: Combine signals (random walks, seasonal patterns, noise) with localized features (peaks, level shifts, pulses)
- **Univariate and multivariate**: Generate single-channel or multi-channel time series
- **Fluent builder API**: Chain methods to define complex datasets concisely
- **YAML configuration**: Define datasets in config files for reproducibility
- **Built-in metrics**: AUC-PR, AUC-ROC, Relevance Mass Accuracy, Relevance Rank Accuracy, and more

## Installation

```bash
pip install xaitimesynth
```

## Quick Start

```python
from xaitimesynth import TimeSeriesBuilder, gaussian_noise, gaussian_pulse, seasonal
from xaitimesynth.metrics import auc_pr_score, relevance_mass_accuracy
import numpy as np

# define dataset structure once
base_builder = (
    TimeSeriesBuilder(n_timesteps=100, normalization="zscore")
    .for_class(0)
    .add_signal(gaussian_noise(sigma=1.0))
    .add_feature(gaussian_pulse(amplitude=3.0), random_location=True, length_pct=0.3)
    .for_class(1)
    .add_signal(gaussian_noise(sigma=1.0))
    .add_feature(
        seasonal(period=10, amplitude=3.0), random_location=True, length_pct=0.3
    )
)

# generate train and test sets with different seeds
train = base_builder.clone(n_samples=200, random_state=42).build()
test = base_builder.clone(n_samples=50, random_state=43).build()

# visualise instances from created dataset (by default first observation from each class)
plot = plot_components(train)
plot.show()

# Replace with your XAI method output; shape must be (n_samples, n_dims, n_timesteps)
attributions = np.random.rand(*test["X"].shape)

# Evaluate against ground truth feature locations
auc = auc_pr_score(attributions, test, normalize=True)
rma = relevance_mass_accuracy(attributions, test)
```

![Example plot](docs/assets/images/quickstart_dataset.png)


## Documentation

Full documentation is available at **[gregorbaer.github.io/xaitimesynth](https://gregorbaer.github.io/xaitimesynth/)**.

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Citation

If you use xaitimesynth in your work, please consider citing:

```
Baer, G. (2026). xaitimesynth: A Python Package for Evaluating Attribution Methods for Time Series with Synthetic Ground Truth (arXiv:2603.06781). arXiv. https://doi.org/10.48550/arXiv.2603.06781
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
