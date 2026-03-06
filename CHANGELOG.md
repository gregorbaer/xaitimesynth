# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/), and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-03-05

### Added

- Fluent builder API (`TimeSeriesBuilder`) for constructing synthetic time series datasets
- Composable signal components: `gaussian_noise`, `random_walk`, `red_noise`, `uniform`, `ecg_like`, `pseudo_periodic`, `seasonal`, `trend`, `constant`
- Localized feature components: `peak`, `trough`, `gaussian_pulse`
- Manual component for user-defined arrays
- Univariate and multivariate time series generation
- Ground truth feature masks for XAI evaluation
- YAML configuration support for reproducible dataset definitions
- Built-in XAI evaluation metrics: AUC-PR, AUC-ROC, NAC, RMA, RRA
- Extensible component registry with decorator-based registration
- Visualization utilities based on lets-plot
- Cylinder-Bell-Funnel benchmark dataset generator
- Normalization utilities (z-score, min-max)
- Clone support for generating train/test splits from a single builder definition
