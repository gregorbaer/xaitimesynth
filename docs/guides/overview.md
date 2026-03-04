# Package Overview

This guide explains the core concept, design, and key terminology of xaitimesynth.

## Purpose

xaitimesynth generates synthetic time series data for evaluating explainable AI (XAI) methods. The key insight is that **synthetic data lets us know exactly where the important features are**, enabling direct measurement of whether attribution methods correctly identify them.

In real-world time series classification, we rarely have ground truth about which time points truly matter. xaitimesynth solves this by generating data where features have **known locations**, tracking those locations, and providing metrics to compare attributions against ground truth.

## Core Concept

Every time series follows an additive composition model:

```
x = background + feature
```

- **Background**: The base signal pattern (noise, random walks, seasonal patterns, trends)
- **Feature**: The class-discriminating pattern placed in a specific time window

For a two-class problem:
- Class 0: background + feature A (e.g., a downward level shift)
- Class 1: background + feature B (e.g., a seasonal burst)

Since we know exactly where each feature is located, we can directly measure whether attribution methods highlight the right regions.

## Design

The builder uses chainable method calls that naturally compose: each call adds a layer that gets summed into the final signal. This mirrors the additive `x = background + feature` model directly in the API:

```python
dataset = (
    TimeSeriesBuilder(n_timesteps=100, n_samples=200)
    .for_class(0)
    .add_signal(gaussian_noise(sigma=0.5))      # background layer
    .for_class(1)
    .add_signal(gaussian_noise(sigma=0.5))
    .add_feature(peak(amplitude=2.0), start_pct=0.3, end_pct=0.7)  # feature layer
    .build()
)
```

Internally, the package has three main layers: component functions (user-facing API in `components.py`), generator functions (array production in `generators.py`), and the registry (`registry.py`) that connects them. See [Adding Generators](adding_generators.md) for details.

## Terminology

### Component Types (Registry Level)

Each component is registered with a type indicating its **intended use**:

| Type | Intended use | Examples |
|------|--------------|----------|
| `"signal"` | Full-length background patterns | `random_walk`, `gaussian_noise`, `seasonal` |
| `"feature"` | Localized discriminative patterns | `peak`, `trough`, `gaussian_pulse` |
| `"both"` | Works well either way | `constant`, `trend`, `manual` |

This is purely for discoverability via `list_signal_components()` / `list_feature_components()`. It does **not** restrict usage — you can use any component with `add_signal()` or `add_feature()`.

### Signals vs Features (Builder Level)

| Method | Purpose | Stored in |
|--------|---------|-----------|
| `add_signal()` | Background patterns | `components.background` |
| `add_feature()` | Class-discriminating patterns with tracked locations | `components.features` |

All signals are combined additively into the background. Features are tracked separately so their locations serve as ground truth for XAI evaluation.

### Key Terms

| Term | Meaning |
|------|---------|
| `n_timesteps` | Length of each time series |
| `n_samples` | Number of time series to generate |
| `n_dimensions` | Number of channels (1 = univariate, >1 = multivariate) |
| Feature mask | Binary array: 1 where a feature is present, 0 elsewhere |
| Shared randomness | When `True`, same random values used across all dimensions |
| Shared location | When `True` with `random_location=True`, feature appears at the same position across dimensions |

## Next Steps

- [Usage Guide](usage.md) — API walkthrough with examples
- [Data Structure Reference](data_structure.md) — Output format, shapes, and access patterns
- [Metrics Guide](metrics.md) — How to evaluate XAI attributions
