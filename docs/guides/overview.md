# Package Overview

This guide explains the design philosophy, architecture, and key terminology of xaitimesynth.

## Table of Contents

1. [Purpose](#purpose)
2. [Core Concept](#core-concept)
3. [Architecture](#architecture)
4. [Terminology](#terminology)
5. [Data Flow](#data-flow)

## Purpose

xaitimesynth generates synthetic time series data for evaluating explainable AI (XAI) methods. The key insight is that **synthetic data lets us know exactly where the important features are**, enabling direct measurement of whether attribution methods correctly identify them.

In real-world time series classification, we rarely have ground truth about which time points truly matter. A model might classify an ECG as abnormal, but we don't have definitive labels for every timestep marking "this is the important region." xaitimesynth solves this by:

1. Generating data where features have **known locations**
2. Tracking these locations in a structured format
3. Providing **metrics** that compare attributions to ground truth

## Core Concept

Every time series in xaitimesynth follows an additive composition model:

```
x = foundation + noise + feature
```

- **Foundation**: The base signal pattern (random walks, seasonal patterns, trends)
- **Noise**: Stochastic variation layered on top
- **Feature**: The class-discriminating pattern in a specific time window

For a two-class problem, you might have:
- Class 0: foundation + noise + feature A (e.g., downward level shift)
- Class 1: foundation + noise + feature B (e.g., upward level shift)

A classifier trained on this data learns to distinguish between the feature types. An XAI method should attribute high importance to the feature window where the discriminative pattern occurs. Since we know exactly where each feature is located, we can directly measure whether the attributions are correct.

## Architecture

### Main Components

```
xaitimesynth/
├── builder.py        # TimeSeriesBuilder - fluent API for dataset construction
├── components.py     # Component definition functions (gaussian, peak, etc.)
├── generators.py     # Implementation that produces actual arrays
├── registry.py       # Component registration system
├── data_structures.py # TimeSeriesComponents dataclass
├── metrics.py        # XAI evaluation metrics
├── parser.py         # YAML configuration loading
└── visualization.py  # Plotting utilities
```

### Design Patterns

**Builder Pattern (ggplot-inspired)**: The API design is inspired by ggplot's "grammar of graphics," where plots are built by adding modular layers. Similarly, xaitimesynth lets you compose datasets by adding modular components:

```python
dataset = (
    TimeSeriesBuilder(n_timesteps=100, n_samples=50)  # Initialize canvas
    .for_class(0)                                      # Select class context
    .add_signal(gaussian(sigma=0.1), role="noise")    # Add a layer
    .for_class(1)
    .add_signal(gaussian(sigma=0.1), role="noise")
    .add_feature(peak(amplitude=1.0), start_pct=0.3, end_pct=0.7)  # Add another layer
    .build()                                           # Render the result
)
```

Just as ggplot separates data, aesthetics, and geometry into composable pieces, xaitimesynth separates the time series structure (builder parameters), class definitions, and signal/feature components into modular, chainable elements.

**Two-Function Pattern**: Each component type has two functions:
1. **Component function** (in `components.py`): User-facing, returns a dictionary definition
2. **Generator function** (in `generators.py`): Internal, produces the actual numpy array

```python
# User calls this (components.py)
peak(amplitude=1.5, width=10)
# Returns: {"type": "peak", "amplitude": 1.5, "width": 10}

# Builder internally calls this (generators.py)
generate_peak(n_timesteps=100, amplitude=1.5, width=10, rng=rng, length=20)
# Returns: numpy array of shape (20,)
```

**Registry Pattern**: Components are registered dynamically, making the system extensible:

```python
from xaitimesynth.registry import register_component

register_component(my_custom_component, "feature")
```

## Terminology

### Component Types vs Signal Roles

xaitimesynth has two distinct classification systems that can be confusing at first:

#### 1. Component Types (Registry Level)

When registering a component, you specify what it can be used as:

| Type | Can be used with | Typical use |
|------|------------------|-------------|
| `"signal"` | `add_signal()` | Full-length background patterns |
| `"feature"` | `add_feature()` | Localized discriminative patterns |
| `"both"` | Either method | Components like `constant` that work both ways |

#### 2. Signal Roles (Builder Level)

When adding a signal, you specify its semantic role:

| Role | Purpose | Stored in |
|------|---------|-----------|
| `"foundation"` | Base pattern the time series is built on | `components.foundation` |
| `"noise"` | Stochastic variation added to the signal | `components.noise` |

```python
# This is a SIGNAL with role="foundation"
builder.add_signal(random_walk(step_size=0.2), role="foundation")

# This is a SIGNAL with role="noise"
builder.add_signal(gaussian(sigma=0.1), role="noise")

# This is a FEATURE (no role needed, always stored as feature)
builder.add_feature(peak(amplitude=1.0), start_pct=0.3, end_pct=0.7)
```

### Other Key Terms

**Timesteps (`n_timesteps`)**: The length of each time series (number of time points).

**Samples (`n_samples`)**: The number of time series to generate.

**Dimensions (`n_dimensions`)**: The number of channels (1 = univariate, >1 = multivariate).

**Feature mask**: A binary array indicating where the feature is located (1 = feature present, 0 = no feature).

**Segment**: A portion of the time series. Features are always segments; signals can optionally be segments too.

**Shared randomness**: When `True`, the same random values are used across all dimensions. When `False`, each dimension gets independent randomness.

**Shared location**: When `True` with `random_location=True`, features appear at the same position across dimensions.

## Data Flow

Here's how data flows through the system when you call `.build()`:

```
1. TimeSeriesBuilder stores class definitions
   └── Each class has: label, weight, components (foundation, noise, features)
       └── Each component is a dictionary: {"type": "peak", "amplitude": 1.0, ...}

2. .build() is called
   └── For each sample:
       └── Select class based on weights
       └── For each dimension:
           └── Generate foundation signal(s)
           └── Generate noise signal(s)
           └── Generate feature(s) at specified locations
           └── Combine: foundation + noise + feature

3. Output dictionary is created:
   ├── "X": numpy array (n_samples, n_dims, n_timesteps)
   ├── "y": numpy array (n_samples,) - class labels
   ├── "feature_masks": numpy array (n_samples, n_dims, n_timesteps) - ground truth
   └── "components": list of TimeSeriesComponents objects with full breakdown
```

### TimeSeriesComponents Dataclass

Each sample's component breakdown is stored in a `TimeSeriesComponents` object:

```python
@dataclass
class TimeSeriesComponents:
    foundation: np.ndarray  # Shape: (n_dims, n_timesteps)
    noise: np.ndarray       # Shape: (n_dims, n_timesteps)
    features: np.ndarray    # Shape: (n_dims, n_timesteps)
    combined: np.ndarray    # Shape: (n_dims, n_timesteps) - final signal
    feature_mask: np.ndarray  # Shape: (n_dims, n_timesteps) - binary mask
    label: int              # Class label
```

This allows you to:
- Visualize how each component contributes to the final signal
- Access ground truth feature locations for evaluation
- Debug data generation issues

## Next Steps

- [Usage Guide](usage.md) - Detailed API walkthrough with examples
- [Metrics Guide](metrics.md) - How to evaluate XAI methods
- [Adding Generators](adding_generators.md) - Create custom components
