# Adding a New Generator to XAITimeSynth: Step-by-Step Guide

This guide walks you through the process of adding a new generator function to the xaitimesynth package. By following these steps, you'll be able to create custom time series components that can be used in the TimeSeriesBuilder API.

## Table of Contents

1. [Overview of the Generator System](#overview-of-the-generator-system)
2. [How Generators Work Internally](#how-generators-work-internally)
3. [The Two-Function Pattern](#the-two-function-pattern)
4. [Standard Parameters Explained](#standard-parameters-explained)
5. [Adding a New Generator](#adding-a-new-generator)
6. [Complete Example](#complete-example)
7. [Testing Your Generator](#testing-your-generator)

## Overview of the Generator System

The xaitimesynth package has a flexible, three-layer system for creating and registering components:

1. **Component Functions** (`components.py`) - User-facing API that creates component definitions
2. **Generator Functions** (`generators.py`) - Internal functions that produce the actual numpy arrays
3. **Registration System** (`registry.py`) - Connects components to generators and tracks component types

```
User Code → Component Function → Component Dict → Builder → generate_component() → Generator Function → numpy array
```

## How Generators Work Internally

### The Flow from User Code to Generated Data

When you write code like this:

```python
builder = TimeSeriesBuilder(n_timesteps=100, n_samples=50)
builder.add_signal(random_walk(step_size=0.2))
```

Here's what happens internally:

1. **Component Definition Creation**: `random_walk(step_size=0.2)` returns a dictionary:
   ```python
   {"type": "random_walk", "step_size": 0.2}
   ```

2. **Builder Storage**: The builder stores this component definition in its internal structure, along with positioning info (whether it's a signal or feature, start/end positions, etc.)

3. **Generation Time**: When `.build()` is called, the builder:
   - Iterates through all component definitions
   - Calls `generate_component(component_type="random_walk", n_timesteps=100, rng=builder_rng, step_size=0.2)`

4. **Dispatch**: `generate_component()` looks up "random_walk" in the `GENERATOR_FUNCS` dictionary and calls:
   ```python
   generate_random_walk(n_timesteps=100, rng=builder_rng, step_size=0.2)
   ```

5. **Generation**: The generator function produces and returns a numpy array

6. **Assembly**: The builder combines all generated components into the final time series according to the specified composition (signals + features)

### The GENERATOR_FUNCS Dictionary

At the bottom of `generators.py`, there's a dictionary that maps component type names to generator functions:

```python
GENERATOR_FUNCS = {
    "constant": generate_constant,
    "random_walk": generate_random_walk,
    "gaussian": generate_gaussian,
    # ...
}
```

This is the lookup table that `generate_component()` uses to find the right generator for each component type.

## The Two-Function Pattern

XAITimeSynth uses a two-function pattern: one component function and one generator function per component type.

### Component Functions (User-Facing)

**Location**: `components.py`
**Purpose**: Provide a clean, user-friendly API for defining components
**Signature**: Takes only user-configurable parameters (no internal stuff)
**Returns**: A dictionary with the component specification

Example:
```python
def random_walk(step_size: float = 0.1, **kwargs) -> Dict[str, Any]:
    """Create a definition for a random walk signal component.

    Args:
        step_size: Standard deviation of random steps. Defaults to 0.1.
        **kwargs: Additional parameters.

    Returns:
        Dict defining the 'random_walk' component with its parameters.
    """
    return {"type": "random_walk", "step_size": step_size, **kwargs}
```

### Generator Functions (Internal)

**Location**: `generators.py`
**Purpose**: Actually create the numpy arrays with the time series data
**Signature**: Always follows a standard pattern (see below)
**Returns**: A 1D numpy array

Example:
```python
def generate_random_walk(
    n_timesteps: int,
    step_size: float = 0.1,
    rng: Optional[np.random.RandomState] = None,
    length: Optional[int] = None,
    **kwargs,
) -> np.ndarray:
    """Generate a random walk time series.

    Args:
        n_timesteps: Nominal length of the time series context.
        step_size: Standard deviation of random steps. Defaults to 0.1.
        rng: Random number generator instance.
        length: Actual desired length (overrides n_timesteps if provided).
        **kwargs: Additional parameters for compatibility.

    Returns:
        A 1D numpy array of the specified length.
    """
    if rng is None:
        rng = np.random.RandomState()
    output_length = length if length is not None else n_timesteps
    steps = rng.normal(0, step_size, output_length)
    return np.cumsum(steps)
```

### Why Two Functions?

This separation provides several benefits:

1. **Clean API**: Users don't see internal parameters like `rng`, `n_timesteps`, or `length`
2. **Flexibility**: Component definitions can be created, stored, modified, and reused before generation
3. **Type Safety**: Component functions can be registered as "signal", "feature", or "both"
4. **Documentation**: Each function can have targeted documentation for its audience
5. **Direct Access**: Advanced users can call generator functions directly if needed

## Standard Parameters Explained

All generator functions must follow a standardized signature. Here's why each standard parameter exists:

### Required Parameters (in order)

#### 1. `n_timesteps: int`

- **Purpose**: The total length of the time series being generated
- **Why it's needed**:
  - Generators may need to scale frequencies or patterns to fit the full series length
  - Even when generating a partial feature (using `length`), knowing the full context helps maintain correct scaling
  - Example: A sine wave with `period=10` should complete the same number of cycles whether it's a full signal or a localized feature

#### 2. Standard Generator-Specific Parameters

- These are the parameters that control the generator's behavior
- Examples: `step_size` for random_walk, `mu` and `sigma` for gaussian
- Placed after `n_timesteps` but before the standard optional parameters

#### 3. `rng: Optional[np.random.RandomState]`

- **Purpose**: Provides reproducible randomness
- **Why it's in every generator**:
  - **Uniform API**: All generators can be called the same way, making the internal dispatch simple
  - **Reproducibility**: The builder can pass its RNG to all generators for reproducible datasets
  - **Future-proofing**: Even deterministic generators can be extended with random variations later
- **For deterministic generators**: Simply ignore this parameter (but still include it in the signature)

#### 4. `length: Optional[int]`

- **Purpose**: Specifies the actual output length when different from `n_timesteps`
- **Why it's needed**:
  - **Signals vs Features**: Signals span the full series; features are localized to a window
  - **Builder flexibility**: The builder can request specific lengths for positioned features
  - **Standard logic**: All generators use the same pattern:
    ```python
    output_length = length if length is not None else n_timesteps
    ```

#### 5. `**kwargs`

- **Purpose**: Catches any extra parameters passed by the builder
- **Why it's needed**:
  - **Forward compatibility**: New builder features won't break existing generators
  - **Flexibility**: Users can pass custom parameters without breaking the API
  - **Tolerates extras**: If a component definition has extra keys, they won't cause errors

### Standard Return Type

All generators must return `np.ndarray` - a 1D numpy array of floats with length equal to `output_length`.

## Adding a New Generator

Follow these steps to add a new generator to the package:

### Step 1: Implement the Generator Function

Add your generator to `generators.py`. Follow this template:

```python
def generate_your_component(
    n_timesteps: int,
    # Your custom parameters here (with defaults)
    param1: type = default_value,
    param2: type = default_value,
    # Standard optional parameters
    rng: Optional[np.random.RandomState] = None,
    length: Optional[int] = None,
    **kwargs,
) -> np.ndarray:
    """Generate your custom component.

    Brief description of what this generator creates.

    Args:
        n_timesteps: The nominal length of the time series context. The actual output
            length is determined by `length` parameter if provided, otherwise `n_timesteps`.
        param1: Description of parameter 1.
        param2: Description of parameter 2.
        rng: Random number generator instance. [If unused, add: "Included for API
            consistency but unused in this deterministic generator."] Defaults to None.
        length: The exact desired length of the output time series array.
            If provided, this overrides `n_timesteps`. If None, `n_timesteps` is used.
            Typically provided by the TimeSeriesBuilder. Defaults to None.
        **kwargs: Catches unused parameters passed by TimeSeriesBuilder for compatibility.

    Returns:
        np.ndarray: A 1D numpy array of the specified length containing [description].

    Example:
        >>> rng = np.random.RandomState(42)
        >>> generate_your_component(n_timesteps=10, param1=value1, rng=rng)
        array([...])
    """
    # Handle RNG if needed
    if rng is None:
        rng = np.random.RandomState()

    # Determine output length
    output_length = length if length is not None else n_timesteps

    # Your implementation here
    result = np.zeros(output_length)  # Or your actual logic
    # ... calculation logic ...

    return result
```

**Key Requirements**:
- Function name must start with `generate_`
- Parameters must be in the order: `n_timesteps`, custom params, `rng`, `length`, `**kwargs`
- Must return a 1D numpy array of length `output_length`
- Must handle both `length=None` (full signal) and `length=N` (partial feature) cases

### Step 2: Register in GENERATOR_FUNCS Dictionary

At the end of `generators.py`, add your generator to the lookup dictionary:

```python
GENERATOR_FUNCS = {
    # ...existing generators...
    "your_component": generate_your_component,
}
```

The key string (e.g., `"your_component"`) is the component type that will be used in component definitions.

### Step 3: Create the Component Function

Add the user-facing function to `components.py`:

```python
def your_component(param1: type = default_value, param2: type = default_value, **kwargs) -> Dict[str, Any]:
    """Create a definition for your custom component.

    Brief user-friendly description of what this component does and when to use it.

    Args:
        param1: Description of parameter 1 from a user perspective.
        param2: Description of parameter 2 from a user perspective.
        **kwargs: Additional parameters passed to the generator during build time.

    Returns:
        Dict[str, Any]: A dictionary defining the 'your_component' component with its parameters.

    Example:
        >>> comp = your_component(param1=value1, param2=value2)
        >>> comp['type']
        'your_component'
    """
    return {"type": "your_component", "param1": param1, "param2": param2, **kwargs}
```

**Key Requirements**:
- Function name should match the GENERATOR_FUNCS key (without the `generate_` prefix)
- Takes only user-configurable parameters (no `n_timesteps`, `rng`, or `length`)
- Returns a dictionary with at least a `"type"` key matching the generator name
- All parameters should be included in the returned dictionary

### Step 4: Register the Component

In `__init__.py`, import and register your component:

```python
from .components import (
    # ...existing imports...
    your_component,
)

# In the registration section
register_component(your_component, "signal")  # Or "feature" or "both"
```

Component types:
- `"signal"`: Can be used with `.add_signal()` (full-length background patterns)
- `"feature"`: Can be used with `.add_feature()` (localized discriminative patterns)
- `"both"`: Can be used with either method

### Step 5: Export the Component

Add your component to the `__all__` list in `__init__.py`:

```python
__all__ = [
    # ...existing exports...
    "your_component",
]
```

This makes it importable directly from the package:
```python
from xaitimesynth import your_component
```

## Complete Example: Sine Wave Generator

Here's a complete example showing how to add a sine wave generator:

### In `generators.py`:

```python
def generate_sine_wave(
    n_timesteps: int,
    frequency: float = 0.1,
    amplitude: float = 1.0,
    phase: float = 0.0,
    rng: Optional[np.random.RandomState] = None,
    length: Optional[int] = None,
    **kwargs,
) -> np.ndarray:
    """Generate a sine wave time series.

    Creates a sinusoidal signal with specified frequency, amplitude, and phase.
    The frequency is relative to the full time series length (n_timesteps).

    Args:
        n_timesteps: The nominal length of the time series context.
        frequency: Frequency of the sine wave as a fraction of the sampling rate.
            Defaults to 0.1.
        amplitude: Peak amplitude of the sine wave. Defaults to 1.0.
        phase: Phase shift in radians. Defaults to 0.0.
        rng: Random number generator instance. Included for API consistency
            but unused in this deterministic generator. Defaults to None.
        length: The exact desired length of the output. If None, uses n_timesteps.
            Defaults to None.
        **kwargs: Catches unused parameters for compatibility.

    Returns:
        np.ndarray: A 1D array of the specified length containing a sine wave.

    Example:
        >>> import numpy as np
        >>> wave = generate_sine_wave(n_timesteps=100, frequency=0.1, amplitude=2.0)
        >>> wave.shape
        (100,)
        >>> np.max(wave)
        2.0
    """
    # Determine output length
    output_length = length if length is not None else n_timesteps

    # Generate time indices
    t = np.arange(output_length)

    # Create sine wave (frequency is relative to n_timesteps for consistent scaling)
    return amplitude * np.sin(2 * np.pi * frequency * t / n_timesteps + phase)


# Add to GENERATOR_FUNCS dictionary
GENERATOR_FUNCS = {
    # ...existing generators...
    "sine_wave": generate_sine_wave,
}
```

### In `components.py`:

```python
def sine_wave(
    frequency: float = 0.1, amplitude: float = 1.0, phase: float = 0.0, **kwargs
) -> Dict[str, Any]:
    """Create a definition for a sine wave component.

    Generates a sinusoidal pattern useful for creating periodic signals or
    oscillating features. Can be used as both a full-length signal or a
    localized feature.

    Args:
        frequency: Frequency of the sine wave as a fraction of the sampling rate.
            Defaults to 0.1 (one cycle every 10 timesteps).
        amplitude: Peak amplitude of the sine wave. Defaults to 1.0.
        phase: Phase shift in radians. Use this to offset the starting point
            of the wave. Defaults to 0.0.
        **kwargs: Additional parameters passed to the generator during build time.

    Returns:
        Dict[str, Any]: A dictionary defining the 'sine_wave' component.

    Example:
        >>> from xaitimesynth import TimeSeriesBuilder, sine_wave
        >>> dataset = (
        ...     TimeSeriesBuilder(n_timesteps=100, n_samples=50)
        ...     .for_class(0)
        ...     .add_signal(sine_wave(frequency=0.05, amplitude=2.0))
        ...     .build()
        ... )
    """
    return {
        "type": "sine_wave",
        "frequency": frequency,
        "amplitude": amplitude,
        "phase": phase,
        **kwargs,
    }
```

### In `__init__.py`:

```python
from .components import (
    # ...existing imports...
    sine_wave,
)

# ...

# In the registration section
register_component(sine_wave, "both")  # Can be used as signal or feature

# ...

__all__ = [
    # ...existing exports...
    "sine_wave",
]
```

### Usage:

```python
from xaitimesynth import TimeSeriesBuilder, sine_wave, gaussian

# As a signal (full-length background)
dataset = (
    TimeSeriesBuilder(n_timesteps=200, n_samples=100)
    .for_class(0)
    .add_signal(sine_wave(frequency=0.05, amplitude=1.5))
    .add_signal(gaussian(sigma=0.1))
    .build()
)

# As a feature (localized pattern)
dataset = (
    TimeSeriesBuilder(n_timesteps=200, n_samples=100)
    .for_class(0)
    .add_signal(gaussian(sigma=0.5))
    .for_class(1)
    .add_signal(gaussian(sigma=0.5))
    .add_feature(sine_wave(frequency=0.2, amplitude=2.0), start_pct=0.3, end_pct=0.7)
    .build()
)
```

## Testing Your Generator

After implementing your generator, you should test it thoroughly:

### 1. Unit Tests for the Generator Function

Test the generator function directly in `tests/test_generators.py`:

```python
def test_generate_sine_wave_shape():
    """Test sine_wave generator produces correct output shape."""
    result = generate_sine_wave(n_timesteps=100, frequency=0.1)
    assert result.shape == (100,)
    assert result.dtype == np.float64

def test_generate_sine_wave_with_length():
    """Test sine_wave generator respects length parameter."""
    result = generate_sine_wave(n_timesteps=100, length=50, frequency=0.1)
    assert result.shape == (50,)

def test_generate_sine_wave_amplitude():
    """Test sine_wave generator produces correct amplitude."""
    amplitude = 2.5
    result = generate_sine_wave(n_timesteps=100, amplitude=amplitude, frequency=0.1)
    assert np.max(np.abs(result)) == pytest.approx(amplitude, rel=1e-10)

def test_generate_sine_wave_reproducible():
    """Test sine_wave generator is deterministic."""
    result1 = generate_sine_wave(n_timesteps=50, frequency=0.1)
    result2 = generate_sine_wave(n_timesteps=50, frequency=0.1)
    np.testing.assert_array_equal(result1, result2)
```

### 2. Unit Tests for the Component Function

Test the component function in `tests/test_components.py`:

```python
def test_sine_wave_default():
    """Test sine_wave component with default parameters."""
    comp = sine_wave()
    assert comp["type"] == "sine_wave"
    assert comp["frequency"] == 0.1
    assert comp["amplitude"] == 1.0
    assert comp["phase"] == 0.0

def test_sine_wave_custom():
    """Test sine_wave component with custom parameters."""
    comp = sine_wave(frequency=0.2, amplitude=3.0, phase=np.pi/2)
    assert comp["type"] == "sine_wave"
    assert comp["frequency"] == 0.2
    assert comp["amplitude"] == 3.0
    assert comp["phase"] == np.pi/2
```

### 3. Integration Tests with TimeSeriesBuilder

Test that your component works correctly in the builder pipeline:

```python
def test_sine_wave_in_builder():
    """Test sine_wave component works in TimeSeriesBuilder."""
    dataset = (
        TimeSeriesBuilder(n_timesteps=100, n_samples=10, random_state=42)
        .for_class(0)
        .add_signal(sine_wave(frequency=0.1, amplitude=1.0))
        .build()
    )

    assert dataset["X"].shape == (10, 1, 100)
    # Verify the signal has the expected properties
    signal = dataset["X"][0, 0, :]
    assert np.max(np.abs(signal)) == pytest.approx(1.0, rel=1e-10)
```

## Alternative: Quick Extension with Decorators

For quick custom extensions or prototyping, you can use the `@register_component_generator` decorator. This combines steps 2-4 into a single decorator:

```python
# In generators.py or your own module
from xaitimesynth.registry import register_component_generator

@register_component_generator(component_type="both")
def generate_sine_wave(
    n_timesteps: int,
    frequency: float = 0.1,
    amplitude: float = 1.0,
    phase: float = 0.0,
    rng: Optional[np.random.RandomState] = None,
    length: Optional[int] = None,
    **kwargs,
) -> np.ndarray:
    """Generate a sine wave signal."""
    output_length = length if length is not None else n_timesteps
    t = np.arange(output_length)
    return amplitude * np.sin(2 * np.pi * frequency * t / n_timesteps + phase)
```

**Limitations of the decorator approach**:
- Component function docstrings won't be visible to users
- Less control over the component function API
- Not recommended for stable package integration

**Best for**:
- Quick experiments
- User-defined custom generators
- Prototyping new components before full integration

## Summary

Adding a generator to xaitimesynth involves:

1. **Generator function** in `generators.py` - implements the actual data generation
2. **Component function** in `components.py` - provides user-friendly API
3. **GENERATOR_FUNCS registration** - enables lookup
4. **Component registration** in `__init__.py` - makes it available to users
5. **Export** in `__all__` - enables direct import
6. **Tests** in `tests/` - ensures correctness

The two-function pattern with standardized signatures provides a clean separation between the user-facing API and the internal implementation, while the registration system keeps everything organized and discoverable.
