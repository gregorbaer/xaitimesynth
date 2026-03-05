# Adding Custom Generators

This guide explains how to create custom time series components for use in the TimeSeriesBuilder API. There are two approaches: the `manual()` component for one-off patterns (see [Custom data generation](../examples/custom_generators.ipynb)), and registering a proper reusable component covered here.

For defining new custom data generators, the **decorator approach** below or [using manual() function](../examples/custom_generators.ipynb) are easiest and quickest. For reusable generators integrated into the package, follow the two-function pattern: a generator function in `generators.py` + a component function in `components.py`, then register in `__init__.py`.


## Quick Extension with Decorators

For quick custom extensions or prototyping, you can use the `@register_component_generator` decorator. This simplifies a lot of the steps below into a single decorator:

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

You can then use your registered component directly in the `TimeSeriesBuilder`. Pass the component as a dictionary with the registered type name, or define a small helper function for a cleaner call site:

```python
from xaitimesynth import TimeSeriesBuilder, gaussian_noise

# Option 1: pass a dict directly
dataset = (
    TimeSeriesBuilder(n_timesteps=200, n_samples=50)
    .for_class(0)
    .add_signal({"type": "sine_wave", "frequency": 0.05, "amplitude": 1.5})
    .add_signal(gaussian_noise(sigma=0.1))
    .for_class(1)
    .add_signal(gaussian_noise(sigma=0.1))
    .add_feature({"type": "sine_wave", "frequency": 0.2, "amplitude": 2.0}, start_pct=0.3, end_pct=0.7)
    .build()
)

# Option 2: define a helper for a cleaner API (mirrors the two-function pattern)
def sine_wave(frequency=0.1, amplitude=1.0, phase=0.0, **kwargs):
    return {"type": "sine_wave", "frequency": frequency, "amplitude": amplitude, "phase": phase, **kwargs}

dataset = (
    TimeSeriesBuilder(n_timesteps=200, n_samples=50)
    .for_class(0)
    .add_signal(sine_wave(frequency=0.05, amplitude=1.5))
    .add_signal(gaussian_noise(sigma=0.1))
    .for_class(1)
    .add_signal(gaussian_noise(sigma=0.1))
    .add_feature(sine_wave(frequency=0.2, amplitude=2.0), start_pct=0.3, end_pct=0.7)
    .build()
)
```

**Limitations of the decorator approach**:

- Component function docstrings won't be visible to users
- Less control over the component function API
- Not recommended for stable package integration

**Best for**:

- Quick experiments
- User-defined custom generators
- Prototyping new components before full integration


## The Two-Function Pattern

**⚠️Note:** You likely will only need or want to read the below if you're thinking of more permantently adding a data generating function to this package either locally, or by contributing to the package. Otherwise it's likely too much detail, and you don't need to know the internals to use the package productively.

XAITimeSynth uses a two-function pattern: one component function and one generator function per component type. This is necessary internally as the `TimeSeriesBuilder` passes the dictionary definitions along and creates the data based on the generator functions from the component functions.

### Component Functions (User-Facing)

- **Location**: `components.py`
- **Purpose**: Provide a clean, user-friendly API for defining components
- **Signature**: Takes only user-configurable parameters (no internal stuff)
- **Returns**: A dictionary with the component specification

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

- **Location**: `generators.py`
- **Purpose**: Actually create the numpy arrays with the time series data
- **Signature**: Always follows a standard pattern (see below)
- **Returns**: A 1D numpy array

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
- Examples: `step_size` for random_walk, `mu` and `sigma` for gaussian_noise
- Placed after `n_timesteps` but before the standard optional parameters

#### 3. `rng: Optional[np.random.RandomState]`

- **Purpose**: Provides reproducible randomness
- **Why it's in every generator**:
    - Uniform API: All generators can be called the same way, making the internal dispatch simple
    - Reproducibility: The builder can pass its RNG to all generators for reproducible datasets
    - Future-proofing: Even deterministic generators can be extended with random variations later
- For deterministic generators: Simply ignore this parameter (but still include it in the signature)

#### 4. `length: Optional[int]`

- **Purpose**: Specifies the actual output length when different from `n_timesteps`
- **Why it's needed**:
    - Signals vs Features: Signals span the full series; features are localized to a window
    - Builder flexibility: The builder can request specific lengths for positioned features
    - Standard logic: All generators use the same pattern:
    ```python
    output_length = length if length is not None else n_timesteps
    ```

#### 5. `**kwargs`

- **Purpose**: Catches any extra parameters passed by the builder
- **Why it's needed**:
    - Forward compatibility: New builder features won't break existing generators
    - Flexibility: Users can pass custom parameters without breaking the API
    - Tolerates extras: If a component definition has extra keys, they won't cause errors

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

### Step 4: Register and Export (for package integration)

If adding to the package itself (rather than user-side code), add to `__init__.py`:

```python
from .components import your_component
register_component(your_component, "signal")  # Or "feature" or "both"
# Add "your_component" to __all__
```

For user-side use, the [decorator approach](#alternative-quick-extension-with-decorators) below is simpler.

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

### Usage:

```python
from xaitimesynth import TimeSeriesBuilder, sine_wave, gaussian_noise

# As a signal (full-length background)
dataset = (
    TimeSeriesBuilder(n_timesteps=200, n_samples=100)
    .for_class(0)
    .add_signal(sine_wave(frequency=0.05, amplitude=1.5))
    .add_signal(gaussian_noise(sigma=0.1))
    .build()
)

# As a feature (localized pattern)
dataset = (
    TimeSeriesBuilder(n_timesteps=200, n_samples=100)
    .for_class(0)
    .add_signal(gaussian_noise(sigma=0.5))
    .for_class(1)
    .add_signal(gaussian_noise(sigma=0.5))
    .add_feature(sine_wave(frequency=0.2, amplitude=2.0), start_pct=0.3, end_pct=0.7)
    .build()
)
```
