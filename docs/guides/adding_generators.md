# Adding a New Generator to XAITimeSynth: Step-by-Step Guide

This guide walks you through the process of adding a new generator function to the xaitimesynth package. By following these steps, you'll be able to create custom time series components that can be used in the TimeSeriesBuilder API.

## Overview of the Generator System

The xaitimesynth package has a flexible system for creating and registering components:

- **Generator Functions**: Define how to create the actual time series data (in generators.py)
- **Component Functions**: Define how to create component definitions (in components.py)
- **Registration System**: Connects generators to components and makes them available to users

## Understanding The Two Functions and Their Use Cases

It's important to understand the different roles of the two function types in the XAITimeSynth architecture:

1. **Component Functions** (e.g., `constant`, `random_walk`):
   - Used directly by end users in the TimeSeriesBuilder pipeline
   - Called when constructing datasets through the fluent API
   - Need complete docstrings to help users understand parameters when coding
   - Example usage in the pipeline:
     ```python
     dataset = (
         TimeSeriesBuilder(n_timesteps=100, n_samples=200)
         .for_class(0)
         .add_signal(random_walk(step_size=0.2))
         .add_signal(gaussian(sigma=0.1), role="noise")
         .for_class(1)
         .add_feature(shapelet(scale=1.0), start_pct=0.4)
         .build()
     )
     ```

2. **Generator Functions** (e.g., `generate_constant`, `generate_random_walk`):
   - Used internally by the system to actually create the data
   - Can be used directly by advanced users who need more control
   - Allow manual generation of specific waveforms outside the pipeline
   - Example direct usage:
     ```python
     import numpy as np
     from xaitimesynth.generators import generate_sine_wave

     # Manually create a sine wave
     rng = np.random.RandomState(42)
     wave = generate_sine_wave(
         n_timesteps=100,
         rng=rng,
         frequency=0.05,
         amplitude=2.0
     )
     ```

This is why it's important to provide complete docstrings for both functions - they serve different audiences and use cases.

## Design Choices

The generator/component architecture follows specific design principles that enable flexibility across different roles:

### Standardized Function Signatures

All generator functions follow this consistent signature pattern:
```python
def generate_xxx(
    n_timesteps: int,
    rng: np.random.RandomState,
    length: Optional[int] = None,
    ...other parameters...
) -> np.ndarray:
```

#### Why include `rng` in every generator?

Even for deterministic generators like `constant` that don't use randomness:

1. **Uniform API**: A consistent interface makes generators interchangeable and simplifies internal systems
2. **Future-proofing**: Enables adding random variations to any component later
3. **Generic dispatch**: The internal `generate_component` function can call any generator without special cases

#### Why include `length` in every generator?

1. **Role flexibility**: Components can be used as signals (full length) or features (partial length)
2. **Generic feature creation**: The builder can request specific lengths for localized patterns
3. **Adapter pattern**: Each generator adapts to the required length with the logic:
   ```python
   if length is None:
       length = n_timesteps
   ```

### Separation of Concerns

The system separates the "what" from the "how":

- **Component functions** (`components.py`): Define what the user wants, with simple parameters
- **Generator functions** (`generators.py`): Define how to generate the actual data
- **Builder** (`builder.py`): Handles composition, positioning, and combining components

This separation makes the API user-friendly while maintaining internal flexibility.

## Step 1: Add the Generator Function

First, add your generator function to generators.py. Follow this function signature pattern:

```python
def generate_your_component(
    n_timesteps: int,
    rng: np.random.RandomState,
    length: Optional[int] = None,  # For features, or None for full signals
    param1: type = default_value,  # Your custom parameters
    param2: type = default_value,
    **kwargs,
) -> np.ndarray:
    """Generate your custom component.

    Args:
        n_timesteps: Length of time series.
        rng: Random number generator.
        length: Length of feature in timesteps. If None, uses n_timesteps.
        param1: Description of parameter 1.
        param2: Description of parameter 2.
        **kwargs: Additional parameters.

    Returns:
        Generated component as a numpy array.
    """
    # Your implementation here
    # For a signal, return an array of length n_timesteps
    # For a feature, return an array of length `length`

    # Example implementation:
    result = np.zeros(length if length is not None else n_timesteps)
    # ... calculation logic ...
    return result
```

Key requirements:

- Function name should start with generate_ (e.g., generate_your_component)
- First three parameters must be n_timesteps, rng, and either length (for features) or another parameter
- Return a numpy array of the appropriate length
- Include detailed docstrings with parameter descriptions

## Step 2: Add to GENERATOR_FUNCS Dictionary

At the end of generators.py, add your generator to the GENERATOR_FUNCS dictionary:

```python
GENERATOR_FUNCS = {
    # ...existing generators...
    "your_component": generate_your_component,
}
```

## Step 3: Create a Component Function (Option 1: Manual - Preferred for Package Integration)

For stable integration into the package, manually adding a component function to components.py is the preferred approach. This method exposes proper docstrings with parameters to users, making the API clear and discoverable:

```python
def your_component(param1: type = default_value, param2: type = default_value, **kwargs) -> Dict[str, Any]:
    """Create a your_component component.

    Args:
        param1: Description of parameter 1.
        param2: Description of parameter 2.
        **kwargs: Additional parameters.

    Returns:
        Component definition dictionary.
    """
    return {"type": "your_component", "param1": param1, "param2": param2, **kwargs}
```

This approach ensures that:
- Function signature is properly exposed in IDE tooltips and documentation
- Parameter descriptions are available to users
- Type hints guide correct usage

## Step 4: Register the Component (if using Option 1)

In __init__.py, import and register your component:

```python
from .components import your_component  # Add this import

# Add this registration
register_component(your_component, "signal")  # Or "feature" or "both"
```

## Step 3+4 Alternative: Use the Decorator (Option 2: For Quick Custom Extensions)

The decorator approach is primarily intended for users who want to quickly add custom generators without modifying multiple files. This is useful for extensions but not recommended for stable package integration:

```python
# In generators.py
from .registry import register_component_generator

@register_component_generator(component_type="signal")  # Or "feature" or "both"
def generate_your_component(
    n_timesteps: int,
    rng: np.random.RandomState,
    length: Optional[int] = None,
    param1: type = default_value,
    param2: type = default_value,
    **kwargs,
) -> np.ndarray:
    """Generate your custom component."""
    # Implementation
```

**Note:** While this approach is convenient, it has limitations:
- The docstrings from the generator function won't be visible in the component API
- Parameter descriptions won't be accessible when users call the component function
- It's harder to customize the component function behavior separately from the generator

## Step 5: Update __init__.py Exports

If using Option 1 or if you want to make the component directly importable, add it to the exports in __init__.py:

```python
from .components import (
    # ...existing components...
    your_component,
)
```

## Step 6: Document Your Component

Add your new component to the package documentation, including:

- Description of what it does
- Parameters and their effects
- Example usage
- Visual example if possible

## Reference: Generator Function Requirements

All generator functions must:

- Take n_timesteps and rng as the first two parameters
- For features, take length as the third parameter
- Return a numpy array of the appropriate length
- Have properly documented parameters
- Be registered in the GENERATOR_FUNCS dictionary

## Reference: Component Function Requirements

All component functions must:

- Take the same parameters as the generator (except for n_timesteps, rng, and length)
- Return a dictionary with at least a "type" key matching the generator name
- Include all parameters as keys in the returned dictionary
- Be registered using either register_component or register_component_generator

## Example: Adding a Sine Wave Generator

Here's a complete example showing both approaches for adding a sine wave generator:

### Option 1: Manual Approach (Preferred for Package Integration)

```python
# In generators.py
def generate_sine_wave(
    n_timesteps: int,
    rng: np.random.RandomState,
    length: Optional[int] = None,
    frequency: float = 0.1,
    amplitude: float = 1.0,
    phase: float = 0.0,
    **kwargs,
) -> np.ndarray:
    """Generate a sine wave signal.

    Args:
        n_timesteps: Length of time series.
        rng: Random number generator.
        length: Length of feature in timesteps. If None, uses n_timesteps.
        frequency: Frequency of the sine wave.
        amplitude: Amplitude of the sine wave.
        phase: Phase shift in radians.
        **kwargs: Additional parameters.

    Returns:
        Sine wave signal vector.
    """
    # If length is not provided, use the entire time series length
    if length is None:
        length = n_timesteps

    t = np.arange(length)
    return amplitude * np.sin(2 * np.pi * frequency * t / n_timesteps + phase)

# Add to the GENERATOR_FUNCS dictionary
GENERATOR_FUNCS = {
    # ...existing generators...
    "sine_wave": generate_sine_wave,
}

# In components.py
def sine_wave(frequency: float = 0.1, amplitude: float = 1.0, phase: float = 0.0, **kwargs) -> Dict[str, Any]:
    """Create a sine wave component.

    Args:
        frequency: Frequency of the sine wave.
        amplitude: Amplitude of the sine wave.
        phase: Phase shift in radians.
        **kwargs: Additional parameters.

    Returns:
        Component definition dictionary.
    """
    return {"type": "sine_wave", "frequency": frequency, "amplitude": amplitude, "phase": phase, **kwargs}

# In __init__.py
from .components import sine_wave
register_component(sine_wave, "both")
```

### Option 2: Decorator Approach (For Quick Custom Extensions)

```python
# In generators.py
from .registry import register_component_generator

@register_component_generator(component_type="both")
def generate_sine_wave(
    n_timesteps: int,
    rng: np.random.RandomState,
    length: Optional[int] = None,
    frequency: float = 0.1,
    amplitude: float = 1.0,
    phase: float = 0.0,
    **kwargs,
) -> np.ndarray:
    """Generate a sine wave signal.

    Args:
        n_timesteps: Length of time series.
        rng: Random number generator.
        length: Length of feature in timesteps. If None, uses n_timesteps.
        frequency: Frequency of the sine wave.
        amplitude: Amplitude of the sine wave.
        phase: Phase shift in radians.
        **kwargs: Additional parameters.

    Returns:
        Sine wave signal vector.
    """
    # If length is not provided, use the entire time series length
    if length is None:
        length = n_timesteps

    t = np.arange(length)
    return amplitude * np.sin(2 * np.pi * frequency * t / n_timesteps + phase)
```

With this approach, you can now use your component in the builder API:

```python
from xaitimesynth import TimeSeriesBuilder, sine_wave

dataset = (
    TimeSeriesBuilder(n_timesteps=100, n_samples=1000)
    .for_class(0)
    .add_signal(sine_wave(frequency=0.05, amplitude=2.0))
    .build()
)
```

## Conclusion

By following these steps, you can extend the xaitimesynth package with custom generators for creating specific time series patterns. The package's registry system makes it easy to add new components that seamlessly integrate with the existing API.
