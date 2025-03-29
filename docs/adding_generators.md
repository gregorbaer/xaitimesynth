# Adding a New Generator to XAITimeSynth: Step-by-Step Guide

This guide walks you through the process of adding a new generator function to the xaitimesynth package. By following these steps, you'll be able to create custom time series components that can be used in the TimeSeriesBuilder API.

## Overview of the Generator System

The xaitimesynth package has a flexible system for creating and registering components:

- **Generator Functions**: Define how to create the actual time series data (in generators.py)
- **Component Functions**: Define how to create component definitions (in components.py)
- **Registration System**: Connects generators to components and makes them available to users

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

## Step 3: Create a Component Function (Option 1: Manual)

You can manually add a component function to components.py:

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

## Step 4: Register the Component (if using Option 1)

In __init__.py, import and register your component:

```python
from .components import your_component  # Add this import

# Add this registration
register_component(your_component, "signal")  # Or "feature" or "both"
```

## Step 3+4 Alternative: Use the Decorator (Option 2: Automatic)

Instead of manually creating a component function, you can use the @register_component_generator decorator:

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

This automatically:

- Creates a component function named your_component
- Registers it in the appropriate registries
- Makes it available in the module namespace

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

Here's a complete example of adding a sine wave generator:

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

# The decorator automatically adds it to GENERATOR_FUNCS and creates the component function
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