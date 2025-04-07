import inspect
from functools import wraps
from typing import Any, Callable, Dict, Optional

# Global registry for components
_COMPONENT_REGISTRY = {}
_SIGNAL_COMPONENTS = set()
_FEATURE_COMPONENTS = set()


def register_component(
    component_func: Optional[Callable] = None, component_type: str = "both"
) -> Callable:
    """Register a component function in the registry.

    Adds the component to the global registry and appropriate component type sets.

    Args:
        component_func: Function that creates a component. When used as a decorator
            without arguments, this will be the decorated function.
        component_type: Type of component. Must be one of "signal", "feature", or "both".
            Determines which registries the component is added to.

    Returns:
        Callable: The original function, unchanged but now registered.
    """

    def decorator(func):
        name = func.__name__
        _COMPONENT_REGISTRY[name] = func

        if component_type in ("signal", "both"):
            _SIGNAL_COMPONENTS.add(name)
        if component_type in ("feature", "both"):
            _FEATURE_COMPONENTS.add(name)

        return func

    if component_func is None:
        return decorator
    return decorator(component_func)


def register_component_generator(
    name: Optional[str] = None, component_type: str = "both", **default_overrides
) -> Callable:
    """Register a generator function and create a component function.

    This decorator simplifies component creation by automatically:
    1. Creating a component definition function based on the generator
    2. Registering both in the appropriate registries
    3. Exposing the component function in the generator's module namespace

    The component name is determined as follows:
    - If the 'name' parameter is provided, that exact name is used
    - Otherwise, the function's name is used, with the 'generate_' prefix removed if present

    Args:
        name: Optional custom name for the component. If not provided,
            derives name from the generator function name.
        component_type: Type of component. Must be one of "signal", "feature", or "both".
            Determines which registries the component is added to.
        **default_overrides: Override default parameter values for the component function.
            These will replace the default values in the generator function.

    Returns:
        Callable: Decorator function that processes the generator function.

    Examples:
        # Basic usage - creates component named "sine_wave"
        @register_component_generator()
        def generate_sine_wave(n_timesteps, rng, length=None, frequency=0.1):
            # Implementation...

        # Explicit naming - creates component named "my_sine"
        @register_component_generator(name="my_sine")
        def generate_wave(n_timesteps, rng, length=None, frequency=0.1):
            # Implementation...

        # Override default values - component default amplitude will be 2.0
        @register_component_generator(amplitude=2.0)
        def generate_sine_wave(n_timesteps, rng, length=None, amplitude=1.0):
            # Implementation...
    """

    def decorator(generator_func: Callable) -> Callable:
        # Get generator function signature
        sig = inspect.signature(generator_func)
        params = sig.parameters

        # Extract parameter defaults (skip first three: n_timesteps, rng, length)
        param_defaults = {}
        for param_name, param in list(params.items())[3:]:
            if param.default is not inspect.Parameter.empty:
                param_defaults[param_name] = param.default

        # Override defaults if provided
        param_defaults.update(default_overrides)

        # Determine component name
        component_name = name or generator_func.__name__
        if component_name.startswith("generate_"):
            component_name = component_name[9:]  # Remove "generate_" prefix

        # Create the component function
        def component_func(**kwargs):
            """Create a component definition.

            Args:
                **kwargs: Component parameters to override defaults.

            Returns:
                Dict[str, Any]: Component definition dictionary with type and parameters.
            """
            # Start with defaults
            component_def = {"type": component_name, **param_defaults}
            # Update with provided kwargs
            component_def.update(kwargs)
            return component_def

        # Set the component function name and docstring
        component_func.__name__ = component_name
        component_func.__doc__ = f"""Create a {component_name} component.
        
        Args:
            **kwargs: Component parameters to override defaults.
            
        Returns:
            Dict[str, Any]: Component definition dictionary.
        """

        # Register the generator function
        from .generators import GENERATOR_FUNCS

        GENERATOR_FUNCS[component_name] = generator_func

        # Attach parameter defaults to the component function
        component_func._param_defaults = param_defaults

        # Register the component function
        _COMPONENT_REGISTRY[component_name] = component_func

        if component_type in ("signal", "both"):
            _SIGNAL_COMPONENTS.add(component_name)
        if component_type in ("feature", "both"):
            _FEATURE_COMPONENTS.add(component_name)

        # Return the original generator function
        @wraps(generator_func)
        def wrapper(*args, **kwargs):
            return generator_func(*args, **kwargs)

        # Register the auto-generated component function globally
        import sys

        # Get the module where the generator function is defined
        module_name = generator_func.__module__

        # Fix for tests: use the module where the function is actually defined
        module = sys.modules[module_name]

        # Set the attribute in that module
        setattr(module, component_name, component_func)

        return wrapper

    return decorator


def list_components() -> Dict[str, Callable]:
    """List all registered components.

    Provides a copy of the complete component registry.

    Returns:
        Dict[str, Callable]: Dictionary mapping component names to their creation functions.
    """
    return _COMPONENT_REGISTRY.copy()


def list_signal_components() -> Dict[str, Callable]:
    """List components commonly used as signals.

    Returns a dictionary of all components registered with the "signal" type.

    Returns:
        Dict[str, Callable]: Dictionary mapping signal component names to their creation functions.
    """
    return {name: _COMPONENT_REGISTRY[name] for name in _SIGNAL_COMPONENTS}


def list_feature_components() -> Dict[str, Callable]:
    """List components commonly used as features.

    Returns a dictionary of all components registered with the "feature" type.

    Returns:
        Dict[str, Callable]: Dictionary mapping feature component names to their creation functions.
    """
    return {name: _COMPONENT_REGISTRY[name] for name in _FEATURE_COMPONENTS}


def get_component_parameters(component_name: str) -> Dict[str, Any]:
    """Get parameters for a component.

    Retrieves the parameter names and default values for a registered component.

    Args:
        component_name: Name of the component in the registry.

    Returns:
        Dict[str, Any]: Dictionary mapping parameter names to their default values.

    Raises:
        ValueError: If the component_name is not found in the registry.
    """
    if component_name not in _COMPONENT_REGISTRY:
        raise ValueError(f"Component {component_name} not registered")

    func = _COMPONENT_REGISTRY[component_name]

    # Retrieve the default parameters stored during registration
    if hasattr(func, "_param_defaults"):
        return func._param_defaults

    # Fallback to inspecting the function signature (for backward compatibility)
    params = inspect.signature(func).parameters
    return {
        name: param.default if param.default is not inspect.Parameter.empty else None
        for name, param in params.items()
        if name != "kwargs"
    }
    return {
        name: param.default if param.default is not inspect.Parameter.empty else None
        for name, param in params.items()
        if name != "kwargs"
    }
