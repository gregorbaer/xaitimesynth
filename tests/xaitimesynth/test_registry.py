# TODO: check tests again whether...
# - they could be consolidated, and
# - whether all tests do what they should

import sys

import numpy as np
import pytest

from xaitimesynth.registry import (
    _COMPONENT_REGISTRY,
    _FEATURE_COMPONENTS,
    _SIGNAL_COMPONENTS,
    get_component_parameters,
    list_components,
    list_feature_components,
    list_signal_components,
    register_component,
    register_component_generator,
)


class MockGeneratorModule:
    """Mock module to simulate the xaitimesynth.generators module."""

    GENERATOR_FUNCS = {}


@pytest.fixture
def mock_generators_module():
    """Fixture to provide a fresh MockGeneratorModule for each test.

    Handles setup and teardown of the module in sys.modules.

    Returns:
        MockGeneratorModule: A fresh instance for the test.
    """
    # Store original module if it exists
    original_generators_module = sys.modules.get("xaitimesynth.generators")

    # Install fresh mock module
    mock_module = MockGeneratorModule()
    mock_module.GENERATOR_FUNCS = {}
    sys.modules["xaitimesynth.generators"] = mock_module

    yield mock_module

    # Restore original module if it existed
    if original_generators_module:
        sys.modules["xaitimesynth.generators"] = original_generators_module
    else:
        del sys.modules["xaitimesynth.generators"]


def setup_function():
    """Reset global registries before each test."""
    # Clear all registries
    _COMPONENT_REGISTRY.clear()
    _SIGNAL_COMPONENTS.clear()
    _FEATURE_COMPONENTS.clear()


def test_register_component_decorator():
    """Test registering components using the decorator syntax.

    Tests that:
    1. Components can be registered with explicit component type
    2. Components can be registered with default type ("both")
    3. Components appear in the correct registry collections
    """

    # Test with explicit component type
    @register_component(component_type="signal")
    def test_signal_component(**kwargs):
        return {"type": "test_signal_component", **kwargs}

    # Test with default component type (should be "both")
    @register_component
    def test_both_component(**kwargs):
        return {"type": "test_both_component", **kwargs}

    # Test with feature component type
    @register_component(component_type="feature")
    def test_feature_component(**kwargs):
        return {"type": "test_feature_component", **kwargs}

    # Verify components are registered correctly
    assert "test_signal_component" in list_components(), (
        "Signal component not found in component registry"
    )
    assert "test_both_component" in list_components(), (
        "Both-type component not found in component registry"
    )
    assert "test_feature_component" in list_components(), (
        "Feature component not found in component registry"
    )

    # Verify component types
    assert "test_signal_component" in list_signal_components(), (
        "Signal component not found in signal registry"
    )
    assert "test_signal_component" not in list_feature_components(), (
        "Signal component incorrectly found in feature registry"
    )

    assert "test_both_component" in list_signal_components(), (
        "Both-type component not found in signal registry"
    )
    assert "test_both_component" in list_feature_components(), (
        "Both-type component not found in feature registry"
    )

    assert "test_feature_component" not in list_signal_components(), (
        "Feature component incorrectly found in signal registry"
    )
    assert "test_feature_component" in list_feature_components(), (
        "Feature component not found in feature registry"
    )


def test_register_component_functional():
    """Test registering components using the functional approach.

    Verifies that components can be registered after being defined
    and that they function as expected.
    """

    def test_component(**kwargs):
        return {"type": "test_component", **kwargs}

    # Register after defining
    register_component(test_component, "both")

    assert "test_component" in list_components(), (
        "Component not found in registry after functional registration"
    )
    assert "test_component" in list_signal_components(), (
        "Component not found in signal registry"
    )
    assert "test_component" in list_feature_components(), (
        "Component not found in feature registry"
    )

    # Test function works as expected
    result = test_component(param1=42)
    assert result == {"type": "test_component", "param1": 42}, (
        "Component function returned incorrect result"
    )


def test_register_component_generator(mock_generators_module):
    """Test the component generator registration functionality.

    Tests:
    1. Default component naming from generator function
    2. Explicit component naming
    3. Default parameter overrides
    4. Component type registration
    5. Generator function registration in generators module
    6. Component function behavior
    """

    try:
        # Test with default naming
        @register_component_generator(component_type="signal")
        def generate_test_signal(
            n_timesteps, rng, length=None, amplitude=1.0, frequency=0.1
        ):
            """Generate a test signal."""
            return np.zeros(n_timesteps)

        # Test with explicit naming
        @register_component_generator(name="custom_feature", component_type="feature")
        def generate_something_else(n_timesteps, rng, length=None, scale=0.5):
            """Generate a custom feature."""
            return np.zeros(length or n_timesteps)

        # Test with default overrides
        @register_component_generator(amplitude=2.0, frequency=0.2)
        def generate_with_overrides(
            n_timesteps, rng, length=None, amplitude=1.0, frequency=0.1
        ):
            """Generate with overridden defaults."""
            return np.zeros(n_timesteps)

        # The component functions are registered in the current module
        # Get references to the auto-generated component functions
        current_module = sys.modules[__name__]
        test_signal = getattr(current_module, "test_signal")
        custom_feature = getattr(current_module, "custom_feature")
        with_overrides = getattr(current_module, "with_overrides")

        # Verify component registration
        assert "test_signal" in list_components(), (
            "Default-named component not properly registered"
        )
        assert "custom_feature" in list_components(), (
            "Custom-named component not properly registered"
        )
        assert "with_overrides" in list_components(), (
            "Component with overrides not properly registered"
        )

        # Verify component types
        assert "test_signal" in list_signal_components(), (
            "Signal component not in signal registry"
        )
        assert "test_signal" not in list_feature_components(), (
            "Signal component incorrectly in feature registry"
        )

        assert "custom_feature" not in list_signal_components(), (
            "Feature component incorrectly in signal registry"
        )
        assert "custom_feature" in list_feature_components(), (
            "Feature component not in feature registry"
        )

        assert "with_overrides" in list_signal_components(), (
            "Default-type component not in signal registry"
        )
        assert "with_overrides" in list_feature_components(), (
            "Default-type component not in feature registry"
        )

        # Verify generator function registration
        assert "test_signal" in mock_generators_module.GENERATOR_FUNCS, (
            "Generator function not registered in generators module"
        )
        assert "custom_feature" in mock_generators_module.GENERATOR_FUNCS, (
            "Custom-named generator function not registered in generators module"
        )
        assert "with_overrides" in mock_generators_module.GENERATOR_FUNCS, (
            "Generator function with overrides not registered in generators module"
        )

        # Test parameters and default overrides
        params = get_component_parameters("with_overrides")
        assert params["amplitude"] == 2.0, (
            "Default parameter override for amplitude not effective"
        )
        assert params["frequency"] == 0.2, (
            "Default parameter override for frequency not effective"
        )

        # Test the component functions work as expected
        test_signal_result = test_signal(amplitude=3.0)
        assert test_signal_result == {
            "type": "test_signal",
            "amplitude": 3.0,
            "frequency": 0.1,
        }, "Signal component function returned incorrect result"

        custom_feature_result = custom_feature(scale=1.0)
        assert custom_feature_result == {"type": "custom_feature", "scale": 1.0}, (
            "Feature component function returned incorrect result"
        )
    finally:
        # Clean up any component functions that were added to the current module
        for name in ["test_signal", "custom_feature", "with_overrides"]:
            if hasattr(current_module, name):
                delattr(current_module, name)


def test_list_components():
    """Test listing all components.

    Verifies that:
    1. Components of different types are correctly registered
    2. List functions return the proper subset of components
    3. The length of returned lists is correct
    """

    @register_component(component_type="signal")
    def comp1(**kwargs):
        return {"type": "comp1", **kwargs}

    @register_component(component_type="feature")
    def comp2(**kwargs):
        return {"type": "comp2", **kwargs}

    @register_component(component_type="both")
    def comp3(**kwargs):
        return {"type": "comp3", **kwargs}

    all_components = list_components()
    assert "comp1" in all_components, (
        "Signal component missing from all components list"
    )
    assert "comp2" in all_components, (
        "Feature component missing from all components list"
    )
    assert "comp3" in all_components, (
        "Both-type component missing from all components list"
    )
    assert len(all_components) == 3, "Unexpected number of components in registry"

    signal_components = list_signal_components()
    assert "comp1" in signal_components, (
        "Signal component missing from signal components list"
    )
    assert "comp2" not in signal_components, (
        "Feature component incorrectly in signal components list"
    )
    assert "comp3" in signal_components, (
        "Both-type component missing from signal components list"
    )
    assert len(signal_components) == 2, "Unexpected number of signal components"

    feature_components = list_feature_components()
    assert "comp1" not in feature_components, (
        "Signal component incorrectly in feature components list"
    )
    assert "comp2" in feature_components, (
        "Feature component missing from feature components list"
    )
    assert "comp3" in feature_components, (
        "Both-type component missing from feature components list"
    )
    assert len(feature_components) == 2, "Unexpected number of feature components"


def test_get_component_parameters():
    """Test retrieving component parameters.

    Verifies that default parameter values are correctly retrieved
    from registered component functions.
    """

    @register_component
    def test_component(param1=10, param2="test", param3=None, **kwargs):
        return {
            "type": "test_component",
            "param1": param1,
            "param2": param2,
            "param3": param3,
            **kwargs,
        }

    params = get_component_parameters("test_component")
    assert params["param1"] == 10, "Default numeric parameter not correctly retrieved"
    assert params["param2"] == "test", (
        "Default string parameter not correctly retrieved"
    )
    assert params["param3"] is None, "Default None parameter not correctly retrieved"


def test_get_nonexistent_component_parameters():
    """Test error handling when retrieving parameters for non-existent component."""
    with pytest.raises(
        ValueError, match="Component nonexistent_component not registered"
    ):
        get_component_parameters("nonexistent_component")


def test_component_generator_docstring(mock_generators_module):
    """Test that generated component functions have appropriate docstrings.

    Verifies that component functions created from generators have
    automatically generated docstrings containing relevant information.
    """
    try:

        @register_component_generator()
        def generate_docstring_test(n_timesteps, rng, length=None, param1=1, param2=2):
            """Generate a test component with nice docstring."""
            return None  # Mock implementation

        # Get reference to the auto-generated component function
        current_module = sys.modules[__name__]
        docstring_test = getattr(current_module, "docstring_test")

        # Check that the component function has a docstring
        assert docstring_test.__doc__ is not None, (
            "Generated component has no docstring"
        )
        assert "Create a docstring_test component" in docstring_test.__doc__, (
            "Expected content missing from generated docstring"
        )
    finally:
        # Clean up by removing the generated component
        current_module = sys.modules[__name__]
        if hasattr(current_module, "docstring_test"):
            delattr(current_module, "docstring_test")


def test_component_generator_module_export(mock_generators_module):
    """Test that component functions are exported to the generator module.

    Verifies that:
    1. Component functions are created in the current module
    2. These functions are callable
    3. The functions return the expected component definitions
    """
    # The current module is where the generator function will be defined
    current_module = sys.modules[__name__]

    try:
        # Define a generator in the current module
        @register_component_generator()
        def generate_exported_component(n_timesteps, rng, length=None, param=1):
            """Generate a component that should be exported."""
            return None

        # Check that the component function is available in the current module
        assert hasattr(current_module, "exported_component"), (
            "Component function not created in module"
        )
        assert callable(current_module.exported_component), (
            "Created component is not callable"
        )

        # Check that it works correctly
        component_def = current_module.exported_component(param=5)
        assert component_def == {"type": "exported_component", "param": 5}, (
            "Component function returned incorrect result"
        )
    finally:
        # Clean up by removing the exported component
        if hasattr(current_module, "exported_component"):
            delattr(current_module, "exported_component")
