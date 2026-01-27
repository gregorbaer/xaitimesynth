from unittest.mock import MagicMock

import numpy as np
import pytest

from xaitimesynth.functions import (
    FeatureAdder,
    SignalAdder,
    add_feature,
    add_signal,
    minmax_normalize,
    normalize,
    zscore_normalize,
)


class TestAdders:
    """Tests for SignalAdder and FeatureAdder classes.

    These tests verify the initialization and calling behavior of the component adder classes,
    which are responsible for adding signal and feature components to time series data.
    """

    def test_signal_adder_initialization(self):
        """Test SignalAdder initializes with correct default and custom values."""
        component = {"type": "sine", "frequency": 0.1}

        # Test default values
        adder = SignalAdder(component)
        assert adder.component == component, "Component not correctly stored"
        assert adder.role == "foundation", "Default role should be 'foundation'"
        assert adder.start_pct is None, "Default start_pct should be None"
        assert adder.shared_location is True, "Default shared_location should be True"

        # Test custom values
        adder = SignalAdder(
            component,
            role="noise",
            start_pct=0.2,
            end_pct=0.8,
            random_location=True,
            shared_location=False,
        )
        assert adder.role == "noise", "Custom role not correctly set"
        assert adder.start_pct == 0.2, "Custom start_pct not correctly set"
        assert adder.end_pct == 0.8, "Custom end_pct not correctly set"
        assert adder.random_location is True, "Custom random_location not correctly set"
        assert adder.shared_location is False, (
            "Custom shared_location not correctly set"
        )

    def test_signal_adder_with_none_values(self):
        """Test SignalAdder properly handles None values for optional parameters."""
        component = {"type": "sine"}
        adder = SignalAdder(component, length_pct=None, start_pct=None, end_pct=None)

        mock_builder = MagicMock()
        mock_builder.add_signal.return_value = mock_builder
        adder(mock_builder)

        # Verify None values are passed through correctly
        mock_builder.add_signal.assert_called_once()
        call_kwargs = mock_builder.add_signal.call_args[1]
        assert call_kwargs["start_pct"] is None, "None start_pct should be preserved"
        assert call_kwargs["end_pct"] is None, "None end_pct should be preserved"
        assert call_kwargs["length_pct"] is None, "None length_pct should be preserved"

    def test_feature_adder_initialization(self):
        """Test FeatureAdder initializes with correct default and custom values."""
        component = {"type": "spike", "amplitude": 2.0}

        # Test default values
        adder = FeatureAdder(component)
        assert adder.component == component, "Component not correctly stored"
        assert adder.start_pct is None, "Default start_pct should be None"
        assert adder.random_location is False, "Default random_location should be False"

        # Test custom values
        adder = FeatureAdder(
            component, start_pct=0.3, end_pct=0.7, length_pct=0.4, random_location=True
        )
        assert adder.start_pct == 0.3, "Custom start_pct not correctly set"
        assert adder.end_pct == 0.7, "Custom end_pct not correctly set"
        assert adder.length_pct == 0.4, "Custom length_pct not correctly set"
        assert adder.random_location is True, "Custom random_location not correctly set"

    def test_adder_call_methods(self):
        """Test the __call__ methods of both adder classes."""
        component = {"type": "test_component"}
        mock_builder = MagicMock()
        # Make these methods return the mock builder for chaining
        mock_builder.add_signal.return_value = mock_builder
        mock_builder.add_feature_component.return_value = mock_builder

        # Test SignalAdder call
        signal_adder = SignalAdder(component, start_pct=0.1, end_pct=0.9)
        result = signal_adder(mock_builder)

        mock_builder.add_signal.assert_called_once_with(
            component,
            role="foundation",
            start_pct=0.1,
            end_pct=0.9,
            length_pct=None,
            random_location=False,
            shared_location=True,
        )
        assert result == mock_builder, "SignalAdder call should return the builder"

        # Reset and test FeatureAdder call
        mock_builder.reset_mock()
        feature_adder = FeatureAdder(component, length_pct=0.25)
        result = feature_adder(mock_builder)

        mock_builder.add_feature_component.assert_called_once_with(
            component,
            start_pct=None,
            end_pct=None,
            length_pct=0.25,
            random_location=False,
        )
        assert result == mock_builder, "FeatureAdder call should return the builder"


class TestAdderFactoryFunctions:
    """Tests for add_signal and add_feature factory functions."""

    def test_add_signal(self):
        """Test add_signal creates and configures a SignalAdder correctly."""
        component = {"type": "sine"}

        # Test with default parameters
        adder = add_signal(component)
        assert isinstance(adder, SignalAdder), "Should return a SignalAdder instance"
        assert adder.component == component, "Component not correctly passed"
        assert adder.role == "foundation", "Default role should be 'foundation'"

        # Test with custom parameters
        adder = add_signal(
            component,
            role="noise",
            start_pct=0.2,
            end_pct=0.7,
            length_pct=0.5,
            random_location=True,
            shared_location=False,
        )
        assert adder.role == "noise", "Custom role not correctly passed"
        assert adder.start_pct == 0.2, "Custom start_pct not correctly passed"
        assert adder.end_pct == 0.7, "Custom end_pct not correctly passed"
        assert adder.length_pct == 0.5, "Custom length_pct not correctly passed"
        assert adder.random_location is True, (
            "Custom random_location not correctly passed"
        )
        assert adder.shared_location is False, (
            "Custom shared_location not correctly passed"
        )

    def test_add_feature(self):
        """Test add_feature creates and configures a FeatureAdder correctly."""
        component = {"type": "spike"}

        # Test with default parameters
        adder = add_feature(component)
        assert isinstance(adder, FeatureAdder), "Should return a FeatureAdder instance"
        assert adder.component == component, "Component not correctly passed"

        # Test with custom parameters
        adder = add_feature(
            component, start_pct=0.3, end_pct=0.6, length_pct=0.3, random_location=True
        )
        assert adder.start_pct == 0.3, "Custom start_pct not correctly passed"
        assert adder.end_pct == 0.6, "Custom end_pct not correctly passed"
        assert adder.length_pct == 0.3, "Custom length_pct not correctly passed"
        assert adder.random_location is True, (
            "Custom random_location not correctly passed"
        )


class TestNormalizationFunctions:
    """Tests for the various normalization functions.

    These tests verify the functionality of data normalization methods including min-max scaling,
    z-score normalization, and the high-level normalize function that provides a unified interface.
    """

    def test_minmax_normalize(self):
        """Test minmax_normalize with various inputs and ranges."""
        # Normal case
        data = np.array([1, 2, 3, 4, 5])
        normalized = minmax_normalize(data)
        expected = np.array([0, 0.25, 0.5, 0.75, 1.0])
        np.testing.assert_array_almost_equal(
            normalized,
            expected,
            err_msg="MinMax normalization with default range failed",
        )

        # Custom range
        normalized = minmax_normalize(data, feature_range=(-1, 1))
        expected = np.array([-1, -0.5, 0, 0.5, 1.0])
        np.testing.assert_array_almost_equal(
            normalized,
            expected,
            err_msg="MinMax normalization with custom range failed",
        )

        # Constant array
        constant_data = np.array([3, 3, 3])
        normalized = minmax_normalize(constant_data)
        np.testing.assert_array_equal(
            normalized,
            constant_data,
            err_msg="MinMax normalization of constant array should return the array unchanged",
        )

    def test_zscore_normalize(self):
        """Test zscore_normalize with various inputs."""
        # Normal case
        data = np.array([1, 2, 3, 4, 5])
        normalized = zscore_normalize(data)
        mean = np.mean(data)
        std = np.std(data)
        expected = (data - mean) / std
        np.testing.assert_array_almost_equal(
            normalized, expected, err_msg="Z-score normalization failed"
        )

        # Check properties
        assert abs(np.mean(normalized)) < 1e-10, (
            "Z-score normalized data should have mean close to 0"
        )
        assert abs(np.std(normalized) - 1.0) < 1e-10, (
            "Z-score normalized data should have std close to 1"
        )

        # Constant array
        constant_data = np.array([3, 3, 3])
        normalized = zscore_normalize(constant_data)
        np.testing.assert_array_equal(
            normalized,
            constant_data,
            err_msg="Z-score normalization of constant array should return the array unchanged",
        )

        # Custom epsilon
        data_with_small_std = np.array([1.0, 1.001, 0.999])
        normalized = zscore_normalize(data_with_small_std, epsilon=0.1)
        np.testing.assert_array_equal(
            normalized,
            data_with_small_std,
            err_msg="Z-score normalization with std < epsilon should return original data",
        )

    def test_normalize(self):
        """Test the normalize function with different methods."""
        data = np.array([1, 2, 3, 4, 5])

        # Test zscore method
        normalized = normalize(data, method="zscore")
        expected = zscore_normalize(data)
        np.testing.assert_array_almost_equal(
            normalized,
            expected,
            err_msg="normalize with 'zscore' method should match zscore_normalize result",
        )

        # Test minmax method
        normalized = normalize(data, method="minmax")
        expected = minmax_normalize(data)
        np.testing.assert_array_almost_equal(
            normalized,
            expected,
            err_msg="normalize with 'minmax' method should match minmax_normalize result",
        )

        # Test none method
        normalized = normalize(data, method="none")
        np.testing.assert_array_equal(
            normalized,
            data,
            err_msg="normalize with 'none' method should return original data",
        )

        # Test with kwargs
        normalized = normalize(data, method="minmax", feature_range=(-1, 1))
        expected = minmax_normalize(data, feature_range=(-1, 1))
        np.testing.assert_array_almost_equal(
            normalized,
            expected,
            err_msg="normalize should pass kwargs to the appropriate normalization function",
        )

        # Test invalid method
        with pytest.raises(ValueError, match="Invalid normalization method"):
            normalize(data, method="invalid_method")

    def test_normalize_with_invalid_inputs(self):
        """Test that normalize properly handles invalid inputs."""
        data = np.array([1, 2, 3])

        # Test with invalid method name
        with pytest.raises(ValueError, match="Invalid normalization method"):
            normalize(data, method="invalid_method")

        # We only test what the function explicitly validates in its code
        # The underlying numpy functions will handle other invalid inputs
