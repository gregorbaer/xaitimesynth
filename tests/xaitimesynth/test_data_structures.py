import numpy as np
import pytest

from xaitimesynth.data_structures import TimeSeriesComponents


@pytest.fixture
def standard_foundation():
    """Fixture providing a standard 1D foundation array."""
    return np.array([1.0, 2.0, 3.0])


@pytest.fixture
def multidim_foundation():
    """Fixture providing a 2D foundation array."""
    return np.array([[1.0, 2.0], [3.0, 4.0]])


def test_time_series_components_initialization(standard_foundation):
    """Test TimeSeriesComponents initialization with different component combinations.

    Verifies both minimal (foundation-only) and complete initialization with
    all components properly stores and validates the provided values.

    Args:
        standard_foundation: A 1D numpy array fixture for testing.
    """
    # Basic initialization with just foundation
    ts_components = TimeSeriesComponents(foundation=standard_foundation)
    assert np.array_equal(ts_components.foundation, standard_foundation), (
        "Foundation not properly stored. "
        f"Expected {standard_foundation}, got {ts_components.foundation}"
    )
    assert ts_components.noise is None, "Noise should be None by default"
    assert ts_components.features is None, "Features should be None by default"
    assert ts_components.feature_masks is None, (
        "Feature masks should be None by default"
    )
    assert ts_components.aggregated is None, "Aggregated should be None by default"

    # Complete initialization with all components
    noise = np.array([0.1, 0.2, 0.3])
    features = {"feature1": np.array([0.4, 0.5, 0.6])}
    feature_masks = {"feature1": np.array([True, False, True])}
    aggregated = standard_foundation + noise + features["feature1"]

    ts_components = TimeSeriesComponents(
        foundation=standard_foundation,
        noise=noise,
        features=features,
        feature_masks=feature_masks,
        aggregated=aggregated,
    )

    assert np.array_equal(ts_components.foundation, standard_foundation), (
        "Foundation not correctly stored with full initialization"
    )
    assert np.array_equal(ts_components.noise, noise), (
        f"Noise array not correctly stored. Expected {noise}, got {ts_components.noise}"
    )
    assert "feature1" in ts_components.features, (
        "Feature key 'feature1' missing from features dictionary"
    )
    assert np.array_equal(ts_components.features["feature1"], features["feature1"]), (
        "Feature array not correctly stored"
    )
    assert "feature1" in ts_components.feature_masks, (
        "Feature mask key 'feature1' missing from feature_masks dictionary"
    )
    assert np.array_equal(
        ts_components.feature_masks["feature1"], feature_masks["feature1"]
    ), "Feature mask array not correctly stored"
    assert np.array_equal(ts_components.aggregated, aggregated), (
        "Aggregated array not correctly stored"
    )


def test_edge_cases():
    """Test TimeSeriesComponents with edge case inputs.

    Verifies the class can handle empty arrays and other edge cases properly.
    """
    foundation = np.array([])
    ts_components = TimeSeriesComponents(foundation=foundation)
    assert np.array_equal(ts_components.foundation, foundation), (
        "Empty array foundation not properly stored"
    )
    assert ts_components.foundation.size == 0, (
        f"Empty foundation should have size 0, got {ts_components.foundation.size}"
    )
    assert ts_components.foundation.shape == (0,), (
        f"Empty foundation should have shape (0,), got {ts_components.foundation.shape}"
    )


def test_shape_validation(standard_foundation, multidim_foundation):
    """Test shape validation across different component types and dimensions.

    Verifies that TimeSeriesComponents correctly validates shapes across all
    component types and raises appropriate errors when shapes don't match.
    Works with both 1D and multi-dimensional arrays.

    Args:
        standard_foundation: A 1D numpy array fixture for testing.
        multidim_foundation: A 2D numpy array fixture for testing.
    """

    # Test cases with standard 1D arrays
    def test_with_foundation(foundation):
        # Test with correct shapes first
        correct_noise = np.ones_like(foundation)
        correct_features = {"feature1": np.ones_like(foundation)}
        correct_masks = {"mask1": np.ones_like(foundation, dtype=bool)}
        correct_aggregated = np.ones_like(foundation)

        # This should not raise any errors
        ts = TimeSeriesComponents(
            foundation=foundation,
            noise=correct_noise,
            features=correct_features,
            feature_masks=correct_masks,
            aggregated=correct_aggregated,
        )

        # Verify all components were stored correctly
        assert np.array_equal(ts.foundation, foundation), (
            "Foundation not correctly stored in shape validation test"
        )
        assert np.array_equal(ts.noise, correct_noise), (
            "Noise with correct shape not properly stored"
        )
        assert np.array_equal(ts.features["feature1"], correct_features["feature1"]), (
            "Feature with correct shape not properly stored"
        )
        assert np.array_equal(ts.feature_masks["mask1"], correct_masks["mask1"]), (
            "Feature mask with correct shape not properly stored"
        )
        assert np.array_equal(ts.aggregated, correct_aggregated), (
            "Aggregated array with correct shape not properly stored"
        )

        # Wrong shape cases - create arrays with wrong shape
        wrong_shape = np.ones(2) if foundation.size > 2 else np.ones(3)

        # Test each component type
        component_types = [
            (
                "noise",
                lambda: TimeSeriesComponents(foundation=foundation, noise=wrong_shape),
            ),
            (
                "feature",
                lambda: TimeSeriesComponents(
                    foundation=foundation, features={"wrong_feature": wrong_shape}
                ),
            ),
            (
                "feature mask",
                lambda: TimeSeriesComponents(
                    foundation=foundation, feature_masks={"wrong_mask": wrong_shape}
                ),
            ),
            (
                "aggregated",
                lambda: TimeSeriesComponents(
                    foundation=foundation, aggregated=wrong_shape
                ),
            ),
        ]

        # Test each case
        for name, factory in component_types:
            with pytest.raises(ValueError) as excinfo:
                factory()
            # Verify the error message mentions the component name
            error_msg = str(excinfo.value).lower()
            assert any(n in error_msg for n in name.lower().split()), (
                f"Error message should mention '{name}' component"
            )
            assert "shape" in error_msg, "Error message should mention shape mismatch"
            assert str(foundation.shape) in error_msg, (
                f"Error message should include expected shape {foundation.shape}"
            )

    # Run tests with both 1D and multi-dimensional arrays
    test_with_foundation(standard_foundation)
    test_with_foundation(multidim_foundation)
