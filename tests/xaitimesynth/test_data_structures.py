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
    """Test basic and complete initialization of TimeSeriesComponents."""
    # Basic initialization with just foundation
    ts_components = TimeSeriesComponents(foundation=standard_foundation)
    assert np.array_equal(ts_components.foundation, standard_foundation)
    assert ts_components.noise is None
    assert ts_components.features is None
    assert ts_components.feature_masks is None
    assert ts_components.aggregated is None

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

    assert np.array_equal(ts_components.foundation, standard_foundation)
    assert np.array_equal(ts_components.noise, noise)
    assert ts_components.features == features
    assert ts_components.feature_masks == feature_masks
    assert np.array_equal(ts_components.aggregated, aggregated)


def test_edge_cases():
    """Test initialization with edge cases like empty arrays."""
    foundation = np.array([])
    ts_components = TimeSeriesComponents(foundation=foundation)
    assert np.array_equal(ts_components.foundation, foundation)


def test_shape_validation(standard_foundation, multidim_foundation):
    """Test shape validation for all component types with both 1D and multi-dimensional arrays."""

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
            assert any(n in error_msg for n in name.lower().split())

    # Run tests with both 1D and multi-dimensional arrays
    test_with_foundation(standard_foundation)
    test_with_foundation(multidim_foundation)
