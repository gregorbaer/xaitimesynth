import numpy as np
import pytest

from xaitimesynth.data_structures import TimeSeriesComponents


@pytest.fixture
def standard_background():
    """Fixture providing a standard 1D background array."""
    return np.array([1.0, 2.0, 3.0])


@pytest.fixture
def multidim_background():
    """Fixture providing a 2D background array."""
    return np.array([[1.0, 2.0], [3.0, 4.0]])


def test_time_series_components_initialization(standard_background):
    """Test TimeSeriesComponents initialization with different component combinations.

    Verifies both minimal (background-only) and complete initialization with
    all components properly stores and validates the provided values.

    Args:
        standard_background: A 1D numpy array fixture for testing.
    """
    # Basic initialization with just background
    ts_components = TimeSeriesComponents(background=standard_background)
    assert np.array_equal(ts_components.background, standard_background), (
        "Background not properly stored. "
        f"Expected {standard_background}, got {ts_components.background}"
    )
    assert ts_components.features is None, "Features should be None by default"
    assert ts_components.feature_masks is None, (
        "Feature masks should be None by default"
    )
    assert ts_components.aggregated is None, "Aggregated should be None by default"

    # Complete initialization with all components
    features = {"feature1": np.array([0.4, 0.5, 0.6])}
    feature_masks = {"feature1": np.array([True, False, True])}
    aggregated = standard_background + features["feature1"]

    ts_components = TimeSeriesComponents(
        background=standard_background,
        features=features,
        feature_masks=feature_masks,
        aggregated=aggregated,
    )

    assert np.array_equal(ts_components.background, standard_background), (
        "Background not correctly stored with full initialization"
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
    background = np.array([])
    ts_components = TimeSeriesComponents(background=background)
    assert np.array_equal(ts_components.background, background), (
        "Empty array background not properly stored"
    )
    assert ts_components.background.size == 0, (
        f"Empty background should have size 0, got {ts_components.background.size}"
    )
    assert ts_components.background.shape == (0,), (
        f"Empty background should have shape (0,), got {ts_components.background.shape}"
    )


def test_shape_validation(standard_background, multidim_background):
    """Test shape validation across different component types and dimensions.

    Verifies that TimeSeriesComponents correctly validates component time dimensions
    and raises appropriate errors when shapes don't match the requirements.
    Works with both 1D and multi-dimensional arrays.

    Args:
        standard_background: A 1D numpy array fixture for testing.
        multidim_background: A 2D numpy array fixture for testing.
    """

    # Test cases with standard 1D arrays
    def test_with_background(background):
        # Test with correct shapes first
        # For features, only the time dimension (first dimension) needs to match
        # So if background is 2D with shape (3, 2), features can be 1D with length 3
        time_length = background.shape[0]
        features_time_match = {"feature1": np.ones(time_length)}

        correct_masks = {"mask1": np.ones(time_length, dtype=bool)}
        correct_aggregated = np.ones_like(background)

        # This should not raise any errors
        ts = TimeSeriesComponents(
            background=background,
            features=features_time_match,  # Only time dimension needs to match
            feature_masks=correct_masks,  # Only time dimension needs to match
            aggregated=correct_aggregated,  # Exact shape match required
        )

        # Verify all components were stored correctly
        assert np.array_equal(ts.background, background), (
            "Background not correctly stored in shape validation test"
        )
        assert np.array_equal(
            ts.features["feature1"], features_time_match["feature1"]
        ), "Feature with correct time dimension not properly stored"
        assert np.array_equal(ts.feature_masks["mask1"], correct_masks["mask1"]), (
            "Feature mask with correct time dimension not properly stored"
        )
        assert np.array_equal(ts.aggregated, correct_aggregated), (
            "Aggregated array with correct shape not properly stored"
        )

        # Wrong time dimension cases - create arrays with wrong first dimension
        wrong_time_length = time_length + 1
        wrong_time_dim = np.ones(wrong_time_length)

        # Wrong shape for aggregated (must match exactly)
        if len(background.shape) == 1:
            wrong_agg_shape = np.ones((time_length, 2))  # Add a dimension
        else:
            wrong_agg_shape = np.ones(
                (time_length, background.shape[1] + 1)
            )  # Add to second dimension

        # Test each component type with wrong dimensions
        component_types = [
            (
                "feature",
                lambda: TimeSeriesComponents(
                    background=background, features={"wrong_feature": wrong_time_dim}
                ),
                "first dimension",
            ),
            (
                "feature mask",
                lambda: TimeSeriesComponents(
                    background=background, feature_masks={"wrong_mask": wrong_time_dim}
                ),
                "first dimension",
            ),
            (
                "aggregated",
                lambda: TimeSeriesComponents(
                    background=background, aggregated=wrong_agg_shape
                ),
                "shape",  # For aggregated, the full shape must match
            ),
        ]

        # Test each case
        for name, factory, error_indicator in component_types:
            with pytest.raises(ValueError) as excinfo:
                factory()
            # Verify the error message mentions the component name and appropriate error indicator
            error_msg = str(excinfo.value).lower()
            assert any(n in error_msg for n in name.lower().split()), (
                f"Error message should mention '{name}' component"
            )
            assert error_indicator in error_msg, (
                f"Error message should mention {error_indicator}"
            )

            # Check if the appropriate dimension info is in the error message
            if error_indicator == "first dimension":
                assert str(time_length) in error_msg, (
                    f"Error message should include expected time dimension length {time_length}"
                )
            else:
                assert str(background.shape) in error_msg, (
                    f"Error message should include expected shape {background.shape}"
                )

    # Run tests with both 1D and multi-dimensional arrays
    test_with_background(standard_background)
    test_with_background(multidim_background)
