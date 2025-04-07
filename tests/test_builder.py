import numpy as np
import pandas as pd
import pytest

from xaitimesynth.builder import TimeSeriesBuilder


@pytest.fixture
def basic_builder() -> TimeSeriesBuilder:
    """Fixture providing a basic TimeSeriesBuilder instance."""
    return TimeSeriesBuilder(n_timesteps=50, n_samples=20, random_state=42)


@pytest.fixture
def two_class_builder(basic_builder) -> TimeSeriesBuilder:
    """Fixture providing a builder with two classes defined."""
    return (
        basic_builder.for_class(0)
        .add_signal({"type": "random_walk"})
        .for_class(1)
        .add_signal({"type": "random_walk"})
    )


def test_builder_init() -> None:
    """Test initialization defaults."""
    builder = TimeSeriesBuilder()
    assert builder.n_timesteps == 100, "Expected default timesteps of 100"
    assert builder.n_samples == 1000, "Expected default samples of 1000"
    assert builder.n_dimensions == 1, "Expected default dimensions of 1"


def test_invalid_dimensions() -> None:
    """Test that invalid dimensions raise ValueError."""
    with pytest.raises(ValueError, match="n_dimensions must be at least 1"):
        TimeSeriesBuilder(n_dimensions=0)


def test_for_class() -> None:
    """Test class creation and label assignment."""
    builder = TimeSeriesBuilder()
    builder.for_class(class_label=2, weight=1.5)
    assert len(builder.class_definitions) == 1, (
        "Should have exactly one class definition"
    )
    assert builder.class_definitions[0]["label"] == 2, (
        "Class label should match the assigned value"
    )
    assert builder.class_definitions[0]["weight"] == 1.5, (
        "Class weight should match the assigned value"
    )


def test_add_signal_no_class() -> None:
    """Test add_signal without selecting a class first."""
    builder = TimeSeriesBuilder()
    with pytest.raises(ValueError, match="No class selected"):
        builder.add_signal({"type": "random_walk"}, role="foundation")


def test_add_signal_invalid_role() -> None:
    """Test add_signal with invalid role parameter."""
    builder = TimeSeriesBuilder().for_class(1)
    with pytest.raises(ValueError, match="Invalid role"):
        builder.add_signal({"type": "random_walk"}, role="invalid_role")


def test_add_signal_invalid_dimensions() -> None:
    """Test add_signal with invalid dimension indices."""
    builder = TimeSeriesBuilder(n_dimensions=2).for_class(1)
    with pytest.raises(ValueError, match="Dimension 2 is out of range"):
        builder.add_signal({"type": "random_walk"}, dim=[2])


def test_add_signal_shared_randomness() -> None:
    """Test add_signal with shared_randomness parameter."""
    builder = TimeSeriesBuilder(n_dimensions=2).for_class(1)
    # With shared_randomness=True, should create a single component with multiple dimensions
    builder.add_signal({"type": "random_walk"}, dim=[0, 1], shared_randomness=True)
    assert len(builder.current_class["components"]["foundation"]) == 1, (
        "With shared_randomness=True, should add a single component entry"
    )

    # With shared_randomness=False, should create separate components for each dimension
    builder = TimeSeriesBuilder(n_dimensions=2).for_class(1)
    builder.add_signal({"type": "random_walk"}, dim=[0, 1], shared_randomness=False)
    assert len(builder.current_class["components"]["foundation"]) == 2, (
        "With shared_randomness=False, should add separate component entries"
    )


def test_add_signal_segment_parameter_validation() -> None:
    """Test parameter validation in add_signal_segment.

    Verifies that add_signal_segment properly validates:
    - Missing length_pct when random_location=True
    - Invalid start_pct/end_pct range
    - Invalid length_pct range
    - Missing start_pct/end_pct when random_location=False
    """
    builder = TimeSeriesBuilder().for_class(1)

    # Test random location without length_pct
    with pytest.raises(ValueError, match="length_pct must be provided"):
        builder.add_signal_segment({"type": "constant"}, random_location=True)

    # Test invalid range for fixed location
    with pytest.raises(ValueError, match="Invalid start_pct or end_pct"):
        builder.add_signal_segment({"type": "constant"}, start_pct=1.1, end_pct=1.2)

    # Test invalid length_pct
    with pytest.raises(ValueError, match="length_pct must be between 0 and 1"):
        builder.add_signal_segment(
            {"type": "constant"}, random_location=True, length_pct=1.5
        )

    # Test fixed location without start_pct or end_pct
    with pytest.raises(ValueError, match="start_pct and end_pct must be provided"):
        builder.add_signal_segment(
            {"type": "constant"},
            random_location=False,  # Just need to specify random_location=False
        )


def test_add_signal_segment_shared_options() -> None:
    """Test shared_location and shared_randomness options in add_signal_segment.

    Verifies that add_signal_segment properly handles:
    - shared_location=True/False settings
    - shared_randomness=True/False settings
    - Dimension configurations
    """
    # Create a builder with 2 dimensions explicitly
    builder = TimeSeriesBuilder(n_dimensions=2).for_class(1)

    # Test shared location (should create one component)
    builder.add_signal_segment(
        {"type": "constant"},
        random_location=True,
        length_pct=0.2,
        dim=[0, 1],
        shared_location=True,
    )
    assert len(builder.current_class["components"]["foundation"]) == 1, (
        "With shared_location=True, should add a single component entry"
    )

    # Test non-shared location (should create separate components)
    builder = TimeSeriesBuilder(n_dimensions=2).for_class(1)
    builder.add_signal_segment(
        {"type": "constant"},
        random_location=True,
        length_pct=0.2,
        dim=[0, 1],
        shared_location=False,
    )
    assert len(builder.current_class["components"]["foundation"]) == 2, (
        "With shared_location=False, should add separate component entries"
    )


def test_add_feature_parameter_validation() -> None:
    """Test add_feature with invalid parameters.

    Validates that add_feature correctly checks for:
    - Missing start_pct/end_pct when random_location=False
    - Invalid start_pct/end_pct ranges
    - Missing length_pct when random_location=True
    """
    builder = TimeSeriesBuilder().for_class(1)

    # Test missing start_pct and end_pct
    with pytest.raises(ValueError, match="start_pct and end_pct must be provided"):
        builder.add_feature({"type": "shapelet"}, random_location=False)

    # Test invalid start_pct and end_pct range
    with pytest.raises(ValueError, match="Invalid start_pct or end_pct"):
        builder.add_feature({"type": "shapelet"}, start_pct=1.1, end_pct=1.5)

    # Test random_location without length_pct
    with pytest.raises(ValueError, match="length_pct must be provided"):
        builder.add_feature({"type": "shapelet"}, random_location=True)


def test_add_feature_random_location() -> None:
    """Test add_feature with random location parameters.

    Ensures length_pct is validated for random feature placement.
    """
    builder = TimeSeriesBuilder().for_class(1)
    with pytest.raises(ValueError, match="length_pct must be between 0 and 1"):
        builder.add_feature({"type": "shapelet"}, random_location=True, length_pct=1.5)


def test_build_no_classes() -> None:
    """Test build with no classes defined."""
    builder = TimeSeriesBuilder()
    with pytest.raises(ValueError, match="No class definitions provided"):
        builder.build()


def test_random_state_reproducibility() -> None:
    """Test that the same random_state produces identical results.

    Verifies that builders initialized with the same random state and
    configured identically produce exactly the same datasets.
    """
    # Create two builders with the same random_state
    builder1 = TimeSeriesBuilder(random_state=42)
    dataset1 = (
        builder1.for_class(0)
        .add_signal({"type": "random_walk"})
        .for_class(1)
        .add_signal({"type": "random_walk"})
        .add_feature({"type": "shapelet"}, random_location=True, length_pct=0.2)
        .build()
    )

    builder2 = TimeSeriesBuilder(random_state=42)
    dataset2 = (
        builder2.for_class(0)
        .add_signal({"type": "random_walk"})
        .for_class(1)
        .add_signal({"type": "random_walk"})
        .add_feature({"type": "shapelet"}, random_location=True, length_pct=0.2)
        .build()
    )

    # The datasets should be identical
    np.testing.assert_array_equal(
        dataset1["X"],
        dataset2["X"],
        err_msg="Datasets with same random_state should be identical",
    )
    np.testing.assert_array_equal(
        dataset1["y"],
        dataset2["y"],
        err_msg="Class labels should be identical with same random_state",
    )


def test_custom_fill_values() -> None:
    """Test that custom fill values work correctly.

    Verifies that custom feature_fill_value is used for non-existent feature regions
    """
    # Create builder with custom fill values
    builder = TimeSeriesBuilder(
        n_timesteps=20,
        n_samples=2,
        feature_fill_value=-999.0,
        foundation_fill_value=-1.0,
        noise_fill_value=-2.0,
        random_state=42,
    )

    # Use a constant component that will generate properly sized output
    dataset = (
        builder.for_class(0)
        .add_signal({"type": "random_walk"})
        .add_feature({"type": "peak", "value": 5.0}, start_pct=0.5, end_pct=0.6)
        .build()
    )

    # Check feature fill value in components
    sample_components = dataset["components"][0]
    feature_name = list(sample_components.features.keys())[0]
    feature_values = sample_components.features[feature_name]

    # The first element should be the feature_fill_value since the feature starts at 0.5
    assert np.isclose(feature_values[0], -999.0), (
        f"Expected feature_fill_value of -999.0, got {feature_values[0]}"
    )


def test_add_vector_handling_nans() -> None:
    """Test the _add_vector_handling_nans method."""
    builder = TimeSeriesBuilder()

    # Case 1: Both vectors have values
    base = np.array([1.0, 2.0, 3.0])
    to_add = np.array([0.1, 0.2, 0.3])
    result = builder._add_vector_handling_nans(base, to_add)
    expected = np.array([1.1, 2.2, 3.3])
    np.testing.assert_almost_equal(
        result, expected, err_msg="Failed to add vectors with actual values"
    )

    # Case 2: Base has NaN, to_add has value
    base = np.array([1.0, np.nan, 3.0])
    to_add = np.array([0.1, 0.2, 0.3])
    result = builder._add_vector_handling_nans(base, to_add)
    expected = np.array([1.1, 0.2, 3.3])
    np.testing.assert_almost_equal(
        result, expected, err_msg="Failed to handle NaN in base vector"
    )

    # Case 3: Base has value, to_add has NaN
    base = np.array([1.0, 2.0, 3.0])
    to_add = np.array([0.1, np.nan, 0.3])
    result = builder._add_vector_handling_nans(base, to_add)
    expected = np.array([1.1, 2.0, 3.3])
    np.testing.assert_almost_equal(
        result, expected, err_msg="Failed to handle NaN in to_add vector"
    )

    # Case 4: Both have NaN at same position
    base = np.array([1.0, np.nan, 3.0])
    to_add = np.array([0.1, np.nan, 0.3])
    result = builder._add_vector_handling_nans(base, to_add)
    assert np.isnan(result[1]), "Result should be NaN when both inputs are NaN"
    np.testing.assert_almost_equal(
        result[0], 1.1, err_msg="Failed with mix of NaN and non-NaN values"
    )
    np.testing.assert_almost_equal(
        result[2], 3.3, err_msg="Failed with mix of NaN and non-NaN values"
    )


def test_to_df_basic(two_class_builder) -> None:
    """Test the basic functionality of to_df method.

    More detailed tests for filtering and formatting are in test_builder_functional.py.
    """
    # Create a minimal dataset
    builder = two_class_builder
    builder.add_feature({"type": "shapelet"}, start_pct=0.4, end_pct=0.6)
    dataset = builder.build()

    # Convert to dataframe
    df = builder.to_df(dataset)

    # Basic validations
    assert isinstance(df, pd.DataFrame), "to_df should return a pandas DataFrame"
    assert not df.empty, "DataFrame should not be empty"

    # Check columns
    expected_columns = [
        "time",
        "value",
        "class",
        "sample",
        "component",
        "feature",
        "dim",
    ]
    assert all(col in df.columns for col in expected_columns), (
        f"DataFrame should contain columns {expected_columns}"
    )

    # Check component types
    components = df["component"].unique()
    assert "aggregated" in components, "DataFrame should contain aggregated component"
    assert "foundation" in components, "DataFrame should contain foundation component"
    assert "features" in components, "DataFrame should contain features component"
