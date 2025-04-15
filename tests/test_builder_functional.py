"""Functional tests for the TimeSeriesBuilder class.

These tests verify that the TimeSeriesBuilder correctly builds time series datasets
with various configurations including both univariate and multivariate series,
different numbers of classes, and different feature configurations.
"""

import numpy as np
import pandas as pd
import pytest

from xaitimesynth import (
    TimeSeriesBuilder,
    constant,
    gaussian,
    level_change,
    peak,
    random_walk,
    shapelet,
)


@pytest.fixture
def basic_univariate_config():
    """Fixture providing basic configuration for univariate tests."""
    return {"n_samples": 100, "n_timesteps": 50, "random_state": 42}


@pytest.fixture
def basic_multivariate_config():
    """Fixture providing basic configuration for multivariate tests."""
    return {"n_samples": 80, "n_timesteps": 150, "n_dimensions": 2, "random_state": 42}


def test_univariate_two_classes(basic_univariate_config):
    """Test building a univariate dataset with two classes.

    Tests that:
    1. The builder correctly creates a dataset dictionary with expected keys
    2. The shapes of X and y are correct for the requested configuration
    3. The class distribution includes both requested classes
    4. The metadata contains correct information about the dataset
    """
    # Create a simple dataset with two classes
    n_samples = basic_univariate_config["n_samples"]
    n_timesteps = basic_univariate_config["n_timesteps"]

    dataset = (
        TimeSeriesBuilder(**basic_univariate_config)
        .for_class(0)  # Class 0: No discriminative features
        .add_signal(random_walk(step_size=0.2))
        .add_signal(gaussian(sigma=0.1), role="noise")
        .for_class(1)  # Class 1: Has shapelet feature
        .add_signal(random_walk(step_size=0.2))
        .add_signal(gaussian(sigma=0.1), role="noise")
        .add_feature(shapelet(scale=1.0), start_pct=0.4, end_pct=0.6)
        .build()
    )

    # Verify the structure of the output
    assert isinstance(dataset, dict), "Dataset should be returned as a dictionary"
    assert "X" in dataset, "Dataset should contain 'X' key with time series data"
    assert "y" in dataset, "Dataset should contain 'y' key with class labels"
    assert "feature_masks" in dataset, "Dataset should contain 'feature_masks' key"
    assert "metadata" in dataset, "Dataset should contain 'metadata' key"
    assert "components" in dataset, "Dataset should contain 'components' key"

    # Check the shapes
    assert dataset["X"].shape == (n_samples, 1, n_timesteps), (
        f"Expected X shape (n_samples={n_samples}, n_dimensions=1, n_timesteps={n_timesteps}), "
        f"got {dataset['X'].shape}"
    )
    assert dataset["y"].shape == (n_samples,), (
        f"Expected y shape (n_samples={n_samples},), got {dataset['y'].shape}"
    )

    # Verify class distribution
    classes, counts = np.unique(dataset["y"], return_counts=True)
    assert len(classes) == 2, f"Expected 2 classes, got {len(classes)}"
    assert 0 in classes, "Class 0 should be present in the dataset"
    assert 1 in classes, "Class 1 should be present in the dataset"

    # Check metadata
    assert dataset["metadata"]["n_samples"] == n_samples, (
        f"Metadata should contain correct n_samples={n_samples}"
    )
    assert dataset["metadata"]["n_timesteps"] == n_timesteps, (
        f"Metadata should contain correct n_timesteps={n_timesteps}"
    )
    assert dataset["metadata"]["n_dimensions"] == 1, (
        "Metadata should indicate univariate data with n_dimensions=1"
    )


def test_univariate_three_classes(basic_univariate_config):
    """Test building a univariate dataset with three classes.

    Tests that:
    1. The builder correctly creates three distinct classes
    2. The shapes of X and y match the requested configuration
    3. All three classes are represented in the dataset
    """
    # Use higher n_samples and timesteps for this test
    n_samples = 120
    n_timesteps = 100

    dataset = (
        TimeSeriesBuilder(
            n_samples=n_samples,
            n_timesteps=n_timesteps,
            random_state=basic_univariate_config["random_state"],
        )
        # Class 0: Base signal only
        .for_class(0)
        .add_signal(random_walk(step_size=0.2))
        .add_signal(gaussian(sigma=0.1), role="noise")
        # Class 1: Base signal + shapelet
        .for_class(1)
        .add_signal(random_walk(step_size=0.2))
        .add_signal(gaussian(sigma=0.1), role="noise")
        .add_feature(shapelet(scale=1.0), start_pct=0.4, end_pct=0.6)
        # Class 2: Base signal + level change
        .for_class(2)
        .add_signal(random_walk(step_size=0.2))
        .add_signal(gaussian(sigma=0.1), role="noise")
        .add_feature(level_change(amplitude=0.8), start_pct=0.6, end_pct=0.8)
        .build()
    )

    # Verify three classes were created
    classes, counts = np.unique(dataset["y"], return_counts=True)
    assert len(classes) == 3, f"Expected 3 classes, got {len(classes)}"
    assert set(classes) == {0, 1, 2}, f"Expected classes {0, 1, 2}, got {set(classes)}"

    # Check the shapes
    assert dataset["X"].shape == (n_samples, 1, n_timesteps), (
        f"Expected X shape (n_samples={n_samples}, n_dimensions=1, n_timesteps={n_timesteps}), "
        f"got {dataset['X'].shape}"
    )
    assert dataset["y"].shape == (n_samples,), (
        f"Expected y shape (n_samples={n_samples},), got {dataset['y'].shape}"
    )


def test_multivariate_two_classes(basic_multivariate_config):
    """Test building a multivariate dataset with two classes.

    Tests that:
    1. The builder correctly creates a multivariate dataset with specified dimensions
    2. Components are properly generated for each dimension
    3. Features are correctly applied to their respective dimensions
    """
    n_samples = basic_multivariate_config["n_samples"]
    n_timesteps = basic_multivariate_config["n_timesteps"]
    n_dimensions = basic_multivariate_config["n_dimensions"]

    dataset = (
        TimeSeriesBuilder(**basic_multivariate_config)
        # Class 0: Base signal in all dimensions
        .for_class(0)
        .add_signal(random_walk(step_size=0.2), dim=[0, 1])
        .add_signal(gaussian(sigma=0.1), role="noise", dim=[0, 1])
        # Class 1: Features in different dimensions
        .for_class(1)
        .add_signal(random_walk(step_size=0.2), dim=[0, 1])
        .add_signal(gaussian(sigma=0.1), role="noise", dim=[0, 1])
        .add_feature(shapelet(scale=1.0), start_pct=0.4, end_pct=0.6, dim=[0])
        .add_feature(level_change(amplitude=0.8), start_pct=0.6, end_pct=0.8, dim=[1])
        .build()
    )

    # Verify the shape includes all dimensions
    assert dataset["X"].shape == (n_samples, n_dimensions, n_timesteps), (
        f"Expected X shape (n_samples={n_samples}, n_dimensions={n_dimensions}, "
        f"n_timesteps={n_timesteps}), got {dataset['X'].shape}"
    )
    assert dataset["metadata"]["n_dimensions"] == n_dimensions, (
        f"Metadata should indicate {n_dimensions} dimensions"
    )

    # Ensure all components were created for each dimension
    sample_idx = np.where(dataset["y"] == 1)[0][0]  # Get first class 1 sample
    components = dataset["components"][sample_idx]

    # Check dimensions of foundation and noise
    assert components.foundation.shape == (n_timesteps, n_dimensions), (
        f"Foundation should have shape ({n_timesteps}, {n_dimensions}), "
        f"got {components.foundation.shape}"
    )
    assert components.noise.shape == (n_timesteps, n_dimensions), (
        f"Noise should have shape ({n_timesteps}, {n_dimensions}), "
        f"got {components.noise.shape}"
    )

    # Check feature names for each dimension
    feature_names = list(components.features.keys())
    assert any("dim0" in name for name in feature_names), (
        "Expected features for dimension 0 but none found"
    )
    assert any("dim1" in name for name in feature_names), (
        "Expected features for dimension 1 but none found"
    )


def test_multivariate_three_classes_and_dimensions():
    """Test building a multivariate dataset with three classes and three dimensions.

    Tests that:
    1. The builder correctly creates three classes with multivariate data
    2. The dataset has the correct shape for the requested dimensions
    3. All three classes are represented in the dataset
    """
    n_samples = 90
    n_timesteps = 40
    n_dimensions = 3

    dataset = (
        TimeSeriesBuilder(
            n_timesteps=n_timesteps,
            n_samples=n_samples,
            n_dimensions=n_dimensions,
            random_state=42,
        )
        # Class 0: Base signal in all dimensions
        .for_class(0)
        .add_signal(random_walk(step_size=0.2), dim=[0, 1, 2])
        .add_signal(gaussian(sigma=0.1), role="noise", dim=[0, 1, 2])
        # Class 1: Features in first two dimensions
        .for_class(1)
        .add_signal(random_walk(step_size=0.2), dim=[0, 1, 2])
        .add_signal(gaussian(sigma=0.1), role="noise", dim=[0, 1, 2])
        .add_feature(shapelet(scale=1.0), start_pct=0.3, end_pct=0.5, dim=[0, 1])
        # Class 2: Feature in third dimension
        .for_class(2)
        .add_signal(random_walk(step_size=0.2), dim=[0, 1, 2])
        .add_signal(gaussian(sigma=0.1), role="noise", dim=[0, 1, 2])
        .add_feature(peak(amplitude=1.0, width=3), start_pct=0.6, end_pct=0.7, dim=[2])
        .build()
    )

    # Verify three classes were created
    classes, counts = np.unique(dataset["y"], return_counts=True)
    assert len(classes) == 3, f"Expected 3 classes, got {len(classes)}"
    assert set(classes) == {0, 1, 2}, f"Expected classes {0, 1, 2}, got {set(classes)}"

    # Check the shapes
    assert dataset["X"].shape == (n_samples, n_dimensions, n_timesteps), (
        f"Expected X shape (n_samples={n_samples}, n_dimensions={n_dimensions}, "
        f"n_timesteps={n_timesteps}), got {dataset['X'].shape}"
    )


def test_train_test_split(basic_univariate_config):
    """Test the train_test_split functionality.

    Tests that:
    1. The data is split into train and test sets with the correct ratio
    2. The shapes of X_train, X_test, y_train, and y_test are correct
    3. Class distribution is preserved in both splits
    """
    n_samples = basic_univariate_config["n_samples"]
    train_ratio = 0.7

    dataset = (
        TimeSeriesBuilder(**basic_univariate_config)
        .for_class(0)
        .add_signal(random_walk(step_size=0.2))
        .for_class(1)
        .add_signal(random_walk(step_size=0.2))
        .add_feature(shapelet(scale=1.0), start_pct=0.4, end_pct=0.6)
        .build(train_test_split=train_ratio)
    )

    # Verify train/test sets were created
    assert "X_train" in dataset, "Dataset should contain 'X_train' key"
    assert "y_train" in dataset, "Dataset should contain 'y_train' key"
    assert "X_test" in dataset, "Dataset should contain 'X_test' key"
    assert "y_test" in dataset, "Dataset should contain 'y_test' key"

    # Check split ratio
    expected_train_size = int(n_samples * train_ratio)
    assert len(dataset["y_train"]) == expected_train_size, (
        f"Expected train set size {expected_train_size}, got {len(dataset['y_train'])}"
    )
    expected_test_size = n_samples - expected_train_size
    assert len(dataset["y_test"]) == expected_test_size, (
        f"Expected test set size {expected_test_size}, got {len(dataset['y_test'])}"
    )

    # Check that class distribution is similar in train and test
    train_classes, train_counts = np.unique(dataset["y_train"], return_counts=True)
    test_classes, test_counts = np.unique(dataset["y_test"], return_counts=True)
    assert set(train_classes) == set(test_classes), (
        "Train and test sets should contain the same classes"
    )

    # Check shapes
    assert dataset["X_train"].shape[0] == expected_train_size, (
        f"Expected {expected_train_size} training samples, "
        f"got {dataset['X_train'].shape[0]}"
    )
    assert dataset["X_test"].shape[0] == expected_test_size, (
        f"Expected {expected_test_size} test samples, got {dataset['X_test'].shape[0]}"
    )
    assert dataset["y_train"].shape[0] == expected_train_size, (
        f"Expected {expected_train_size} training labels, "
        f"got {dataset['y_train'].shape[0]}"
    )
    assert dataset["y_test"].shape[0] == expected_test_size, (
        f"Expected {expected_test_size} test labels, got {dataset['y_test'].shape[0]}"
    )


def test_random_feature_locations():
    """Test dataset with randomly located features.

    Tests that:
    1. The builder correctly generates features at random locations
    2. Feature masks are created with the correct dimensions
    3. Features are only present for the specified class
    """
    n_samples = 50
    n_timesteps = 100

    dataset = (
        TimeSeriesBuilder(n_timesteps=n_timesteps, n_samples=n_samples, random_state=42)
        .for_class(0)
        .add_signal(random_walk(step_size=0.2))
        .add_signal(gaussian(sigma=0.1), role="noise")
        .for_class(1)
        .add_signal(random_walk(step_size=0.2))
        .add_signal(gaussian(sigma=0.1), role="noise")
        .add_feature(shapelet(scale=1.0), random_location=True, length_pct=0.2)
        .add_feature(level_change(amplitude=0.5), random_location=True, length_pct=0.2)
        .build()
    )

    # Check feature masks
    feature_masks = dataset["feature_masks"]
    assert len(feature_masks) > 0, "Dataset should contain feature masks"

    # Get a sample from class 1
    class1_indices = np.where(dataset["y"] == 1)[0]
    assert len(class1_indices) > 0, "Dataset should contain samples from class 1"

    # Check that feature masks have correct dimensions
    for mask_key, mask in feature_masks.items():
        assert mask.shape == (n_samples, n_timesteps), (
            f"Feature mask should have shape ({n_samples}, {n_timesteps}), "
            f"got {mask.shape}"
        )

    # Verify features are present for class 1 but not class 0
    class1_features_present = False
    for mask_key, mask in feature_masks.items():
        if "class_1" in mask_key:
            class1_sample_idx = class1_indices[0]
            # At least one feature should have True values
            if mask[class1_sample_idx].any():
                class1_features_present = True
                break

    assert class1_features_present, "No features found for class 1 samples"


def test_shared_features_across_dimensions():
    """Test shared features across multiple dimensions.

    Tests that:
    1. Features can be shared across multiple dimensions
    2. When shared_location is True, features appear at the same position in all dimensions
    3. Feature masks correctly reflect the shared locations
    """
    n_samples = 50
    n_timesteps = 80
    n_dimensions = 2

    dataset = (
        TimeSeriesBuilder(
            n_timesteps=n_timesteps,
            n_samples=n_samples,
            n_dimensions=n_dimensions,
            random_state=42,
        )
        .for_class(0)
        .add_signal(random_walk(step_size=0.2), dim=[0, 1])
        .for_class(1)
        .add_signal(random_walk(step_size=0.2), dim=[0, 1])
        .add_feature(
            shapelet(scale=1.2),
            random_location=True,
            length_pct=0.15,
            dim=[0, 1],
            shared_location=True,
        )
        .build()
    )

    # Get a sample from class 1
    class1_indices = np.where(dataset["y"] == 1)[0]
    sample_idx = class1_indices[0]
    components = dataset["components"][sample_idx]

    # Get feature names
    feature_names = list(components.features.keys())
    dim0_features = [name for name in feature_names if "dim0" in name]
    dim1_features = [name for name in feature_names if "dim1" in name]

    # Shared locations should have features at the same positions in both dimensions
    for dim0_name in dim0_features:
        base_name = dim0_name.split("_dim")[0]
        dim1_name = f"{base_name}_dim1"

        if dim1_name in dim1_features:
            # Get masks for both dimensions
            dim0_mask = components.feature_masks[dim0_name]
            dim1_mask = components.feature_masks[dim1_name]

            # With shared location, the masks should be identical
            assert np.array_equal(dim0_mask, dim1_mask), (
                "Feature masks not identical with shared_location=True. "
                "Features should occur at the same position in both dimensions."
            )


def test_add_signal_segment_functionality():
    """Test add_signal_segment functionality with fixed and random locations.

    Tests that:
    1. Signal segments can be added at fixed locations
    2. Signal segments can be added at random locations
    3. Signal segments only affect the specified portion of the time series
    """
    n_samples = 50
    n_timesteps = 200

    # Create a dataset with signal segments at fixed locations
    dataset_fixed = (
        TimeSeriesBuilder(n_timesteps=n_timesteps, n_samples=n_samples, random_state=42)
        .for_class(0)
        .add_signal(
            random_walk(step_size=0.1)
        )  # Background signal for the entire series
        .add_signal_segment(
            constant(value=2.0),
            start_pct=0.25,
            end_pct=0.35,
        )  # Signal in first half
        .for_class(1)
        .add_signal(
            random_walk(step_size=0.1)
        )  # Background signal for the entire series
        .add_signal_segment(
            constant(value=2.0),
            start_pct=0.65,
            end_pct=0.75,
        )  # Signal in second half
        .build()
    )

    # Create a dataset with signal segments at random locations
    dataset_random = (
        TimeSeriesBuilder(n_timesteps=n_timesteps, n_samples=n_samples, random_state=42)
        .for_class(0)
        .add_signal(
            random_walk(step_size=0.1)
        )  # Background signal for the entire series
        .add_signal_segment(
            constant(value=2.0),
            random_location=True,
            length_pct=0.1,
        )
        .for_class(1)
        .add_signal(
            random_walk(step_size=0.1)
        )  # Background signal for the entire series
        .add_signal_segment(
            constant(value=2.0),
            random_location=True,
            length_pct=0.1,
        )
        .build()
    )

    # Verify the shape of the datasets
    assert dataset_fixed["X"].shape == (n_samples, 1, n_timesteps), (
        "Dataset with fixed signal segments has incorrect shape"
    )
    assert dataset_random["X"].shape == (n_samples, 1, n_timesteps), (
        "Dataset with random signal segments has incorrect shape"
    )

    # Check that the datasets contain two classes
    for dataset in [dataset_fixed, dataset_random]:
        classes, counts = np.unique(dataset["y"], return_counts=True)
        assert len(classes) == 2, f"Expected 2 classes, got {len(classes)}"


def test_to_df_functionality():
    """Test to_df method functionality for visualization preparation.

    Tests that:
    1. The to_df method converts a dataset to a proper pandas DataFrame
    2. The DataFrame contains the correct structure for visualization
    3. Different filtering and formatting options work correctly
    """
    # Create a simple dataset
    n_samples = 20
    n_timesteps = 30
    n_dimensions = 2

    dataset = (
        TimeSeriesBuilder(
            n_timesteps=n_timesteps,
            n_samples=n_samples,
            n_dimensions=n_dimensions,
            random_state=42,
        )
        .for_class(0)
        .add_signal(random_walk(step_size=0.2), dim=[0, 1])
        .for_class(1)
        .add_signal(random_walk(step_size=0.2), dim=[0, 1])
        .add_feature(shapelet(scale=1.0), start_pct=0.4, end_pct=0.6, dim=[0])
        .build()
    )

    # Convert to DataFrame with different filtering options
    df = TimeSeriesBuilder().to_df(dataset)
    df_samples = TimeSeriesBuilder().to_df(dataset, samples=[0, 2])
    df_classes = TimeSeriesBuilder().to_df(dataset, classes=[1])
    df_components = TimeSeriesBuilder().to_df(dataset, components=["aggregated"])
    df_dimensions = TimeSeriesBuilder().to_df(dataset, dimensions=[0])
    df_formatted = TimeSeriesBuilder().to_df(dataset, format_classes=True)

    # Check the structure of the basic DataFrame
    assert isinstance(df, pd.DataFrame), "to_df should return a pandas DataFrame"
    assert len(df) > 0, "DataFrame should not be empty"
    assert all(
        col in df.columns
        for col in ["time", "value", "class", "sample", "component", "feature", "dim"]
    ), "DataFrame should have all expected columns"

    # Check filtering results
    assert set(df_samples["sample"].unique()) == {0, 2}, (
        "Sample filtering should only include specified samples"
    )
    assert set(df_classes["class"].unique()) == {1}, (
        "Class filtering should only include specified classes"
    )
    assert set(df_components["component"].unique()) == {"aggregated"}, (
        "Component filtering should only include specified components"
    )
    assert set(df_dimensions["dim"].unique()) == {0}, (
        "Dimension filtering should only include specified dimensions"
    )

    # Check class formatting
    assert all("Class" in str(cls) for cls in df_formatted["class"].unique()), (
        "When format_classes=True, class labels should be formatted as 'Class X'"
    )


def test_normalization_options():
    """Test different normalization options.

    Tests that:
    1. Z-score normalization works correctly
    2. Min-max normalization works correctly
    3. No normalization works correctly
    """
    n_samples = 30
    n_timesteps = 50

    # Create datasets with different normalization options
    datasets = {}
    for norm in ["zscore", "minmax", "none"]:
        datasets[norm] = (
            TimeSeriesBuilder(
                n_timesteps=n_timesteps,
                n_samples=n_samples,
                normalization=norm,
                random_state=42,
            )
            .for_class(0)
            .add_signal(random_walk(step_size=0.5))
            .build()
        )

    # Z-score normalization should have mean close to 0 and std dev close to 1
    zscore_data = datasets["zscore"]["X"].reshape(-1)
    assert -0.1 < np.mean(zscore_data) < 0.1, (
        f"Z-score normalized data should have mean close to 0, got {np.mean(zscore_data)}"
    )
    assert 0.9 < np.std(zscore_data) < 1.1, (
        f"Z-score normalized data should have std dev close to 1, got {np.std(zscore_data)}"
    )

    # Min-max normalization should be in range [0, 1]
    minmax_data = datasets["minmax"]["X"].reshape(-1)
    assert np.min(minmax_data) >= 0 and np.max(minmax_data) <= 1, (
        "Min-max normalized data should be in range [0, 1]"
    )

    # No normalization should preserve data range
    none_data = datasets["none"]["X"].reshape(-1)
    # Different from z-score and min-max
    assert not (-0.1 < np.mean(none_data) < 0.1 and 0.9 < np.std(none_data) < 1.1), (
        "Non-normalized data should not match z-score normalization pattern"
    )
    assert not (np.min(none_data) >= 0 and np.max(none_data) <= 1), (
        "Non-normalized data should not match min-max normalization pattern"
    )
