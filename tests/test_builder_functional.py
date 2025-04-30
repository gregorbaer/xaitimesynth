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
    peak,
    random_walk,
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
        .for_class(1)  # Class 1: Has constant feature
        .add_signal(random_walk(step_size=0.2))
        .add_signal(gaussian(sigma=0.1), role="noise")
        .add_feature(constant(value=1.0), start_pct=0.4, end_pct=0.6)
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
        # Class 1: Base signal + constant level change
        .for_class(1)
        .add_signal(random_walk(step_size=0.2))
        .add_signal(gaussian(sigma=0.1), role="noise")
        .add_feature(constant(value=1.0), start_pct=0.4, end_pct=0.6)
        # Class 2: Base signal + constant level change
        .for_class(2)
        .add_signal(random_walk(step_size=0.2))
        .add_signal(gaussian(sigma=0.1), role="noise")
        .add_feature(constant(value=0.8), start_pct=0.6, end_pct=0.8)
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
        .add_feature(constant(value=1.0), start_pct=0.4, end_pct=0.6, dim=[0])
        .add_feature(constant(value=0.8), start_pct=0.6, end_pct=0.8, dim=[1])
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
        .add_feature(constant(value=1.0), start_pct=0.3, end_pct=0.5, dim=[0, 1])
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


def test_clone_method(basic_univariate_config):
    """Test the clone method for creating train/test splits with consistent patterns.

    Tests that:
    1. The clone method preserves class definitions from the original builder
    2. Different n_samples parameters correctly control the size of the datasets
    3. Different random_state parameters produce different but structurally similar datasets
    4. All dataset properties are correctly maintained in the cloned builders
    """
    # Create a base builder with class definitions
    base_builder = (
        TimeSeriesBuilder(**basic_univariate_config)
        .for_class(0)  # Class 0: No discriminative features
        .add_signal(random_walk(step_size=0.2))
        .add_signal(gaussian(sigma=0.1), role="noise")
        .for_class(1)  # Class 1: Has constant feature
        .add_signal(random_walk(step_size=0.2))
        .add_signal(gaussian(sigma=0.1), role="noise")
        .add_feature(constant(scale=1.0), start_pct=0.4, end_pct=0.6)
    )

    # Generate train dataset with 70% of samples and the same random seed
    train_samples = int(basic_univariate_config["n_samples"] * 0.7)
    train_dataset = base_builder.clone(n_samples=train_samples, random_state=42).build()

    # Generate test dataset with 30% of samples and a different random seed
    test_samples = int(basic_univariate_config["n_samples"] * 0.3)
    test_dataset = base_builder.clone(n_samples=test_samples, random_state=43).build()

    # Verify the datasets have the correct sizes
    assert train_dataset["X"].shape[0] == train_samples, (
        f"Expected {train_samples} training samples, got {train_dataset['X'].shape[0]}"
    )
    assert test_dataset["X"].shape[0] == test_samples, (
        f"Expected {test_samples} test samples, got {test_dataset['X'].shape[0]}"
    )

    # Check that both datasets have the same class labels
    train_classes = set(np.unique(train_dataset["y"]))
    test_classes = set(np.unique(test_dataset["y"]))
    assert train_classes == test_classes, (
        f"Train and test datasets should have the same classes. "
        f"Train: {train_classes}, Test: {test_classes}"
    )

    # Verify datasets are different due to different random seeds
    # Get first sample from each dataset
    train_first_sample = train_dataset["X"][0]
    test_first_sample = test_dataset["X"][0]

    assert not np.array_equal(train_first_sample, test_first_sample), (
        "Different random seeds should produce different data patterns"
    )

    # Test cloning with different parameters
    modified_builder = base_builder.clone(
        n_timesteps=200,  # Different time steps
        normalization="minmax",  # Different normalization
    )
    modified_dataset = modified_builder.build()

    # Check that the parameters were properly updated
    assert modified_dataset["X"].shape[2] == 200, (
        "Modified timesteps parameter should be reflected in the data shape"
    )

    # Check normalization change (minmax should be in range [0, 1])
    data_values = modified_dataset["X"].reshape(-1)
    assert np.min(data_values) >= 0 and np.max(data_values) <= 1, (
        "Min-max normalized data should be in range [0, 1]"
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
        .add_feature(constant(scale=1.0), random_location=True, length_pct=0.2)
        .add_feature(constant(amplitude=0.5), random_location=True, length_pct=0.2)
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
            constant(scale=1.2),
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
        .add_feature(constant(scale=1.0), start_pct=0.4, end_pct=0.6, dim=[0])
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


def test_deterministic_class_counts():
    """Test deterministic_class_counts parameter in the build method.

    Tests that:
    1. When deterministic_class_counts=True, class counts exactly match the expected proportions
    2. When deterministic_class_counts=False, class counts follow multinomial distribution
    3. Both methods produce valid datasets with the correct total sample count
    """
    n_samples = 100
    n_timesteps = 50

    # Define class weights
    class0_weight = 0.2
    class1_weight = 0.3
    class2_weight = 0.5

    # Expected counts with deterministic sampling
    expected_class0 = int(n_samples * class0_weight)  # 20
    expected_class1 = int(n_samples * class1_weight)  # 30
    expected_class2 = n_samples - expected_class0 - expected_class1  # 50

    # Create a builder with three classes and different weights
    builder = (
        TimeSeriesBuilder(n_timesteps=n_timesteps, n_samples=n_samples, random_state=42)
        .for_class(0, weight=class0_weight)
        .add_signal(random_walk(step_size=0.2))
        .add_signal(gaussian(sigma=0.1), role="noise")
        .for_class(1, weight=class1_weight)
        .add_signal(random_walk(step_size=0.2))
        .add_signal(gaussian(sigma=0.1), role="noise")
        .for_class(2, weight=class2_weight)
        .add_signal(random_walk(step_size=0.2))
        .add_signal(gaussian(sigma=0.1), role="noise")
    )

    # Build with deterministic class counts
    deterministic_dataset = builder.build(deterministic_class_counts=True)

    # Build with probabilistic class counts (the default)
    probabilistic_dataset = builder.build(deterministic_class_counts=False)

    # Verify total sample counts
    assert len(deterministic_dataset["y"]) == n_samples, (
        f"Deterministic dataset should have {n_samples} samples, "
        f"got {len(deterministic_dataset['y'])}"
    )
    assert len(probabilistic_dataset["y"]) == n_samples, (
        f"Probabilistic dataset should have {n_samples} samples, "
        f"got {len(probabilistic_dataset['y'])}"
    )

    # Get class counts for both datasets
    deterministic_classes, deterministic_counts = np.unique(
        deterministic_dataset["y"], return_counts=True
    )
    probabilistic_classes, probabilistic_counts = np.unique(
        probabilistic_dataset["y"], return_counts=True
    )

    # Create dictionaries mapping class labels to counts
    det_count_dict = dict(zip(deterministic_classes, deterministic_counts))
    prob_count_dict = dict(zip(probabilistic_classes, probabilistic_counts))

    # Verify deterministic counts exactly match expected proportions
    assert det_count_dict[0] == expected_class0, (
        f"Expected exactly {expected_class0} samples for class 0 with deterministic sampling, "
        f"got {det_count_dict[0]}"
    )
    assert det_count_dict[1] == expected_class1, (
        f"Expected exactly {expected_class1} samples for class 1 with deterministic sampling, "
        f"got {det_count_dict[1]}"
    )
    assert det_count_dict[2] == expected_class2, (
        f"Expected exactly {expected_class2} samples for class 2 with deterministic sampling, "
        f"got {det_count_dict[2]}"
    )

    # For probabilistic counts, we can't test exact values since they're random
    # Instead, verify that all classes are present and the distribution seems reasonable
    assert len(probabilistic_classes) == 3, (
        f"Expected all 3 classes to be present with probabilistic sampling, "
        f"got {len(probabilistic_classes)}"
    )

    # Optional: Run multiple probabilistic builds to verify the distribution varies
    # Note: This is a statistical test, so there's a tiny chance it could fail randomly
    counts_vary = False
    for _ in range(5):
        another_dataset = builder.build(deterministic_class_counts=False)
        another_classes, another_counts = np.unique(
            another_dataset["y"], return_counts=True
        )
        another_count_dict = dict(zip(another_classes, another_counts))

        # If any count differs from our first probabilistic sample, we've shown variation
        if any(another_count_dict[i] != prob_count_dict[i] for i in range(3)):
            counts_vary = True
            break

    assert counts_vary, (
        "Probabilistic sampling should produce varying class counts across multiple runs"
    )
