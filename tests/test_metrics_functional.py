"""Functional tests for metrics module.

This module contains tests that verify the functionality of the attribution metrics,
including precision, recall, and F1 score. Tests cover various scenarios including
perfect/partial attributions, multivariate time series, continuous attributions with
different thresholds, and various averaging methods.
"""

import numpy as np
import pytest

from xaitimesynth import (
    TimeSeriesBuilder,
    gaussian,
    level_change,
    seasonal,
)
from xaitimesynth.metrics import (
    auc_pr_score,
    auc_roc_score,
    correlation_score,
    f1_score,
    nac_score,
    precision_score,
    recall_score,
)


@pytest.fixture(scope="module")
def simple_dataset():
    """Create a simple dataset with a clear feature at known position.

    This dataset has two classes (0 and 1), with class 1 containing a distinct
    level change feature between 40-60% of the time series length.

    Returns:
        dict: Dataset with a simple feature in class 1 at fixed position.
    """
    dataset = (
        TimeSeriesBuilder(n_timesteps=20, n_samples=2, random_state=42)
        .for_class(0)
        .add_signal(gaussian(), role="foundation")
        .for_class(1)
        .add_signal(gaussian(), role="foundation")
        .add_feature(level_change(amplitude=1.0), start_pct=0.4, end_pct=0.6)
        .build()
    )
    return dataset


@pytest.fixture(scope="module")
def multivariate_dataset():
    """Create a simple multivariate dataset with features in different dimensions.

    This dataset has two dimensions with distinct features in each dimension:
    - Dimension 0 has a level change feature at 30-40% of the time series
    - Dimension 1 has a level change feature at 60-70% of the time series

    Returns:
        dict: Dataset with features in different dimensions.
    """
    dataset = (
        TimeSeriesBuilder(n_timesteps=20, n_samples=2, n_dimensions=2, random_state=42)
        .for_class(0)
        .add_signal(gaussian(), dim=[0, 1])
        .for_class(1)
        .add_signal(gaussian(), dim=[0, 1])
        # Feature in dimension 0
        .add_feature(level_change(amplitude=1.0), start_pct=0.3, end_pct=0.4, dim=[0])
        # Feature in dimension 1
        .add_feature(level_change(amplitude=1.0), start_pct=0.6, end_pct=0.7, dim=[1])
        .build()
    )
    return dataset


@pytest.fixture(scope="module")
def random_feature_dataset():
    """Create a simple dataset with randomly positioned features.

    This dataset contains randomly positioned level change features
    in class 1 samples, which helps test the metrics' ability to handle
    features that aren't at fixed positions.

    Returns:
        dict: Dataset with randomly positioned features.
    """
    dataset = (
        TimeSeriesBuilder(n_timesteps=20, n_samples=2, random_state=42)
        .for_class(0)
        .add_signal(gaussian())
        .for_class(1)
        .add_signal(gaussian())
        .add_feature(level_change(amplitude=1.0), random_location=True, length_pct=0.2)
        .build()
    )
    return dataset


@pytest.fixture
def get_class1_sample_and_feature_mask():
    """Generate a function to extract first class 1 sample and its feature mask.

    This helper function simplifies access to feature masks for testing.

    Returns:
        function: Helper function to extract sample and mask.
    """

    def _get_sample_and_mask(dataset):
        class1_indices = np.where(dataset["y"] == 1)[0]
        sample_idx = class1_indices[0]
        feature_key = next(k for k in dataset["feature_masks"].keys())
        mask = dataset["feature_masks"][feature_key][sample_idx]
        return sample_idx, mask

    return _get_sample_and_mask


@pytest.mark.parametrize(
    "attribution_creator,expected_precision,expected_recall,expected_f1,test_name",
    [
        # Perfect attribution case - attribution exactly matches the feature mask
        (
            lambda mask: mask,
            1.0,
            1.0,
            1.0,
            "perfect_attribution",
        ),
        # Partial attribution case - attribution covers only the first half of the feature
        (
            lambda mask: np.logical_and(
                mask, np.arange(len(mask)) < (np.argmax(mask) + np.sum(mask) // 2)
            )
            if np.any(mask)
            else np.zeros_like(mask),
            1.0,
            0.5,
            0.6667,  # 2 * (1.0 * 0.5) / (1.0 + 0.5) ≈ 0.6667
            "partial_attribution",
        ),
        # Completely wrong attribution - feature is identified in the wrong location
        (
            lambda mask: np.roll(mask, 10),
            0.0,
            0.0,
            0.0,
            "wrong_attribution",
        ),
    ],
)
def test_attribution_scenarios(
    simple_dataset,
    get_class1_sample_and_feature_mask,
    attribution_creator,
    expected_precision,
    expected_recall,
    expected_f1,
    test_name,
):
    """Test different attribution scenarios against expected metrics.

    This test covers three key scenarios:
    1. Perfect attribution: Exactly identifying the feature
    2. Partial attribution: Identifying only part of the feature
    3. Wrong attribution: Completely missing the feature location

    Each scenario validates precision, recall and F1 scores against expected values.
    """
    sample_idx, mask = get_class1_sample_and_feature_mask(simple_dataset)
    n_timesteps = simple_dataset["metadata"]["n_timesteps"]
    n_dimensions = simple_dataset["metadata"]["n_dimensions"]

    # Create attribution based on the parametrized creator function
    attr = np.zeros((1, n_timesteps, n_dimensions), dtype=bool)
    attr[0, :, 0] = attribution_creator(mask)

    # Calculate metrics
    precision = precision_score(
        attr, simple_dataset, sample_indices=[sample_idx], class_label=1
    )
    recall = recall_score(
        attr, simple_dataset, sample_indices=[sample_idx], class_label=1
    )
    f1 = f1_score(attr, simple_dataset, sample_indices=[sample_idx], class_label=1)

    # Assert expected results with informative error messages
    assert precision == pytest.approx(expected_precision, abs=0.05), (
        f"{test_name} should have precision of {expected_precision}"
    )
    assert recall == pytest.approx(expected_recall, abs=0.05), (
        f"{test_name} should have recall of {expected_recall}"
    )
    assert f1 == pytest.approx(expected_f1, abs=0.05), (
        f"{test_name} should have F1 score of {expected_f1}"
    )


def test_wider_attribution(simple_dataset, get_class1_sample_and_feature_mask):
    """Test attribution wider than the feature.

    This test evaluates what happens when the attribution region is larger than
    the actual feature - precision should decrease (due to false positives)
    while recall should remain perfect (all true feature points are captured).
    """
    sample_idx, mask = get_class1_sample_and_feature_mask(simple_dataset)
    n_timesteps = simple_dataset["metadata"]["n_timesteps"]
    n_dimensions = simple_dataset["metadata"]["n_dimensions"]

    # Find feature boundaries
    feature_start = np.argmax(mask)
    feature_end = len(mask) - np.argmax(mask[::-1])
    feature_length = feature_end - feature_start

    # Create wider attribution (feature + padding on both sides)
    padding = 2  # Reduced from 5 for smaller dataset
    wider_attr = np.zeros((1, n_timesteps, n_dimensions), dtype=bool)
    start_idx = max(0, feature_start - padding)
    end_idx = min(n_timesteps, feature_end + padding)
    wider_attr[0, start_idx:end_idx, 0] = True

    # Calculate metrics
    precision = precision_score(
        wider_attr, simple_dataset, sample_indices=[sample_idx], class_label=1
    )
    recall = recall_score(
        wider_attr, simple_dataset, sample_indices=[sample_idx], class_label=1
    )

    expected_precision = feature_length / (feature_length + 2 * padding)

    assert precision == pytest.approx(expected_precision, abs=0.05), (
        "Precision should decrease with wider attribution"
    )
    assert recall == 1.0, "Wider attribution should have recall of 1.0"


@pytest.mark.parametrize(
    "average,expected_precision,expected_recall",
    [
        ("macro", 1.0, 1.0),
        ("per_dimension", [1.0, 1.0], [1.0, 1.0]),
    ],
)
def test_multivariate_perfect_attribution(
    multivariate_dataset, average, expected_precision, expected_recall
):
    """Test metrics for multivariate time series with perfect attribution.

    This test verifies that both 'macro' and 'per_dimension' averaging methods
    work correctly when attributions perfectly match the features in both dimensions.
    """
    # Get class 1 sample
    class1_indices = np.where(multivariate_dataset["y"] == 1)[0]
    sample_idx = class1_indices[0]
    n_timesteps = multivariate_dataset["metadata"]["n_timesteps"]
    n_dimensions = multivariate_dataset["metadata"]["n_dimensions"]

    # Create perfect attribution for both dimensions
    attribution = np.zeros((1, n_timesteps, n_dimensions), dtype=bool)

    # First, find all feature masks for this sample
    feature_masks = {}
    for key, masks in multivariate_dataset["feature_masks"].items():
        feature_masks[key] = masks[sample_idx]

    # For each dimension, combine all feature masks for that dimension
    for dim_idx in range(n_dimensions):
        dim_mask = np.zeros(n_timesteps, dtype=bool)
        for key, mask in feature_masks.items():
            if f"_dim{dim_idx}" in key:
                dim_mask |= mask
            # Also include masks without explicit dimension (they belong to dim0)
            elif dim_idx == 0 and "_dim" not in key:
                dim_mask |= mask

        # Set the attribution to match the combined mask for this dimension
        attribution[0, :, dim_idx] = dim_mask

    # Calculate metrics
    precision = precision_score(
        attribution,
        multivariate_dataset,
        sample_indices=[sample_idx],
        average=average,
    )
    recall = recall_score(
        attribution,
        multivariate_dataset,
        sample_indices=[sample_idx],
        average=average,
    )

    # Assert with informative messages
    if isinstance(expected_precision, list):
        assert precision[0] == pytest.approx(expected_precision[0], abs=0.05), (
            f"Dimension 0 precision should be {expected_precision[0]}"
        )
        assert precision[1] == pytest.approx(expected_precision[1], abs=0.05), (
            f"Dimension 1 precision should be {expected_precision[1]}"
        )
        assert recall[0] == pytest.approx(expected_recall[0], abs=0.05), (
            f"Dimension 0 recall should be {expected_recall[0]}"
        )
        assert recall[1] == pytest.approx(expected_recall[1], abs=0.05), (
            f"Dimension 1 recall should be {expected_recall[1]}"
        )
    else:
        assert precision == pytest.approx(expected_precision, abs=0.05), (
            f"Macro precision should be {expected_precision}"
        )
        assert recall == pytest.approx(expected_recall, abs=0.05), (
            f"Macro recall should be {expected_recall}"
        )


def test_mixed_performance(multivariate_dataset):
    """Test mixed performance across dimensions - perfect in dim0, poor in dim1.

    This test verifies that the metrics correctly handle cases where attributions
    perform well in one dimension but poorly in another. It tests:
    1. Per-dimension metrics showing the performance difference
    2. Macro averaging, which should reflect the average performance across dimensions
    """
    # Get class 1 sample
    class1_indices = np.where(multivariate_dataset["y"] == 1)[0]
    sample_idx = class1_indices[0]
    n_timesteps = multivariate_dataset["metadata"]["n_timesteps"]
    n_dimensions = multivariate_dataset["metadata"]["n_dimensions"]

    # Create mixed attribution
    mixed_attr = np.zeros((1, n_timesteps, n_dimensions), dtype=bool)

    # First, find all feature masks for this sample
    feature_masks = {}
    for key, masks in multivariate_dataset["feature_masks"].items():
        feature_masks[key] = masks[sample_idx]

    # For dimension 0, combine all feature masks to make perfect attribution
    dim0_mask = np.zeros(n_timesteps, dtype=bool)
    for key, mask in feature_masks.items():
        if "_dim0" in key or "_dim" not in key:
            dim0_mask |= mask

    # Set dimension 0 attribution to be perfect
    mixed_attr[0, :, 0] = dim0_mask

    # For dimension 1, find the feature but place it elsewhere
    dim1_mask = np.zeros(n_timesteps, dtype=bool)
    for key, mask in feature_masks.items():
        if "_dim1" in key:
            dim1_mask |= mask

    # If we found a dimension 1 feature
    if np.any(dim1_mask):
        feature_start = np.argmax(dim1_mask)
        feature_length = np.sum(dim1_mask)

        # Put attribution in wrong place - reduced shift due to smaller dataset
        wrong_start = (feature_start + 6) % (n_timesteps - feature_length)
        mixed_attr[0, wrong_start : wrong_start + feature_length, 1] = True

    # Calculate per-dimension metrics
    precision_per_dim = precision_score(
        mixed_attr,
        multivariate_dataset,
        sample_indices=[sample_idx],
        average="per_dimension",
    )
    recall_per_dim = recall_score(
        mixed_attr,
        multivariate_dataset,
        sample_indices=[sample_idx],
        average="per_dimension",
    )

    # Per-dimension assertions
    assert precision_per_dim[0] == 1.0, "Dimension 0 should have perfect precision"
    assert recall_per_dim[0] == 1.0, "Dimension 0 should have perfect recall"
    assert precision_per_dim[1] == 0.0, "Dimension 1 should have zero precision"
    assert recall_per_dim[1] == 0.0, "Dimension 1 should have zero recall"

    # Macro average assertions
    precision_macro = precision_score(
        mixed_attr,
        multivariate_dataset,
        sample_indices=[sample_idx],
        average="macro",
    )
    recall_macro = recall_score(
        mixed_attr,
        multivariate_dataset,
        sample_indices=[sample_idx],
        average="macro",
    )

    assert precision_macro == 0.5, "Macro precision should be 0.5"
    assert recall_macro == 0.5, "Macro recall should be 0.5"


@pytest.mark.parametrize(
    "threshold,expected_precision,expected_recall",
    [
        (0.2, 0.8, 0.8),  # Low threshold - high recall but lower precision
        (0.5, 0.9, 0.5),  # Medium threshold - balanced precision/recall
        (0.8, 1.0, 0.2),  # High threshold - high precision but lower recall
    ],
)
def test_thresholding(
    simple_dataset,
    get_class1_sample_and_feature_mask,
    threshold,
    expected_precision,
    expected_recall,
):
    """Test continuous attribution values with different thresholds.

    This test verifies the threshold parameter functionality by checking how
    different threshold values affect precision and recall when applied to
    continuous attribution values. It demonstrates the precision-recall tradeoff
    that occurs when adjusting thresholds.
    """
    sample_idx, mask = get_class1_sample_and_feature_mask(simple_dataset)
    n_timesteps = simple_dataset["metadata"]["n_timesteps"]
    n_dimensions = simple_dataset["metadata"]["n_dimensions"]

    # Find feature center
    feature_start = np.argmax(mask)
    feature_end = len(mask) - np.argmax(mask[::-1])
    feature_center = feature_start + (feature_end - feature_start) // 2

    # Create continuous attribution values (highest at feature center)
    continuous_attr = np.zeros((1, n_timesteps, n_dimensions))
    for i in range(n_timesteps):
        # Distance from center, normalized to 0-1 range
        distance = abs(i - feature_center) / (n_timesteps // 2)
        continuous_attr[0, i, 0] = max(0, 1 - distance)  # Linear decay from center

    # Binary attribution for manual calculation
    binary_attr = continuous_attr >= threshold

    # Manual calculation
    true_positives = np.sum(binary_attr[0, :, 0] & mask)
    false_positives = np.sum(binary_attr[0, :, 0] & ~mask)
    false_negatives = np.sum(~binary_attr[0, :, 0] & mask)

    expected_precision_calc = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else 0
    )
    expected_recall_calc = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0
        else 0
    )

    # Calculate metrics using the library functions
    precision = precision_score(
        continuous_attr,
        simple_dataset,
        sample_indices=[sample_idx],
        class_label=1,
        threshold=threshold,
    )
    recall = recall_score(
        continuous_attr,
        simple_dataset,
        sample_indices=[sample_idx],
        class_label=1,
        threshold=threshold,
    )

    # Assert using calculated values instead of fixed expected values
    assert precision == pytest.approx(expected_precision_calc, abs=0.05), (
        f"Precision with threshold {threshold} should be approximately {expected_precision_calc}"
    )
    assert recall == pytest.approx(expected_recall_calc, abs=0.05), (
        f"Recall with threshold {threshold} should be approximately {expected_recall_calc}"
    )


def test_multiple_samples(multivariate_dataset):
    """Test metrics calculated on multiple samples.

    This test verifies that metrics can be calculated across multiple samples
    and confirms that per-sample metrics work correctly, as well as the overall
    macro average across all samples.
    """
    # Get all class 1 samples
    class1_indices = np.where(multivariate_dataset["y"] == 1)[0]
    n_samples = len(class1_indices)
    n_timesteps = multivariate_dataset["metadata"]["n_timesteps"]
    n_dimensions = multivariate_dataset["metadata"]["n_dimensions"]

    # Create attributions for all samples - perfect for dim0, wrong for dim1
    attributions = np.zeros((n_samples, n_timesteps, n_dimensions), dtype=bool)

    for i, sample_idx in enumerate(class1_indices):
        # Get all feature masks for this sample
        feature_masks = {}
        for key, masks in multivariate_dataset["feature_masks"].items():
            feature_masks[key] = masks[sample_idx]

        # For dimension 0, perfect attribution
        dim0_mask = np.zeros(n_timesteps, dtype=bool)
        for key, mask in feature_masks.items():
            if "_dim0" in key or "_dim" not in key:
                dim0_mask |= mask
        attributions[i, :, 0] = dim0_mask

        # For dimension 1, find the feature but place it elsewhere
        dim1_mask = np.zeros(n_timesteps, dtype=bool)
        for key, mask in feature_masks.items():
            if "_dim1" in key:
                dim1_mask |= mask

        if np.any(dim1_mask):
            feature_start = np.argmax(dim1_mask)
            feature_length = np.sum(dim1_mask)
            # Reduced shift for smaller dataset
            wrong_start = (feature_start + 6) % (n_timesteps - feature_length)
            attributions[i, wrong_start : wrong_start + feature_length, 1] = True

    # Calculate per-sample metrics
    precision_per_sample = precision_score(
        attributions,
        multivariate_dataset,
        sample_indices=class1_indices,
        average="per_sample",
    )

    # Each sample should have 0.5 precision
    for sample_idx in class1_indices:
        assert precision_per_sample[sample_idx] == pytest.approx(0.5, abs=0.05), (
            f"Sample {sample_idx} precision should be 0.5"
        )

    # Overall macro average should be 0.5
    precision_macro = precision_score(
        attributions,
        multivariate_dataset,
        sample_indices=class1_indices,
        average="macro",
    )
    assert precision_macro == pytest.approx(0.5, abs=0.05), (
        "Macro precision for all samples should be 0.5"
    )


def test_random_feature_dataset(random_feature_dataset):
    """Test metrics on a dataset with randomly positioned features.

    This test verifies that metrics work correctly even when feature positions
    are randomly determined rather than at fixed positions.
    """
    class1_indices = np.where(random_feature_dataset["y"] == 1)[0]
    sample_idx = class1_indices[0]
    n_timesteps = random_feature_dataset["metadata"]["n_timesteps"]
    n_dimensions = random_feature_dataset["metadata"]["n_dimensions"]

    # Get the feature mask
    feature_key = next(k for k in random_feature_dataset["feature_masks"].keys())
    mask = random_feature_dataset["feature_masks"][feature_key][sample_idx]

    # Create perfect attribution
    perfect_attr = np.zeros((1, n_timesteps, n_dimensions), dtype=bool)
    perfect_attr[0, :, 0] = mask

    # Calculate metrics
    precision = precision_score(
        perfect_attr,
        random_feature_dataset,
        sample_indices=[sample_idx],
        class_label=1,
    )
    recall = recall_score(
        perfect_attr,
        random_feature_dataset,
        sample_indices=[sample_idx],
        class_label=1,
    )

    assert precision == 1.0, (
        "Perfect attribution should have precision of 1.0 even for random features"
    )
    assert recall == 1.0, (
        "Perfect attribution should have recall of 1.0 even for random features"
    )


def test_per_sample_dimension_average():
    """Test the 'per_sample_dimension' averaging method.

    This test verifies that the per_sample_dimension averaging method correctly
    returns metrics for each combination of sample and dimension.
    """
    # Create a small dataset with features in different dimensions
    dataset = (
        TimeSeriesBuilder(n_timesteps=20, n_samples=2, n_dimensions=2, random_state=42)
        .for_class(0)
        .add_signal(gaussian(), dim=[0, 1])
        .for_class(1)
        .add_signal(gaussian(), dim=[0, 1])
        .add_feature(level_change(amplitude=1.0), start_pct=0.3, end_pct=0.4, dim=[0])
        .add_feature(level_change(amplitude=1.0), start_pct=0.6, end_pct=0.7, dim=[1])
        .build()
    )

    # Get sample indices for class 1
    class1_indices = np.where(dataset["y"] == 1)[0]
    sample_idx = class1_indices[0]
    n_timesteps = dataset["metadata"]["n_timesteps"]
    n_dimensions = dataset["metadata"]["n_dimensions"]

    # Create perfect attribution for dimension 0, wrong for dimension 1
    attr = np.zeros((1, n_timesteps, n_dimensions), dtype=bool)

    # Get feature masks
    feature_masks = {}
    for key, masks in dataset["feature_masks"].items():
        if "_dim0" in key or "_dim" not in key:  # dim0 or unspecified dimension
            feature_masks[key] = masks[sample_idx]

    # Create perfect attribution for dimension 0
    dim0_mask = np.zeros(n_timesteps, dtype=bool)
    for mask in feature_masks.values():
        dim0_mask |= mask
    attr[0, :, 0] = dim0_mask

    # For dimension 1, create wrong attribution
    attr[0, 0:4, 1] = True  # Arbitrary wrong location

    # Calculate per_sample_dimension metrics
    precision = precision_score(
        attr, dataset, sample_indices=[sample_idx], average="per_sample_dimension"
    )

    # Check results for each sample-dimension pair
    assert precision[(sample_idx, 0)] == 1.0, (
        "Dimension 0 should have perfect precision"
    )
    assert precision[(sample_idx, 1)] == 0.0, "Dimension 1 should have zero precision"


def test_empty_attribution():
    """Test metrics with empty attributions (no features identified).

    This test verifies that metrics handle the edge case of empty attributions
    (where no features are identified) correctly.
    """
    # Create a simple dataset with a feature
    dataset = (
        TimeSeriesBuilder(n_timesteps=20, n_samples=1, random_state=42)
        .for_class(1)
        .add_signal(gaussian())
        .add_feature(level_change(amplitude=1.0), start_pct=0.4, end_pct=0.6)
        .build()
    )

    # Create empty attribution (all zeros)
    n_timesteps = dataset["metadata"]["n_timesteps"]
    n_dimensions = dataset["metadata"]["n_dimensions"]
    attr = np.zeros((1, n_timesteps, n_dimensions), dtype=bool)

    # Calculate metrics
    precision = precision_score(attr, dataset, sample_indices=[0], class_label=1)
    recall = recall_score(attr, dataset, sample_indices=[0], class_label=1)
    f1 = f1_score(attr, dataset, sample_indices=[0], class_label=1)

    # For empty attribution, precision should be undefined (returning 0)
    # Recall should be 0 (no true positives out of all positives)
    assert precision == 0.0, "Precision should be 0 for empty attribution"
    assert recall == 0.0, "Recall should be 0 for empty attribution"
    assert f1 == 0.0, "F1 score should be 0 for empty attribution"


def test_error_handling():
    """Test error handling for invalid inputs.

    This test verifies that appropriate error messages are raised when
    invalid inputs are provided to the metric functions.
    """
    # Create a simple dataset
    dataset = (
        TimeSeriesBuilder(n_timesteps=20, n_samples=2, random_state=42)
        .for_class(0)
        .add_signal(gaussian())
        .for_class(1)
        .add_signal(gaussian())
        .add_feature(level_change(amplitude=1.0), start_pct=0.4, end_pct=0.6)
        .build()
    )

    # Case 1: Non-boolean attribution without threshold
    with pytest.raises(ValueError, match="Attribution must be boolean type"):
        continuous_attr = np.random.rand(1, 20, 1)  # Continuous values between 0 and 1
        precision_score(continuous_attr, dataset, sample_indices=[0])

    # Case 2: Sample index out of range
    with pytest.raises(ValueError, match="Sample index .* out of range"):
        attr = np.zeros((1, 20, 1), dtype=bool)
        precision_score(attr, dataset, sample_indices=[10])  # Only 2 samples in dataset

    # Case 3: Wrong shape attributions
    with pytest.raises(ValueError, match="doesn't match dataset dimensions"):
        attr = np.zeros((1, 30, 1), dtype=bool)  # Dataset has 20 timesteps
        precision_score(attr, dataset, sample_indices=[0])


# Update existing test case and add new tests for data format conversion


def test_data_format_conversion():
    """Test the tensor data format conversion feature.

    This test verifies that:
    1. Default format can be changed to channels_first (PyTorch/tsai format)
    2. Dataset with channels_first can be converted to channels_last and back
    3. Metrics work correctly with both data formats
    """
    # Create dataset with default channels_first format
    dataset = (
        TimeSeriesBuilder(
            n_timesteps=20, n_samples=2, random_state=42, data_format="channels_first"
        )
        .for_class(0)
        .add_signal(gaussian(), role="foundation")
        .for_class(1)
        .add_signal(gaussian(), role="foundation")
        .add_feature(level_change(amplitude=1.0), start_pct=0.4, end_pct=0.6)
        .build()
    )

    # Verify format is channels_first
    assert dataset["metadata"]["data_format"] == "channels_first"

    # Check tensor shapes - channels_first means [batch_size, channels, time_steps]
    assert dataset["X"].shape == (2, 1, 20)  # (n_samples, n_dimensions, n_timesteps)

    # Convert to channels_last
    channels_last_dataset = TimeSeriesBuilder.convert_data_format(
        dataset, "channels_last"
    )

    # Verify format is now channels_last
    assert channels_last_dataset["metadata"]["data_format"] == "channels_last"

    # Check tensor shapes - channels_last means [batch_size, time_steps, channels]
    assert channels_last_dataset["X"].shape == (
        2,
        20,
        1,
    )  # (n_samples, n_timesteps, n_dimensions)

    # Create attribution in channels_first format
    class1_indices = np.where(dataset["y"] == 1)[0]
    sample_idx = class1_indices[0]
    n_timesteps = dataset["metadata"]["n_timesteps"]
    n_dimensions = dataset["metadata"]["n_dimensions"]

    # Create a perfect attribution in channels_first format
    feature_key = next(k for k in dataset["feature_masks"].keys())
    mask = dataset["feature_masks"][feature_key][sample_idx]

    # Create attribution with shape [1, n_dimensions, n_timesteps] for channels_first
    attribution_channels_first = np.zeros((1, n_dimensions, n_timesteps), dtype=bool)
    attribution_channels_first[0, 0, :] = mask  # Perfect attribution

    # Calculate metrics with channels_first dataset
    precision_cf = precision_score(
        attribution_channels_first, dataset, sample_indices=[sample_idx], class_label=1
    )
    recall_cf = recall_score(
        attribution_channels_first, dataset, sample_indices=[sample_idx], class_label=1
    )

    # Metrics should show perfect attribution
    assert precision_cf == 1.0, "Precision should be 1.0 with channels_first format"
    assert recall_cf == 1.0, "Recall should be 1.0 with channels_first format"

    # Create a perfect attribution in channels_last format
    attribution_channels_last = np.zeros((1, n_timesteps, n_dimensions), dtype=bool)
    attribution_channels_last[0, :, 0] = mask  # Perfect attribution

    # Calculate metrics with channels_last dataset
    precision_cl = precision_score(
        attribution_channels_last,
        channels_last_dataset,
        sample_indices=[sample_idx],
        class_label=1,
    )
    recall_cl = recall_score(
        attribution_channels_last,
        channels_last_dataset,
        sample_indices=[sample_idx],
        class_label=1,
    )

    # Metrics should also show perfect attribution
    assert precision_cl == 1.0, "Precision should be 1.0 with channels_last format"
    assert recall_cl == 1.0, "Recall should be 1.0 with channels_last format"


def test_auc_roc_score():
    """Test area under ROC curve score for continuous attribution values.

    This test verifies that the AUC-ROC score correctly measures how well
    the attribution value rankings match the ground truth feature locations.
    """
    # Create a simple dataset with a clear feature at known position
    dataset = (
        TimeSeriesBuilder(n_timesteps=20, n_samples=1, random_state=42)
        .for_class(1)
        .add_signal(gaussian())
        .add_feature(level_change(amplitude=1.0), start_pct=0.4, end_pct=0.6)
        .build()
    )

    # Get the feature mask
    feature_key = next(k for k in dataset["feature_masks"].keys())
    mask = dataset["feature_masks"][feature_key][0]
    feature_start = np.argmax(mask)
    feature_end = len(mask) - np.argmax(mask[::-1])
    feature_center = feature_start + (feature_end - feature_start) // 2

    n_timesteps = dataset["metadata"]["n_timesteps"]
    n_dimensions = dataset["metadata"]["n_dimensions"]

    # Test cases:

    # Perfect ranking case: attribution values decrease with distance from feature
    perfect_attr = np.zeros((1, n_timesteps, n_dimensions))
    for i in range(n_timesteps):
        # Distance from center, normalized to 0-1 range
        distance = abs(i - feature_center) / (n_timesteps // 2)
        perfect_attr[0, i, 0] = max(0, 1 - distance)  # Linear decay from center

    # Random case: attribution values randomly distributed
    np.random.seed(42)  # For reproducibility
    random_attr = np.random.rand(1, n_timesteps, n_dimensions)

    # Inverse case: attribution values increase with distance from feature
    inverse_attr = np.zeros((1, n_timesteps, n_dimensions))
    for i in range(n_timesteps):
        # Distance from center, normalized to 0-1 range
        distance = abs(i - feature_center) / (n_timesteps // 2)
        inverse_attr[0, i, 0] = min(1, distance)  # Linear increase with distance

    perfect_score = auc_roc_score(perfect_attr, dataset, sample_indices=[0])
    random_score = auc_roc_score(random_attr, dataset, sample_indices=[0])
    inverse_score = auc_roc_score(inverse_attr, dataset, sample_indices=[0])

    # Perfect ranking should have AUC-ROC close to 1.0
    assert perfect_score > 0.90, (
        f"Perfect ranking should have AUC-ROC > 0.90, got {perfect_score}"
    )

    # Random ranking should have AUC-ROC around 0.5
    assert 0.4 <= random_score <= 0.7, (
        f"Random ranking should have AUC-ROC ≈ 0.5 (0.4-0.7 range), got {random_score}"
    )

    # Inverse ranking should have AUC-ROC close to 0.0
    assert inverse_score < 0.10, (
        f"Inverse ranking should have AUC-ROC < 0.10, got {inverse_score}"
    )

    # Test different average methods
    perfect_per_dim = auc_roc_score(
        perfect_attr, dataset, sample_indices=[0], average="per_dimension"
    )
    perfect_per_sample = auc_roc_score(
        perfect_attr, dataset, sample_indices=[0], average="per_sample"
    )
    perfect_per_sample_dim = auc_roc_score(
        perfect_attr, dataset, sample_indices=[0], average="per_sample_dimension"
    )

    # All average methods should maintain high score for perfect attribution
    if isinstance(perfect_per_dim, dict):
        # If perfect_per_dim is a dictionary (with a single key), get its value
        assert perfect_per_dim[0] > 0.90, (
            f"per_dimension AUC-ROC should be > 0.90 for perfect ranking, got {perfect_per_dim[0]}"
        )
    else:
        assert perfect_per_dim > 0.90, (
            f"per_dimension AUC-ROC should be > 0.90 for perfect ranking, got {perfect_per_dim}"
        )
    assert perfect_per_sample[0] > 0.90, (
        "per_sample AUC-ROC should be > 0.90 for perfect ranking"
    )
    assert perfect_per_sample_dim[(0, 0)] > 0.90, (
        "per_sample_dimension AUC-ROC should be > 0.90 for perfect ranking"
    )


def test_nac_score():
    """Test normalized attribution correlation (NAC) score.

    This test verifies that the NAC score correctly measures the correlation
    between attribution values and ground truth feature importance.
    """
    # Create a simple dataset with a clear feature at known position
    dataset = (
        TimeSeriesBuilder(n_timesteps=20, n_samples=1, random_state=42)
        .for_class(1)
        .add_signal(gaussian())
        .add_feature(level_change(amplitude=1.0), start_pct=0.4, end_pct=0.6)
        .build()
    )

    # Get the feature mask
    feature_key = next(k for k in dataset["feature_masks"].keys())
    mask = dataset["feature_masks"][feature_key][0]

    n_timesteps = dataset["metadata"]["n_timesteps"]
    n_dimensions = dataset["metadata"]["n_dimensions"]

    # Test cases:

    # Perfect case: high attribution values at feature locations
    perfect_attr = np.zeros((1, n_timesteps, n_dimensions))
    for i in range(n_timesteps):
        if mask[i]:
            perfect_attr[0, i, 0] = 1.0

    # Add small random noise to make standardization work properly
    np.random.seed(42)
    perfect_attr += np.random.normal(0, 0.01, perfect_attr.shape)

    # Random case: attribution values randomly distributed
    np.random.seed(42)
    random_attr = np.random.rand(1, n_timesteps, n_dimensions)

    # Inverse case: attribution values are low at feature locations
    inverse_attr = np.ones((1, n_timesteps, n_dimensions))
    for i in range(n_timesteps):
        if mask[i]:
            inverse_attr[0, i, 0] = 0.0

    # Add small random noise
    np.random.seed(42)
    inverse_attr += np.random.normal(0, 0.01, inverse_attr.shape)

    # Calculate NAC scores with ground_truth_only=True
    perfect_nac = nac_score(
        perfect_attr, dataset, sample_indices=[0], ground_truth_only=True
    )
    random_nac = nac_score(
        random_attr, dataset, sample_indices=[0], ground_truth_only=True
    )
    inverse_nac = nac_score(
        inverse_attr, dataset, sample_indices=[0], ground_truth_only=True
    )

    # Perfect attribution should have high positive NAC (standardized values are high at feature locations)
    assert perfect_nac > 0.5, (
        f"Perfect attribution should have NAC > 0.5, got {perfect_nac}"
    )

    # Random attribution should have NAC around 0
    assert -0.3 <= random_nac <= 0.4, (
        f"Random attribution should have NAC ≈ 0 (in range -0.3 to 0.4), got {random_nac}"
    )

    # Inverse attribution should have negative NAC (standardized values are low at feature locations)
    assert inverse_nac < -0.5, (
        f"Inverse attribution should have NAC < -0.5, got {inverse_nac}"
    )

    # Test with ground_truth_only=False (focusing on non-feature areas)
    perfect_nac_inv = nac_score(
        perfect_attr, dataset, sample_indices=[0], ground_truth_only=False
    )
    inverse_nac_inv = nac_score(
        inverse_attr, dataset, sample_indices=[0], ground_truth_only=False
    )

    # Now the results should be inverted since we're looking at non-feature regions
    assert perfect_nac_inv < 0, (
        "NAC should be negative for perfect attribution with ground_truth_only=False"
    )
    assert inverse_nac_inv > 0, (
        "NAC should be positive for inverse attribution with ground_truth_only=False"
    )

    perfect_per_dim = nac_score(
        perfect_attr, dataset, sample_indices=[0], average="per_dimension"
    )
    perfect_per_sample = nac_score(
        perfect_attr, dataset, sample_indices=[0], average="per_sample"
    )
    perfect_per_sample_dim = nac_score(
        perfect_attr, dataset, sample_indices=[0], average="per_sample_dimension"
    )

    # All average methods should maintain high score for perfect attribution
    if isinstance(perfect_per_dim, dict):
        assert perfect_per_dim[0] > 0.5, (
            f"per_dimension NAC should be > 0.5 for perfect attribution, got {perfect_per_dim[0]}"
        )
    else:
        assert perfect_per_dim > 0.5, (
            f"per_dimension NAC should be > 0.5 for perfect attribution, got {perfect_per_dim}"
        )
    assert perfect_per_sample[0] > 0.5, (
        "per_sample NAC should be > 0.5 for perfect attribution"
    )
    assert perfect_per_sample_dim[(0, 0)] > 0.5, (
        "per_sample_dimension NAC should be > 0.5 for perfect attribution"
    )

    # Test with multivariate data
    multivariate_dataset = (
        TimeSeriesBuilder(n_timesteps=20, n_samples=1, n_dimensions=2, random_state=42)
        .for_class(1)
        .add_signal(gaussian(), dim=[0, 1])
        .add_feature(level_change(amplitude=1.0), start_pct=0.3, end_pct=0.4, dim=[0])
        .add_feature(level_change(amplitude=1.0), start_pct=0.6, end_pct=0.7, dim=[1])
        .build()
    )

    # Create mixed attribution: good for dim0, random for dim1
    mixed_attr = np.zeros((1, n_timesteps, 2))

    # Get feature masks for dimension 0
    dim0_mask = np.zeros(n_timesteps, dtype=bool)
    for key, masks in multivariate_dataset["feature_masks"].items():
        if "_dim0" in key or ("_dim" not in key):
            dim0_mask |= masks[0]

    # Perfect attribution for dimension 0
    for i in range(n_timesteps):
        if dim0_mask[i]:
            mixed_attr[0, i, 0] = 1.0

    # Random attribution for dimension 1
    np.random.seed(42)
    mixed_attr[0, :, 1] = np.random.rand(n_timesteps)

    # Add noise for proper standardization
    np.random.seed(42)
    mixed_attr += np.random.normal(0, 0.01, mixed_attr.shape)

    # Calculate per-dimension NAC scores
    nac_per_dim = nac_score(mixed_attr, multivariate_dataset, average="per_dimension")

    # Dimension 0 should have high NAC, dimension 1 should have NAC close to 0
    assert nac_per_dim[0] > 0.5, (
        f"Dimension 0 should have high NAC, got {nac_per_dim[0]}"
    )
    assert -0.3 <= nac_per_dim[1] <= 0.3, (
        f"Dimension 1 should have NAC around 0, got {nac_per_dim[1]}"
    )


def test_auc_pr_score():
    """Test Area Under the Precision-Recall Curve (AUC-PR) score.

    This test verifies that the AUC-PR score correctly discriminates between
    good, random, and poor attribution rankings. The test evaluates perfect
    attribution (high score), random attribution (mid-range score), and inverse
    attribution (low score), as well as different averaging methods.
    """
    # Create a simple dataset with a clear feature at known position
    dataset = (
        TimeSeriesBuilder(n_timesteps=20, n_samples=1, random_state=42)
        .for_class(1)
        .add_signal(gaussian())
        .add_feature(level_change(amplitude=1.0), start_pct=0.4, end_pct=0.6)
        .build()
    )

    # Get the feature mask and find its center
    feature_key = next(k for k in dataset["feature_masks"].keys())
    mask = dataset["feature_masks"][feature_key][0]
    feature_start = np.argmax(mask)
    feature_end = len(mask) - np.argmax(mask[::-1])
    feature_center = feature_start + (feature_end - feature_start) // 2

    n_timesteps = dataset["metadata"]["n_timesteps"]
    n_dimensions = dataset["metadata"]["n_dimensions"]

    # Perfect ranking case: attribution values highest at feature location
    perfect_attr = np.zeros((1, n_timesteps, n_dimensions))
    for i in range(n_timesteps):
        # Distance from center, normalized to 0-1 range
        distance = abs(i - feature_center) / (n_timesteps // 2)
        perfect_attr[0, i, 0] = max(0, 1 - distance)  # Linear decay from center

    # Random case: attribution values randomly distributed
    np.random.seed(42)  # For reproducibility
    random_attr = np.random.rand(1, n_timesteps, n_dimensions)

    # Inverse case: attribution values lowest at feature location
    inverse_attr = np.zeros((1, n_timesteps, n_dimensions))
    for i in range(n_timesteps):
        # Distance from center, normalized to 0-1 range
        distance = abs(i - feature_center) / (n_timesteps // 2)
        inverse_attr[0, i, 0] = min(1, distance)  # Linear increase with distance

    # Calculate AUC-PR scores
    perfect_score = auc_pr_score(perfect_attr, dataset, sample_indices=[0])
    random_score = auc_pr_score(random_attr, dataset, sample_indices=[0])
    inverse_score = auc_pr_score(inverse_attr, dataset, sample_indices=[0])

    # Perfect ranking should have high AUC-PR (close to 1.0)
    assert perfect_score > 0.90, (
        f"Perfect ranking should have AUC-PR > 0.90, got {perfect_score}"
    )

    # Random ranking should have AUC-PR roughly around the positive class proportion
    assert 0.2 <= random_score <= 0.8, (
        f"Random ranking should have AUC-PR in a reasonable range, got {random_score}"
    )

    # Inverse ranking should have low AUC-PR (worse than random)
    assert inverse_score < random_score, (
        f"Inverse ranking should have AUC-PR < random ({random_score}), got {inverse_score}"
    )

    # Test different average methods
    perfect_per_dim = auc_pr_score(
        perfect_attr, dataset, sample_indices=[0], average="per_dimension"
    )
    perfect_per_sample = auc_pr_score(
        perfect_attr, dataset, sample_indices=[0], average="per_sample"
    )
    perfect_per_sample_dim = auc_pr_score(
        perfect_attr, dataset, sample_indices=[0], average="per_sample_dimension"
    )

    # All average methods should maintain high score for perfect attribution
    if isinstance(perfect_per_dim, dict):
        assert perfect_per_dim[0] > 0.90, (
            f"per_dimension AUC-PR should be > 0.90 for perfect ranking, got {perfect_per_dim[0]}"
        )
    else:
        assert perfect_per_dim > 0.90, (
            f"per_dimension AUC-PR should be > 0.90 for perfect ranking, got {perfect_per_dim}"
        )
    assert perfect_per_sample[0] > 0.90, (
        "per_sample AUC-PR should be > 0.90 for perfect ranking"
    )
    assert perfect_per_sample_dim[(0, 0)] > 0.90, (
        "per_sample_dimension AUC-PR should be > 0.90 for perfect ranking"
    )

    # Test with sparse features - create a dataset with a small feature
    sparse_dataset = (
        TimeSeriesBuilder(n_timesteps=20, n_samples=1, random_state=42)
        .for_class(1)
        .add_signal(gaussian())
        .add_feature(
            level_change(amplitude=1.0), start_pct=0.45, end_pct=0.5
        )  # Small feature
        .build()
    )

    # Create perfect attribution for the sparse feature
    sparse_mask = sparse_dataset["feature_masks"][
        next(k for k in sparse_dataset["feature_masks"].keys())
    ][0]
    sparse_perfect_attr = np.zeros((1, n_timesteps, n_dimensions))

    # Set high values at feature locations and small values elsewhere
    for i in range(n_timesteps):
        if sparse_mask[i]:
            sparse_perfect_attr[0, i, 0] = 1.0
        else:
            sparse_perfect_attr[0, i, 0] = 0.1

    # Calculate AUC-PR for sparse case
    sparse_auc_pr = auc_pr_score(
        sparse_perfect_attr, sparse_dataset, sample_indices=[0]
    )

    # Even with sparse features, perfect attribution should yield high AUC-PR
    assert sparse_auc_pr > 0.90, (
        f"For sparse features with good attribution, AUC-PR should be high, got {sparse_auc_pr}"
    )


def test_correlation_score():
    """Test correlation score between attribution values and ground truth features.

    This test verifies that the correlation_score function correctly measures the
    correlation between attribution values and ground truth features in different
    scenarios including perfect correlation, inverse correlation, and random values.
    It tests both absolute correlation (default behavior) and raw correlation with sign,
    as well as different feature sources (isolated components vs. aggregated values).
    """
    # Create a dataset with distinct feature patterns that have a different shape from the foundation
    dataset = (
        TimeSeriesBuilder(n_timesteps=50, n_samples=1, random_state=42)
        .for_class(1)
        .add_signal(gaussian(), role="foundation")
        .add_feature(level_change(amplitude=2.0), start_pct=0.4, end_pct=0.6)
        .build(return_components=True)
    )

    # Get feature mask for our test
    feature_key = next(k for k in dataset["feature_masks"].keys())
    mask = dataset["feature_masks"][feature_key][0]

    n_timesteps = dataset["metadata"]["n_timesteps"]
    n_dimensions = dataset["metadata"]["n_dimensions"]

    # Handle the different data formats appropriately
    data_format = dataset["metadata"].get("data_format", "channels_first")

    # Get both feature component values and aggregated values for comparison
    if data_format == "channels_first":
        # In channels_first, X shape is [n_samples, n_dimensions, n_timesteps]
        aggregated_values = dataset["X"][
            0, 0, :
        ].copy()  # First sample, first dimension
    else:  # channels_last
        # In channels_last, X shape is [n_samples, n_timesteps, n_dimensions]
        aggregated_values = dataset["X"][
            0, :, 0
        ].copy()  # First sample, first dimension

    # Extract the isolated feature values from the components
    feature_component = None
    for feature_name, values in dataset["components"][0].features.items():
        if "level_change" in feature_name:
            feature_component = values
            break

    # Ensure we found the level_change feature component
    assert feature_component is not None, (
        "Could not find level_change feature in components"
    )

    # Test cases:

    # 1. Perfect correlation with isolated feature: attribution values match feature component
    perfect_attr_isolated = np.zeros((1, n_timesteps, n_dimensions))
    for i in range(n_timesteps):
        if mask[i] and not np.isnan(feature_component[i]):
            perfect_attr_isolated[0, i, 0] = feature_component[i]

    # 2. Perfect correlation with aggregated: attribution values match aggregated values
    perfect_attr_aggregated = np.zeros((1, n_timesteps, n_dimensions))
    perfect_attr_aggregated[0, :, 0] = aggregated_values

    # 3. Inverse correlation: attribution values are inversely related to feature values
    inverse_attr = np.zeros((1, n_timesteps, n_dimensions))
    for i in range(n_timesteps):
        if mask[i] and not np.isnan(feature_component[i]):
            inverse_attr[0, i, 0] = -feature_component[i]

    # 4. Random attribution: attribution values have no relation to feature values
    np.random.seed(42)  # For reproducibility
    random_attr = np.random.rand(1, n_timesteps, n_dimensions)

    # Import correlation_score

    # Test 1: Correlation with isolated feature components (default)
    perfect_isolated_corr = correlation_score(
        perfect_attr_isolated, dataset, sample_indices=[0], feature_source="isolated"
    )

    inverse_isolated_corr = correlation_score(
        inverse_attr,
        dataset,
        sample_indices=[0],
        feature_source="isolated",
        absolute=False,
    )

    # Perfect correlation with isolated features should be close to 1.0
    assert perfect_isolated_corr > 0.99, (
        f"Perfect correlation with isolated features should be > 0.99, got {perfect_isolated_corr}"
    )

    # Inverse correlation with isolated features should be close to -1.0 (if absolute=False)
    assert inverse_isolated_corr < -0.99, (
        f"Inverse correlation with isolated features should be < -0.99, got {inverse_isolated_corr}"
    )

    # Test 2: Correlation with aggregated time series
    perfect_aggregated_corr = correlation_score(
        perfect_attr_aggregated,
        dataset,
        sample_indices=[0],
        feature_source="aggregated",
    )

    # Perfect correlation with aggregated values should be close to 1.0
    assert perfect_aggregated_corr > 0.99, (
        f"Perfect correlation with aggregated values should be > 0.99, got {perfect_aggregated_corr}"
    )

    # Test with random attributions
    random_isolated_corr = correlation_score(
        random_attr,
        dataset,
        sample_indices=[0],
        feature_source="isolated",
        absolute=True,
    )

    random_aggregated_corr = correlation_score(
        random_attr,
        dataset,
        sample_indices=[0],
        feature_source="aggregated",
        absolute=True,
    )

    # Random correlation should be relatively low (allowing for random variation)
    assert random_isolated_corr < 0.5, (
        f"Random correlation with isolated features should be < 0.5, got {random_isolated_corr}"
    )
    assert random_aggregated_corr < 0.5, (
        f"Random correlation with aggregated values should be < 0.5, got {random_aggregated_corr}"
    )

    # Test with multiple features (both sine and level change)
    multi_feature_dataset = (
        TimeSeriesBuilder(n_timesteps=50, n_samples=1, random_state=42)
        .for_class(1)
        .add_signal(gaussian(), role="foundation")
        .add_feature(seasonal(period=5), start_pct=0.2, end_pct=0.4)
        .add_feature(level_change(amplitude=2.0), start_pct=0.5, end_pct=0.7)
        .build(return_components=True)
    )

    # Create a perfect attribution that matches the combined feature components
    multi_perfect_attr = np.zeros((1, n_timesteps, n_dimensions))
    feature_sum = np.zeros(n_timesteps)

    # Combine all feature components
    for feat_name, feat_values in multi_feature_dataset["components"][
        0
    ].features.items():
        for i in range(n_timesteps):
            if not np.isnan(feat_values[i]):
                feature_sum[i] += feat_values[i]

    # Set attribution to match combined features
    multi_perfect_attr[0, :, 0] = feature_sum

    # Test correlation with isolated features
    multi_feature_corr = correlation_score(
        multi_perfect_attr,
        multi_feature_dataset,
        sample_indices=[0],
        feature_source="isolated",
    )

    # Correlation should be high
    assert multi_feature_corr > 0.95, (
        f"Correlation with multiple isolated features should be > 0.95, got {multi_feature_corr}"
    )

    # Test multivariate case
    multivariate_dataset = (
        TimeSeriesBuilder(n_timesteps=50, n_samples=1, n_dimensions=2, random_state=42)
        .for_class(1)
        .add_signal(gaussian(), dim=[0, 1])
        .add_feature(level_change(amplitude=1.0), start_pct=0.3, end_pct=0.4, dim=[0])
        .add_feature(level_change(amplitude=2.0), start_pct=0.6, end_pct=0.7, dim=[1])
        .build(return_components=True)
    )

    # Create a mixed attribution: perfect for dim0, inverse for dim1
    multi_attr = np.zeros((1, n_timesteps, 2))

    # For each dimension, get the feature component values
    for dim_idx in range(2):
        feature_values = np.zeros(n_timesteps)
        for feat_name, feat_vals in multivariate_dataset["components"][
            0
        ].features.items():
            if f"_dim{dim_idx}" in feat_name:
                for i in range(n_timesteps):
                    if not np.isnan(feat_vals[i]):
                        if dim_idx == 0:
                            # Match features for dimension 0
                            feature_values[i] += feat_vals[i]
                        else:
                            # Inverse features for dimension 1
                            feature_values[i] -= feat_vals[i]

        multi_attr[0, :, dim_idx] = feature_values

    # Test different averaging methods with absolute correlation
    per_dim_abs_corr = correlation_score(
        multi_attr,
        multivariate_dataset,
        sample_indices=[0],
        average="per_dimension",
        feature_source="isolated",
        absolute=True,
    )

    per_sample_dim_abs_corr = correlation_score(
        multi_attr,
        multivariate_dataset,
        sample_indices=[0],
        average="per_sample_dimension",
        feature_source="isolated",
        absolute=True,
    )

    # Both dimensions should have high absolute correlation
    assert per_dim_abs_corr[0] > 0.9, (
        f"Dimension 0 absolute correlation should be > 0.9, got {per_dim_abs_corr[0]}"
    )
    assert per_dim_abs_corr[1] > 0.9, (
        f"Dimension 1 absolute correlation should be > 0.9, got {per_dim_abs_corr[1]}"
    )
    assert per_sample_dim_abs_corr[(0, 0)] > 0.9, (
        f"Dimension 0 absolute correlation should be > 0.9, got {per_sample_dim_abs_corr[(0, 0)]}"
    )
    assert per_sample_dim_abs_corr[(0, 1)] > 0.9, (
        f"Dimension 1 absolute correlation should be > 0.9, got {per_sample_dim_abs_corr[(0, 1)]}"
    )

    # Test different averaging methods with raw correlation (with sign)
    per_dim_raw_corr = correlation_score(
        multi_attr,
        multivariate_dataset,
        sample_indices=[0],
        average="per_dimension",
        feature_source="isolated",
        absolute=False,
    )

    # Check raw correlations (dimension 0 positive, dimension 1 negative)
    assert per_dim_raw_corr[0] > 0.9, (
        f"Dimension 0 raw correlation should be > 0.9, got {per_dim_raw_corr[0]}"
    )
    assert per_dim_raw_corr[1] < -0.9, (
        f"Dimension 1 raw correlation should be < -0.9, got {per_dim_raw_corr[1]}"
    )
