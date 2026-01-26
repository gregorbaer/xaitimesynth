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
    constant,
    gaussian,
)
from xaitimesynth.metrics import (
    auc_pr_score,
    auc_roc_score,
    nac_score,
    relevance_mass_accuracy,
    relevance_rank_accuracy,
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
        .add_feature(constant(value=1.0), start_pct=0.4, end_pct=0.6)
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
        .add_feature(constant(value=1.0), start_pct=0.3, end_pct=0.4, dim=[0])
        # Feature in dimension 1
        .add_feature(constant(value=1.0), start_pct=0.6, end_pct=0.7, dim=[1])
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
        .add_feature(constant(value=1.0), random_location=True, length_pct=0.2)
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
        .add_feature(constant(value=1.0), start_pct=0.4, end_pct=0.6)
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

    # Calculate normalized scores
    perfect_score_norm = auc_roc_score(
        perfect_attr, dataset, sample_indices=[0], normalize=True
    )
    random_score_norm = auc_roc_score(
        random_attr, dataset, sample_indices=[0], normalize=True
    )
    inverse_score_norm = auc_roc_score(
        inverse_attr, dataset, sample_indices=[0], normalize=True
    )

    # Assert raw scores
    assert perfect_score > 0.90, (
        f"Perfect ranking should have AUC-ROC > 0.90, got {perfect_score}"
    )
    assert 0.4 <= random_score <= 0.7, (
        f"Random ranking should have AUC-ROC ≈ 0.5 (0.4-0.7 range), got {random_score}"
    )
    assert inverse_score < 0.10, (
        f"Inverse ranking should have AUC-ROC < 0.10, got {inverse_score}"
    )

    # Assert normalized scores
    # Perfect normalized score should be close to (1.0 - 0.5) / 0.5 = 1.0
    assert perfect_score_norm > 0.80, (
        f"Normalized perfect score should be > 0.80, got {perfect_score_norm}"
    )
    # Random normalized score should be close to (0.5 - 0.5) / 0.5 = 0.0
    assert -0.2 <= random_score_norm <= 0.4, (
        f"Normalized random score should be ≈ 0.0 (-0.2 to 0.4 range), got {random_score_norm}"
    )
    # Inverse normalized score should be close to (0.0 - 0.5) / 0.5 = -1.0
    assert inverse_score_norm < -0.80, (
        f"Normalized inverse score should be < -0.80, got {inverse_score_norm}"
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
        .add_feature(constant(value=1.0), start_pct=0.4, end_pct=0.6)
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
        .add_feature(constant(value=1.0), start_pct=0.3, end_pct=0.4, dim=[0])
        .add_feature(constant(value=1.0), start_pct=0.6, end_pct=0.7, dim=[1])
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
        .add_feature(constant(value=1.0), start_pct=0.4, end_pct=0.6)
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
        inverse_attr[0, i, 0] = min(1, distance)  # Higher values further away

    # Calculate AUC-PR scores
    perfect_score = auc_pr_score(perfect_attr, dataset, sample_indices=[0])
    random_score = auc_pr_score(random_attr, dataset, sample_indices=[0])
    inverse_score = auc_pr_score(inverse_attr, dataset, sample_indices=[0])

    # Calculate prevalence for normalization
    prevalence = np.sum(mask) / mask.size

    # Calculate normalized scores
    perfect_score_norm = auc_pr_score(
        perfect_attr, dataset, sample_indices=[0], normalize=True
    )
    random_score_norm = auc_pr_score(
        random_attr, dataset, sample_indices=[0], normalize=True
    )
    inverse_score_norm = auc_pr_score(
        inverse_attr, dataset, sample_indices=[0], normalize=True
    )

    # Assert raw scores
    assert perfect_score > 0.90, (
        f"Perfect ranking should have AUC-PR > 0.90, got {perfect_score}"
    )
    # Random ranking should have AUC-PR roughly around the positive class proportion
    # Allow a wider range due to randomness
    assert (prevalence * 0.5) <= random_score <= (prevalence * 1.5 + 0.3), (
        f"Random ranking AUC-PR ({random_score}) should be around prevalence ({prevalence})"
    )
    # Inverse ranking should have low AUC-PR (worse than random)
    assert inverse_score < random_score, (
        f"Inverse ranking should have AUC-PR < random ({random_score}), got {inverse_score}"
    )

    # Assert normalized scores
    # Perfect normalized score should be close to (1.0 - prevalence) / (1.0 - prevalence) = 1.0
    assert perfect_score_norm > 0.80, (
        f"Normalized perfect score should be > 0.80, got {perfect_score_norm}"
    )
    # Random normalized score should be close to (prevalence - prevalence) / (1.0 - prevalence) = 0.0
    # Widen the range slightly to account for randomness
    assert -0.5 <= random_score_norm <= 0.5, (
        f"Normalized random score should be ≈ 0.0 (-0.5 to 0.5 range), got {random_score_norm}"
    )
    # Inverse normalized score should be significantly lower than perfect, potentially slightly positive
    assert inverse_score_norm < 0.1, (
        f"Normalized inverse score should be low (< 0.1), got {inverse_score_norm}"
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
        .add_feature(constant(value=1.0), start_pct=0.45, end_pct=0.5)  # Small feature
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


def test_relevance_mass_accuracy(
    simple_dataset, get_class1_sample_and_feature_mask, multivariate_dataset
):
    """Test the relevance_mass_accuracy metric in various scenarios.

    This test covers:
    1. Perfect attribution (all relevance inside the mask)
    2. Partial attribution (split relevance inside/outside)
    3. All relevance outside the mask
    4. Empty attribution (all zeros)
    5. Multivariate case with per-dimension and macro averaging
    """

    # --- Univariate, single sample ---
    sample_idx, mask = get_class1_sample_and_feature_mask(simple_dataset)
    n_timesteps = simple_dataset["metadata"]["n_timesteps"]
    n_dimensions = simple_dataset["metadata"]["n_dimensions"]

    # 1. Perfect attribution: all relevance inside the mask
    attr_perfect = np.zeros((1, n_timesteps, n_dimensions))
    attr_perfect[0, :, 0] = mask.astype(float)
    score = relevance_mass_accuracy(
        attr_perfect, simple_dataset, sample_indices=[sample_idx], class_label=1
    )
    assert score == 1.0, (
        "Perfect attribution should yield relevance_mass_accuracy of 1.0"
    )

    # 2. Partial attribution: half inside, half outside
    attr_partial = np.zeros((1, n_timesteps, n_dimensions))
    # Put 0.5 in mask, 0.5 outside mask
    attr_partial[0, :, 0] = 0.5 * mask + 0.5 * (~mask)
    score = relevance_mass_accuracy(
        attr_partial, simple_dataset, sample_indices=[sample_idx], class_label=1
    )
    expected = np.sum(0.5 * mask) / np.sum(attr_partial[0, :, 0])
    assert np.isclose(score, expected), (
        f"Partial attribution should yield score {expected}, got {score}"
    )

    # 3. All relevance outside the mask
    attr_outside = np.zeros((1, n_timesteps, n_dimensions))
    attr_outside[0, :, 0] = (~mask).astype(float)
    score = relevance_mass_accuracy(
        attr_outside, simple_dataset, sample_indices=[sample_idx], class_label=1
    )
    assert score == 0.0, "All relevance outside mask should yield score 0.0"

    # 4. Empty attribution (all zeros)
    attr_empty = np.zeros((1, n_timesteps, n_dimensions))
    score = relevance_mass_accuracy(
        attr_empty, simple_dataset, sample_indices=[sample_idx], class_label=1
    )
    assert score == 0.0, "Empty attribution should yield score 0.0"

    # 6. Complex scenario: non-uniform attribution values (mass test)
    attr_mass = np.zeros((1, n_timesteps, n_dimensions))
    # Assign 2.0 inside the mask, 0.5 outside
    attr_mass[0, :, 0] = 2.0 * mask + 0.5 * (~mask)
    expected_mass = np.sum(2.0 * mask) / np.sum(attr_mass[0, :, 0])
    score_mass = relevance_mass_accuracy(
        attr_mass, simple_dataset, sample_indices=[sample_idx], class_label=1
    )
    assert np.isclose(score_mass, expected_mass), (
        f"Complex mass scenario: expected {expected_mass}, got {score_mass} (should reflect mass of attributions)"
    )

    # --- Multivariate, per-dimension and macro averaging ---
    class1_indices = np.where(multivariate_dataset["y"] == 1)[0]
    sample_idx = class1_indices[0]
    n_timesteps = multivariate_dataset["metadata"]["n_timesteps"]
    n_dimensions = multivariate_dataset["metadata"]["n_dimensions"]

    # Get feature masks for each dimension
    feature_masks = {
        k: v[sample_idx] for k, v in multivariate_dataset["feature_masks"].items()
    }
    dim0_mask = np.zeros(n_timesteps, dtype=bool)
    dim1_mask = np.zeros(n_timesteps, dtype=bool)
    for k, m in feature_masks.items():
        if "_dim0" in k or ("_dim" not in k):
            dim0_mask |= m
        if "_dim1" in k:
            dim1_mask |= m

    # Perfect attribution for both dimensions
    attr_mv = np.zeros((1, n_timesteps, n_dimensions))
    attr_mv[0, :, 0] = dim0_mask.astype(float)
    attr_mv[0, :, 1] = dim1_mask.astype(float)
    score_macro = relevance_mass_accuracy(
        attr_mv, multivariate_dataset, sample_indices=[sample_idx], average="macro"
    )
    assert score_macro == 1.0, (
        "Perfect multivariate attribution should yield macro score 1.0"
    )
    score_per_dim = relevance_mass_accuracy(
        attr_mv,
        multivariate_dataset,
        sample_indices=[sample_idx],
        average="per_dimension",
    )
    assert all(np.isclose(s, 1.0) for s in score_per_dim), (
        "Perfect attribution should yield per-dimension scores of 1.0"
    )

    # Mixed: only dim0 is perfect, dim1 is all outside
    attr_mv[0, :, 1] = (~dim1_mask).astype(float)
    score_macro = relevance_mass_accuracy(
        attr_mv, multivariate_dataset, sample_indices=[sample_idx], average="macro"
    )
    assert np.isclose(score_macro, 0.5), (
        f"One perfect, one zero: macro should be 0.5, got {score_macro}"
    )
    score_per_dim = relevance_mass_accuracy(
        attr_mv,
        multivariate_dataset,
        sample_indices=[sample_idx],
        average="per_dimension",
    )
    assert np.isclose(score_per_dim[0], 1.0), (
        f"Dim0 should be 1.0, got {score_per_dim[0]}"
    )
    assert np.isclose(score_per_dim[1], 0.0), (
        f"Dim1 should be 0.0, got {score_per_dim[1]}"
    )


def test_relevance_rank_accuracy(
    simple_dataset, get_class1_sample_and_feature_mask, multivariate_dataset
):
    """Test the relevance_rank_accuracy metric in various scenarios.

    This test covers:
    1. Perfect attribution (top-K all inside mask)
    2. Partial attribution (top-K split between inside/outside)
    3. All top-K outside mask
    4. Empty attribution (all zeros)
    5. Multivariate case with per-dimension and macro averaging
    """
    # --- Univariate, single sample ---
    sample_idx, mask = get_class1_sample_and_feature_mask(simple_dataset)
    n_timesteps = simple_dataset["metadata"]["n_timesteps"]
    n_dimensions = simple_dataset["metadata"]["n_dimensions"]
    K = int(np.sum(mask))

    # 1. Perfect attribution: top-K all inside mask
    attr_perfect = np.zeros((1, n_timesteps, n_dimensions))
    attr_perfect[0, :, 0] = mask.astype(float)
    score = relevance_rank_accuracy(
        attr_perfect, simple_dataset, sample_indices=[sample_idx], class_label=1
    )
    assert score == 1.0, (
        "Perfect attribution should yield relevance_rank_accuracy of 1.0"
    )

    # 2. Partial attribution: top-K half inside, half outside
    attr_partial = np.zeros((1, n_timesteps, n_dimensions))
    # Assign high values to half of mask, rest to outside
    mask_indices = np.where(mask)[0]
    half = K // 2
    attr_partial[0, :, 0] = 0.1  # base value
    attr_partial[0, mask_indices[:half], 0] = 1.0  # high in-mask
    attr_partial[0, :half, 0] = 0.9  # high outside-mask (if outside)
    score = relevance_rank_accuracy(
        attr_partial, simple_dataset, sample_indices=[sample_idx], class_label=1
    )
    # The top-K will be the half in-mask and half outside
    expected = half / K
    assert np.isclose(score, expected), (
        f"Partial attribution should yield score {expected}, got {score}"
    )

    # 3. All top-K outside mask
    attr_outside = np.zeros((1, n_timesteps, n_dimensions))
    attr_outside[0, :, 0] = (~mask).astype(float)
    score = relevance_rank_accuracy(
        attr_outside, simple_dataset, sample_indices=[sample_idx], class_label=1
    )
    assert score == 0.0, "All top-K outside mask should yield score 0.0"

    # 4. Empty attribution (all zeros)
    attr_empty = np.zeros((1, n_timesteps, n_dimensions))
    score = relevance_rank_accuracy(
        attr_empty, simple_dataset, sample_indices=[sample_idx], class_label=1
    )
    assert score == 0.0, "Empty attribution should yield score 0.0"

    # 5. Multivariate, per-dimension and macro averaging
    class1_indices = np.where(multivariate_dataset["y"] == 1)[0]
    sample_idx = class1_indices[0]
    n_timesteps = multivariate_dataset["metadata"]["n_timesteps"]
    n_dimensions = multivariate_dataset["metadata"]["n_dimensions"]

    # Get feature masks for each dimension
    feature_masks = {
        k: v[sample_idx] for k, v in multivariate_dataset["feature_masks"].items()
    }
    dim0_mask = np.zeros(n_timesteps, dtype=bool)
    dim1_mask = np.zeros(n_timesteps, dtype=bool)
    for k, m in feature_masks.items():
        if "_dim0" in k or ("_dim" not in k):
            dim0_mask |= m
        if "_dim1" in k:
            dim1_mask |= m

    # Perfect attribution for both dimensions
    attr_mv = np.zeros((1, n_timesteps, n_dimensions))
    attr_mv[0, :, 0] = dim0_mask.astype(float)
    attr_mv[0, :, 1] = dim1_mask.astype(float)
    score_macro = relevance_rank_accuracy(
        attr_mv, multivariate_dataset, sample_indices=[sample_idx], average="macro"
    )
    assert score_macro == 1.0, (
        "Perfect multivariate attribution should yield macro score 1.0"
    )
    score_per_dim = relevance_rank_accuracy(
        attr_mv,
        multivariate_dataset,
        sample_indices=[sample_idx],
        average="per_dimension",
    )
    assert all(np.isclose(s, 1.0) for s in score_per_dim), (
        "Perfect attribution should yield per-dimension scores of 1.0"
    )

    # Mixed: only dim0 is perfect, dim1 is all outside
    attr_mv[0, :, 1] = (~dim1_mask).astype(float)
    score_macro = relevance_rank_accuracy(
        attr_mv, multivariate_dataset, sample_indices=[sample_idx], average="macro"
    )
    assert np.isclose(score_macro, 0.5), (
        f"One perfect, one zero: macro should be 0.5, got {score_macro}"
    )
    score_per_dim = relevance_rank_accuracy(
        attr_mv,
        multivariate_dataset,
        sample_indices=[sample_idx],
        average="per_dimension",
    )
    assert np.isclose(score_per_dim[0], 1.0), (
        f"Dim0 should be 1.0, got {score_per_dim[0]}"
    )
    assert np.isclose(score_per_dim[1], 0.0), (
        f"Dim1 should be 0.0, got {score_per_dim[1]}"
    )
