"""Tests for metrics module.

Structure:
1. One verification test per metric (verifies expected values and return types)
2. Multivariate functional tests
"""

import numpy as np
import pytest

from xaitimesynth import TimeSeriesBuilder, constant
from xaitimesynth.metrics import (
    auc_pr_score,
    auc_roc_score,
    nac_score,
    relevance_mass_accuracy,
    relevance_rank_accuracy,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def simple_dataset():
    """Dataset with feature at timesteps 4-7 (40-80%) out of 10.

    - n_timesteps=10, n_samples=2
    - Class 0: constant signal, no features
    - Class 1: constant signal + feature at indices 4,5,6,7 (4 timesteps)
    - Prevalence = 4/10 = 0.4
    """
    return (
        TimeSeriesBuilder(n_timesteps=10, n_samples=2, random_state=42)
        .for_class(0)
        .add_signal(constant(value=0.0), role="foundation")
        .for_class(1)
        .add_signal(constant(value=0.0), role="foundation")
        .add_feature(constant(value=1.0), start_pct=0.4, end_pct=0.8)
        .build()
    )


@pytest.fixture(scope="module")
def multivariate_dataset():
    """Dataset with features in different dimensions.

    - n_timesteps=10, n_dimensions=2
    - Dim 0: feature at 20-40% (indices 2,3)
    - Dim 1: feature at 60-80% (indices 6,7)
    """
    return (
        TimeSeriesBuilder(n_timesteps=10, n_samples=2, n_dimensions=2, random_state=42)
        .for_class(0)
        .add_signal(constant(value=0.0), dim=[0, 1])
        .for_class(1)
        .add_signal(constant(value=0.0), dim=[0, 1])
        .add_feature(constant(value=1.0), start_pct=0.2, end_pct=0.4, dim=[0])
        .add_feature(constant(value=1.0), start_pct=0.6, end_pct=0.8, dim=[1])
        .build()
    )


@pytest.fixture
def sample_and_mask(simple_dataset):
    """Extract first class-1 sample index and its feature mask."""
    sample_idx = np.where(simple_dataset["y"] == 1)[0][0]
    mask = next(iter(simple_dataset["feature_masks"].values()))[sample_idx]
    return sample_idx, mask


# =============================================================================
# Verification tests per metric
# =============================================================================


def test_rma_verification(simple_dataset, sample_and_mask):
    """Verification tests for relevance_mass_accuracy.

    Formula: sum(attr[mask]) / sum(attr)
    """
    sample_idx, mask = sample_and_mask
    n_timesteps = simple_dataset["metadata"]["n_timesteps"]

    # Perfect: all mass inside mask -> 1.0
    attr = np.zeros((1, n_timesteps, 1))
    attr[0, mask, 0] = 1.0
    score = relevance_mass_accuracy(attr, simple_dataset, sample_indices=[sample_idx])
    assert score == 1.0, "All mass inside mask should give 1.0"

    # Zero: all mass outside mask -> 0.0
    attr = np.zeros((1, n_timesteps, 1))
    attr[0, ~mask, 0] = 1.0
    score = relevance_mass_accuracy(attr, simple_dataset, sample_indices=[sample_idx])
    assert score == 0.0, "All mass outside mask should give 0.0"

    # Half: equal mass inside and outside -> 0.5
    n_inside = np.sum(mask)
    n_outside = n_timesteps - n_inside
    attr = np.zeros((1, n_timesteps, 1))
    attr[0, mask, 0] = 1.0 / n_inside
    attr[0, ~mask, 0] = 1.0 / n_outside
    score = relevance_mass_accuracy(attr, simple_dataset, sample_indices=[sample_idx])
    assert np.isclose(score, 0.5), "Equal mass inside/outside should give 0.5"

    # Empty attribution -> 0.0
    attr = np.zeros((1, n_timesteps, 1))
    score = relevance_mass_accuracy(attr, simple_dataset, sample_indices=[sample_idx])
    assert score == 0.0, "Empty attribution should give 0.0"

    # Return types
    attr = np.ones((1, n_timesteps, 1))
    assert isinstance(
        relevance_mass_accuracy(
            attr, simple_dataset, sample_indices=[sample_idx], average="macro"
        ),
        float,
    ), "macro should return float"
    assert isinstance(
        relevance_mass_accuracy(
            attr, simple_dataset, sample_indices=[sample_idx], average="per_sample"
        ),
        dict,
    ), "per_sample should return dict"


def test_rra_verification(simple_dataset, sample_and_mask):
    """Verification tests for relevance_rank_accuracy.

    Formula: fraction of top-K attributions in mask (K = mask size)
    """
    sample_idx, mask = sample_and_mask
    n_timesteps = simple_dataset["metadata"]["n_timesteps"]

    # Perfect: top-K all inside mask -> 1.0
    attr = np.zeros((1, n_timesteps, 1))
    attr[0, mask, 0] = 1.0
    score = relevance_rank_accuracy(attr, simple_dataset, sample_indices=[sample_idx])
    assert score == 1.0, "Top-K all inside mask should give 1.0"

    # Zero: top-K all outside mask -> 0.0
    attr = np.zeros((1, n_timesteps, 1))
    attr[0, ~mask, 0] = 1.0
    score = relevance_rank_accuracy(attr, simple_dataset, sample_indices=[sample_idx])
    assert score == 0.0, "Top-K all outside mask should give 0.0"

    # Empty attribution -> valid range (arbitrary top-K selection)
    attr = np.zeros((1, n_timesteps, 1))
    score = relevance_rank_accuracy(attr, simple_dataset, sample_indices=[sample_idx])
    assert 0.0 <= score <= 1.0, "Empty attribution should give valid score"

    # Return types
    attr = np.ones((1, n_timesteps, 1))
    assert isinstance(
        relevance_rank_accuracy(
            attr, simple_dataset, sample_indices=[sample_idx], average="macro"
        ),
        float,
    ), "macro should return float"
    assert isinstance(
        relevance_rank_accuracy(
            attr, simple_dataset, sample_indices=[sample_idx], average="per_sample"
        ),
        dict,
    ), "per_sample should return dict"


def test_auc_roc_verification(simple_dataset, sample_and_mask):
    """Verification tests for auc_roc_score.

    AUC-ROC: 1.0 = perfect separation, 0.5 = random, 0.0 = inverse
    Normalized: (AUC - 0.5) / 0.5
    """
    sample_idx, mask = sample_and_mask
    n_timesteps = simple_dataset["metadata"]["n_timesteps"]

    # Perfect separation -> 1.0
    attr = np.zeros((1, n_timesteps, 1))
    attr[0, mask, 0] = 1.0
    attr[0, ~mask, 0] = 0.0
    score = auc_roc_score(attr, simple_dataset, sample_indices=[sample_idx])
    assert score == 1.0, "Perfect separation should give 1.0"

    # Inverse separation -> 0.0
    attr = np.zeros((1, n_timesteps, 1))
    attr[0, mask, 0] = 0.0
    attr[0, ~mask, 0] = 1.0
    score = auc_roc_score(attr, simple_dataset, sample_indices=[sample_idx])
    assert score == 0.0, "Inverse separation should give 0.0"

    # Normalized perfect -> 1.0
    attr = np.zeros((1, n_timesteps, 1))
    attr[0, mask, 0] = 1.0
    score = auc_roc_score(
        attr, simple_dataset, sample_indices=[sample_idx], normalize=True
    )
    assert score == 1.0, "Normalized perfect should give 1.0"

    # Normalized inverse -> -1.0
    attr = np.zeros((1, n_timesteps, 1))
    attr[0, ~mask, 0] = 1.0
    score = auc_roc_score(
        attr, simple_dataset, sample_indices=[sample_idx], normalize=True
    )
    assert score == -1.0, "Normalized inverse should give -1.0"

    # Return types
    attr = np.ones((1, n_timesteps, 1))
    assert isinstance(
        auc_roc_score(
            attr, simple_dataset, sample_indices=[sample_idx], average="macro"
        ),
        float,
    ), "macro should return float"
    assert isinstance(
        auc_roc_score(
            attr, simple_dataset, sample_indices=[sample_idx], average="per_sample"
        ),
        dict,
    ), "per_sample should return dict"


def test_auc_pr_verification(simple_dataset, sample_and_mask):
    """Verification tests for auc_pr_score.

    AUC-PR: 1.0 = perfect ranking, baseline = prevalence
    Normalized: (AUC - prevalence) / (1 - prevalence)
    """
    sample_idx, mask = sample_and_mask
    n_timesteps = simple_dataset["metadata"]["n_timesteps"]

    # Perfect ranking -> 1.0
    attr = np.zeros((1, n_timesteps, 1))
    attr[0, mask, 0] = 1.0
    score = auc_pr_score(attr, simple_dataset, sample_indices=[sample_idx])
    assert score == 1.0, "Perfect ranking should give 1.0"

    # Normalized perfect -> 1.0
    attr = np.zeros((1, n_timesteps, 1))
    attr[0, mask, 0] = 1.0
    score = auc_pr_score(
        attr, simple_dataset, sample_indices=[sample_idx], normalize=True
    )
    assert score == 1.0, "Normalized perfect should give 1.0"

    # Inverse should be strictly below perfect (but not necessarily 0.0)
    perfect_attr = np.zeros((1, n_timesteps, 1))
    perfect_attr[0, mask, 0] = 1.0
    inverse_attr = np.zeros((1, n_timesteps, 1))
    inverse_attr[0, ~mask, 0] = 1.0

    perfect_score = auc_pr_score(
        perfect_attr, simple_dataset, sample_indices=[sample_idx]
    )
    inverse_score = auc_pr_score(
        inverse_attr, simple_dataset, sample_indices=[sample_idx]
    )
    assert inverse_score < perfect_score, "Inverse should be below perfect"
    assert 0.0 < inverse_score < 1.0, "Inverse should be in valid range"

    # Return types
    attr = np.ones((1, n_timesteps, 1))
    assert isinstance(
        auc_pr_score(
            attr, simple_dataset, sample_indices=[sample_idx], average="macro"
        ),
        float,
    ), "macro should return float"
    assert isinstance(
        auc_pr_score(
            attr, simple_dataset, sample_indices=[sample_idx], average="per_sample"
        ),
        dict,
    ), "per_sample should return dict"


def test_nac_verification(simple_dataset, sample_and_mask):
    """Verification tests for nac_score.

    NAC: mean of z-scored attribution at feature locations
    Positive = good attribution, Negative = inverse attribution
    """
    sample_idx, mask = sample_and_mask
    n_timesteps = simple_dataset["metadata"]["n_timesteps"]

    # Good attribution (high at feature) -> positive
    attr = np.zeros((1, n_timesteps, 1))
    attr[0, mask, 0] = 1.0
    attr[0, ~mask, 0] = 0.0
    score = nac_score(attr, simple_dataset, sample_indices=[sample_idx])
    assert score > 0, "High attribution at features should give positive NAC"

    # Inverse attribution (low at feature) -> negative
    attr = np.zeros((1, n_timesteps, 1))
    attr[0, mask, 0] = 0.0
    attr[0, ~mask, 0] = 1.0
    score = nac_score(attr, simple_dataset, sample_indices=[sample_idx])
    assert score < 0, "Low attribution at features should give negative NAC"

    # ground_truth_only=False measures non-feature regions
    attr = np.zeros((1, n_timesteps, 1))
    attr[0, mask, 0] = 1.0
    attr[0, ~mask, 0] = 0.0
    score = nac_score(
        attr, simple_dataset, sample_indices=[sample_idx], ground_truth_only=False
    )
    assert score < 0, "Good attribution with ground_truth_only=False should be negative"

    # Return types
    attr = np.ones((1, n_timesteps, 1))
    assert isinstance(
        nac_score(attr, simple_dataset, sample_indices=[sample_idx], average="macro"),
        float,
    ), "macro should return float"
    assert isinstance(
        nac_score(
            attr, simple_dataset, sample_indices=[sample_idx], average="per_sample"
        ),
        dict,
    ), "per_sample should return dict"


# =============================================================================
# Multivariate tests
# =============================================================================


def test_multivariate_per_dimension(multivariate_dataset):
    """Metrics correctly compute per-dimension scores for multivariate data."""
    sample_idx = np.where(multivariate_dataset["y"] == 1)[0][0]
    n_timesteps = multivariate_dataset["metadata"]["n_timesteps"]

    # Get masks for each dimension
    dim0_mask = np.zeros(n_timesteps, dtype=bool)
    dim1_mask = np.zeros(n_timesteps, dtype=bool)
    for key, masks in multivariate_dataset["feature_masks"].items():
        if "_dim0" in key or "_dim" not in key:
            dim0_mask |= masks[sample_idx]
        if "_dim1" in key:
            dim1_mask |= masks[sample_idx]

    # Perfect for dim0, inverse for dim1
    attr = np.zeros((1, n_timesteps, 2))
    attr[0, dim0_mask, 0] = 1.0
    attr[0, ~dim1_mask, 1] = 1.0

    # RMA per-dimension: dim0=1.0, dim1=0.0
    scores = relevance_mass_accuracy(
        attr, multivariate_dataset, sample_indices=[sample_idx], average="per_dimension"
    )
    assert np.isclose(scores[0], 1.0), "Dim0 RMA should be 1.0"
    assert np.isclose(scores[1], 0.0), "Dim1 RMA should be 0.0"

    # Macro should be average: (1.0 + 0.0) / 2 = 0.5
    macro_score = relevance_mass_accuracy(
        attr, multivariate_dataset, sample_indices=[sample_idx], average="macro"
    )
    assert np.isclose(macro_score, 0.5), "Macro should average to 0.5"

    # per_sample_dimension returns dict with (sample, dim) keys
    scores = relevance_mass_accuracy(
        attr,
        multivariate_dataset,
        sample_indices=[sample_idx],
        average="per_sample_dimension",
    )
    assert isinstance(scores, dict), "per_sample_dimension should return dict"
    assert (sample_idx, 0) in scores, "Should have key for (sample, dim0)"
    assert (sample_idx, 1) in scores, "Should have key for (sample, dim1)"
