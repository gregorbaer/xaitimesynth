"""Metrics for evaluating feature attributions against ground truth.

All metrics compare attribution values against binary ground truth masks from
TimeSeriesBuilder datasets. They support flexible aggregation via the `average`
parameter.

Input Format
------------
Attributions can be provided in several shapes:

    1D: (n_timesteps,)
        Single sample, single dimension. Automatically reshaped to (1, T, 1).

    2D: (n_timesteps, n_dims)
        Single sample, multiple dimensions. Reshaped to (1, T, D).

    3D: (n_samples, n_timesteps, n_dims)
        Multiple samples and dimensions. This is the canonical format.
        Also accepts (n_samples, n_dims, n_timesteps) - auto-detected and transposed.

The dataset parameter must be the dictionary returned by TimeSeriesBuilder.build(),
which contains 'metadata' and 'feature_masks' keys.

Example::

    import numpy as np
    from xaitimesynth import TimeSeriesBuilder, constant
    from xaitimesynth.metrics import relevance_mass_accuracy

    # Create dataset with 10 samples, 100 timesteps, 2 dimensions
    dataset = (
        TimeSeriesBuilder(n_timesteps=100, n_samples=10, n_dimensions=2)
        .for_class(0).add_signal(constant(0))
        .for_class(1).add_signal(constant(0))
        .add_feature(constant(1), start_pct=0.3, end_pct=0.5, dim=[0])
        .add_feature(constant(1), start_pct=0.6, end_pct=0.8, dim=[1])
        .build()
    )

    # Attributions must match: (n_samples, n_timesteps, n_dims)
    # Here: 5 samples to evaluate, 100 timesteps, 2 dimensions
    attributions = np.random.rand(5, 100, 2)

    # Evaluate first 5 class-1 samples
    class1_indices = np.where(dataset["y"] == 1)[0][:5].tolist()
    score = relevance_mass_accuracy(attributions, dataset, sample_indices=class1_indices)

    # Single sample, single dimension
    attr_1d = np.random.rand(100)  # auto-reshaped to (1, 100, 1)
    score = relevance_mass_accuracy(attr_1d, dataset, sample_indices=[class1_indices[0]])

Note: External XAI packages (Captum, SHAP, etc.) may return attributions in different
shapes. Check their documentation and reshape to (n_samples, n_timesteps, n_dims)
if needed. The auto-detection of (n_samples, n_dims, n_timesteps) handles some cases,
but explicit reshaping is safer.

Aggregation Modes
-----------------
    - 'macro': Mean across all samples and dimensions -> float
    - 'per_sample': Mean per sample across dimensions -> Dict[sample_idx, score]
    - 'per_dimension': Mean per dimension across samples -> Dict[dim_idx, score]
    - None: No aggregation, raw scores -> Dict[(sample_idx, dim_idx), score]
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np

# =============================================================================
# Helper Functions
# =============================================================================


def _prepare_inputs(
    attributions: np.ndarray,
    dataset: Dict,
    sample_indices: Optional[List[int]] = None,
    dim_indices: Optional[List[int]] = None,
) -> Tuple[np.ndarray, np.ndarray, List[int], List[int]]:
    """Validate inputs and prepare attributions and ground truth masks.

    Args:
        attributions: Attribution values, shape (n_samples, n_timesteps, n_dims).
        dataset: Dataset dictionary from TimeSeriesBuilder.build().
        sample_indices: Which samples to evaluate. Defaults to all.
        dim_indices: Which dimensions to evaluate. Defaults to all.

    Returns:
        Tuple of:
            - attributions: Validated attribution array (n_samples, n_timesteps, n_dims)
            - masks: Ground truth masks (n_samples, n_timesteps, n_dims)
            - sample_indices: List of sample indices
            - dim_indices: List of dimension indices

    Raises:
        ValueError: If inputs are invalid or incompatible.
    """
    # Validate dataset
    if "metadata" not in dataset:
        raise ValueError("Dataset missing 'metadata'. Use TimeSeriesBuilder.build().")
    if "feature_masks" not in dataset or not dataset["feature_masks"]:
        raise ValueError(
            "Dataset missing 'feature_masks'. Add features before building."
        )

    meta = dataset["metadata"]
    n_samples_data = meta["n_samples"]
    n_timesteps = meta["n_timesteps"]
    n_dims_data = meta["n_dimensions"]

    # Validate and reshape attributions to 3D
    attr = np.asarray(attributions)
    if attr.ndim == 1:
        if attr.shape[0] != n_timesteps:
            raise ValueError(
                f"1D attribution length {attr.shape[0]} != dataset timesteps {n_timesteps}"
            )
        attr = attr.reshape(1, n_timesteps, 1)
    elif attr.ndim == 2:
        if attr.shape[0] != n_timesteps:
            raise ValueError(
                f"2D attribution shape {attr.shape} doesn't match (n_timesteps, n_dims)"
            )
        attr = attr.reshape(1, attr.shape[0], attr.shape[1])
    elif attr.ndim == 3:
        # Detect format: (batch, time, dims) vs (batch, dims, time)
        if attr.shape[1] == n_timesteps:
            pass  # Already (batch, time, dims)
        elif attr.shape[2] == n_timesteps:
            attr = np.transpose(attr, (0, 2, 1))  # Convert to (batch, time, dims)
        else:
            raise ValueError(
                f"Attribution shape {attr.shape} doesn't match dataset "
                f"(n_timesteps={n_timesteps}, n_dims={n_dims_data})"
            )
    else:
        raise ValueError(f"Attribution must be 1D, 2D, or 3D, got {attr.ndim}D")

    n_samples_attr = attr.shape[0]
    n_dims_attr = attr.shape[2]

    # Set defaults for indices
    if sample_indices is None:
        sample_indices = list(range(min(n_samples_attr, n_samples_data)))
    if dim_indices is None:
        dim_indices = list(range(min(n_dims_attr, n_dims_data)))

    # Validate indices
    if max(sample_indices) >= n_samples_data:
        raise ValueError(
            f"sample_indices contains {max(sample_indices)}, "
            f"but dataset has {n_samples_data} samples"
        )
    if max(dim_indices) >= n_dims_attr:
        raise ValueError(
            f"dim_indices contains {max(dim_indices)}, "
            f"but attributions have {n_dims_attr} dimensions"
        )

    # Build combined ground truth masks per dimension
    # Masks in dataset have keys like "class_1_feature_constant_dim0"
    feature_masks = dataset["feature_masks"]
    n_samples_out = len(sample_indices)
    n_dims_out = len(dim_indices)

    masks = np.zeros((n_samples_out, n_timesteps, n_dims_out), dtype=bool)

    for mask_key, mask_array in feature_masks.items():
        # Parse dimension from key (e.g., "class_1_feature_constant_dim0")
        if "_dim" in mask_key:
            try:
                dim_idx = int(mask_key.split("_dim")[-1])
            except ValueError:
                continue
        else:
            dim_idx = 0  # Default to dim 0 if not specified

        if dim_idx not in dim_indices:
            continue

        j = dim_indices.index(dim_idx)
        for i, sample_idx in enumerate(sample_indices):
            masks[i, :, j] |= mask_array[sample_idx]

    # Slice attributions to match sample count
    attr = attr[:n_samples_out]

    return attr, masks, sample_indices, dim_indices


def _aggregate_results(
    results: Dict[Tuple[int, int], float],
    sample_indices: List[int],
    dim_indices: List[int],
    average: Optional[str],
) -> Union[float, Dict[int, float], Dict[Tuple[int, int], float]]:
    """Aggregate per-sample-dimension results according to averaging method.

    Args:
        results: Raw scores as {(sample_idx, dim_idx): score}
        sample_indices: List of sample indices used
        dim_indices: List of dimension indices used
        average: Aggregation method:
            - 'macro': Mean across all -> float
            - 'per_sample': Mean per sample -> Dict[sample_idx, float]
            - 'per_dimension': Mean per dimension -> Dict[dim_idx, float]
            - None: No aggregation -> Dict[(sample_idx, dim_idx), float]

    Returns:
        Aggregated results.
    """
    if not results:
        if average == "macro":
            return 0.0
        elif average == "per_sample":
            return {s: 0.0 for s in sample_indices}
        elif average == "per_dimension":
            return {d: 0.0 for d in dim_indices}
        else:
            return {}

    if average == "macro":
        return float(np.mean(list(results.values())))

    elif average == "per_sample":
        return {
            s: float(np.mean([results[(s, d)] for d in dim_indices]))
            for s in sample_indices
        }

    elif average == "per_dimension":
        return {
            d: float(np.mean([results[(s, d)] for s in sample_indices]))
            for d in dim_indices
        }

    elif average is None:
        return results

    else:
        raise ValueError(
            f"Invalid average='{average}'. "
            f"Choose from: 'macro', 'per_sample', 'per_dimension', or None"
        )


# =============================================================================
# Metrics
# =============================================================================


def relevance_mass_accuracy(
    attributions: np.ndarray,
    dataset: Dict,
    sample_indices: Optional[List[int]] = None,
    dim_indices: Optional[List[int]] = None,
    average: Optional[str] = "macro",
) -> Union[float, Dict[int, float], Dict[Tuple[int, int], float]]:
    """Ratio of attribution mass inside ground truth regions.

    Measures what fraction of total attribution "mass" falls within the
    ground truth mask. Higher is better (1.0 = all mass inside mask).

    Formula: sum(attr[mask]) / sum(attr)

    Args:
        attributions: Attribution values, shape (n_samples, n_timesteps, n_dims).
        dataset: Dataset dictionary from TimeSeriesBuilder.build().
        sample_indices: Which samples to evaluate. Defaults to all.
        dim_indices: Which dimensions to evaluate. Defaults to all.
        average: Aggregation method:
            - 'macro': Mean across all samples and dimensions -> float
            - 'per_sample': Mean per sample across dimensions -> Dict[sample_idx, score]
            - 'per_dimension': Mean per dimension across samples -> Dict[dim_idx, score]
            - None: No aggregation -> Dict[(sample_idx, dim_idx), score]

    Returns:
        Score(s) in range [0, 1]. Higher is better.

    References:
        Arras et al. (2022). CLEVR-XAI: A benchmark dataset for the ground truth
        evaluation of neural network explanations. Information Fusion, 81, 14-40.
    """
    attr, masks, sample_indices, dim_indices = _prepare_inputs(
        attributions, dataset, sample_indices, dim_indices
    )

    results = {}
    for i, s in enumerate(sample_indices):
        for j, d in enumerate(dim_indices):
            a, m = attr[i, :, j], masks[i, :, j]
            total = np.sum(a)
            results[(s, d)] = float(np.sum(a[m]) / total) if total > 0 else 0.0

    return _aggregate_results(results, sample_indices, dim_indices, average)


def relevance_rank_accuracy(
    attributions: np.ndarray,
    dataset: Dict,
    sample_indices: Optional[List[int]] = None,
    dim_indices: Optional[List[int]] = None,
    average: Optional[str] = "macro",
) -> Union[float, Dict[int, float], Dict[Tuple[int, int], float]]:
    """Fraction of top-K attributions that fall within ground truth.

    Selects K timesteps with highest attribution (where K = mask size),
    then measures what fraction of these are actually in the mask.
    Higher is better (1.0 = top-K perfectly matches mask).

    Args:
        attributions: Attribution values, shape (n_samples, n_timesteps, n_dims).
        dataset: Dataset dictionary from TimeSeriesBuilder.build().
        sample_indices: Which samples to evaluate. Defaults to all.
        dim_indices: Which dimensions to evaluate. Defaults to all.
        average: Aggregation method:
            - 'macro': Mean across all samples and dimensions -> float
            - 'per_sample': Mean per sample across dimensions -> Dict[sample_idx, score]
            - 'per_dimension': Mean per dimension across samples -> Dict[dim_idx, score]
            - None: No aggregation -> Dict[(sample_idx, dim_idx), score]

    Returns:
        Score(s) in range [0, 1]. Higher is better.

    References:
        Arras et al. (2022). CLEVR-XAI: A benchmark dataset for the ground truth
        evaluation of neural network explanations. Information Fusion, 81, 14-40.
    """
    attr, masks, sample_indices, dim_indices = _prepare_inputs(
        attributions, dataset, sample_indices, dim_indices
    )

    results = {}
    for i, s in enumerate(sample_indices):
        for j, d in enumerate(dim_indices):
            a, m = attr[i, :, j], masks[i, :, j]
            k = int(np.sum(m))
            if k == 0:
                results[(s, d)] = 0.0
            else:
                top_k = np.argpartition(-a, k - 1)[:k]
                results[(s, d)] = float(np.sum(m[top_k]) / k)

    return _aggregate_results(results, sample_indices, dim_indices, average)


def pointing_game(
    attributions: np.ndarray,
    dataset: Dict,
    sample_indices: Optional[List[int]] = None,
    dim_indices: Optional[List[int]] = None,
    average: Optional[str] = "macro",
) -> Union[float, Dict[int, float], Dict[Tuple[int, int], float]]:
    """Whether the maximum attribution falls within ground truth.

    A simple binary check: is the single highest-attributed timestep
    inside the ground truth mask? Returns 1.0 if yes, 0.0 if no.

    Args:
        attributions: Attribution values, shape (n_samples, n_timesteps, n_dims).
        dataset: Dataset dictionary from TimeSeriesBuilder.build().
        sample_indices: Which samples to evaluate. Defaults to all.
        dim_indices: Which dimensions to evaluate. Defaults to all.
        average: Aggregation method:
            - 'macro': Mean across all samples and dimensions -> float
            - 'per_sample': Mean per sample across dimensions -> Dict[sample_idx, score]
            - 'per_dimension': Mean per dimension across samples -> Dict[dim_idx, score]
            - None: No aggregation -> Dict[(sample_idx, dim_idx), score]

    Returns:
        Score(s) of 0.0 or 1.0 per sample-dimension, aggregated per `average`.

    References:
        Zhang et al. (2018). Top-down neural attention by excitation backprop.
        International Journal of Computer Vision, 126(10), 1084-1102.
    """
    attr, masks, sample_indices, dim_indices = _prepare_inputs(
        attributions, dataset, sample_indices, dim_indices
    )

    results = {}
    for i, s in enumerate(sample_indices):
        for j, d in enumerate(dim_indices):
            a, m = attr[i, :, j], masks[i, :, j]
            results[(s, d)] = 1.0 if m[np.argmax(a)] else 0.0

    return _aggregate_results(results, sample_indices, dim_indices, average)


def auc_roc_score(
    attributions: np.ndarray,
    dataset: Dict,
    sample_indices: Optional[List[int]] = None,
    dim_indices: Optional[List[int]] = None,
    average: Optional[str] = "macro",
    normalize: bool = False,
) -> Union[float, Dict[int, float], Dict[Tuple[int, int], float]]:
    """Area Under the ROC Curve for attribution ranking.

    Measures how well attributions discriminate between ground truth
    and non-ground-truth timesteps. Score of 0.5 = random, 1.0 = perfect.

    Args:
        attributions: Attribution values, shape (n_samples, n_timesteps, n_dims).
        dataset: Dataset dictionary from TimeSeriesBuilder.build().
        sample_indices: Which samples to evaluate. Defaults to all.
        dim_indices: Which dimensions to evaluate. Defaults to all.
        average: Aggregation method:
            - 'macro': Mean across all samples and dimensions -> float
            - 'per_sample': Mean per sample across dimensions -> Dict[sample_idx, score]
            - 'per_dimension': Mean per dimension across samples -> Dict[dim_idx, score]
            - None: No aggregation -> Dict[(sample_idx, dim_idx), score]
        normalize: If True, normalize to [-1, 1] range: (AUC - 0.5) / 0.5

    Returns:
        Score(s) in range [0, 1] (or [-1, 1] if normalized). Higher is better.
    """
    attr, masks, sample_indices, dim_indices = _prepare_inputs(
        attributions, dataset, sample_indices, dim_indices
    )

    results = {}
    for i, s in enumerate(sample_indices):
        for j, d in enumerate(dim_indices):
            a, m = attr[i, :, j], masks[i, :, j]

            if np.all(m) or not np.any(m):
                auc = 0.5
            else:
                # Compute AUC-ROC via trapezoidal rule
                thresholds = np.unique(a)
                thresholds = np.append(thresholds, thresholds.max() + 1)

                n_pos, n_neg = np.sum(m), np.sum(~m)
                tpr_list, fpr_list = [], []

                for thresh in sorted(thresholds, reverse=True):
                    pred = a >= thresh
                    tpr_list.append(np.sum(pred & m) / n_pos)
                    fpr_list.append(np.sum(pred & ~m) / n_neg)

                tpr = np.array(tpr_list)
                fpr = np.array(fpr_list)

                # Sort by FPR for proper integration
                order = np.argsort(fpr)
                auc = float(np.trapezoid(tpr[order], fpr[order]))

            if normalize:
                auc = (auc - 0.5) / 0.5

            results[(s, d)] = auc

    return _aggregate_results(results, sample_indices, dim_indices, average)


def auc_pr_score(
    attributions: np.ndarray,
    dataset: Dict,
    sample_indices: Optional[List[int]] = None,
    dim_indices: Optional[List[int]] = None,
    average: Optional[str] = "macro",
    normalize: bool = False,
) -> Union[float, Dict[int, float], Dict[Tuple[int, int], float]]:
    """Area Under the Precision-Recall Curve for attribution ranking.

    Measures precision-recall trade-off at different thresholds. Particularly
    useful for sparse ground truth (low prevalence). Baseline = prevalence.

    Args:
        attributions: Attribution values, shape (n_samples, n_timesteps, n_dims).
        dataset: Dataset dictionary from TimeSeriesBuilder.build().
        sample_indices: Which samples to evaluate. Defaults to all.
        dim_indices: Which dimensions to evaluate. Defaults to all.
        average: Aggregation method:
            - 'macro': Mean across all samples and dimensions -> float
            - 'per_sample': Mean per sample across dimensions -> Dict[sample_idx, score]
            - 'per_dimension': Mean per dimension across samples -> Dict[dim_idx, score]
            - None: No aggregation -> Dict[(sample_idx, dim_idx), score]
        normalize: If True, normalize relative to prevalence:
            (AUC - prevalence) / (1 - prevalence)

    Returns:
        Score(s) in range [0, 1]. Higher is better. Baseline = prevalence.
    """
    attr, masks, sample_indices, dim_indices = _prepare_inputs(
        attributions, dataset, sample_indices, dim_indices
    )

    results = {}
    for i, s in enumerate(sample_indices):
        for j, d in enumerate(dim_indices):
            a, m = attr[i, :, j], masks[i, :, j]

            n_pos = np.sum(m)
            prevalence = n_pos / m.size if m.size > 0 else 0.0

            if n_pos == m.size:
                auc = 1.0
            elif n_pos == 0:
                auc = 0.0
            else:
                thresholds = np.unique(a)
                thresholds = np.append(thresholds, thresholds.max() + 1)

                prec_list, rec_list = [], []
                for thresh in sorted(thresholds, reverse=True):
                    pred = a >= thresh
                    tp = np.sum(pred & m)
                    fp = np.sum(pred & ~m)
                    prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
                    rec = tp / n_pos
                    prec_list.append(prec)
                    rec_list.append(rec)

                prec = np.array(prec_list)
                rec = np.array(rec_list)

                # Sort by recall, keep max precision per recall
                order = np.argsort(rec)
                rec_sorted, prec_sorted = rec[order], prec[order]

                unique_rec, idx = np.unique(rec_sorted, return_index=True)
                unique_prec = np.array(
                    [np.max(prec_sorted[rec_sorted == r]) for r in unique_rec]
                )

                # Add (0, 1) anchor point
                if unique_rec[0] != 0:
                    unique_rec = np.concatenate([[0], unique_rec])
                    unique_prec = np.concatenate([[1.0], unique_prec])

                auc = float(np.trapezoid(unique_prec, unique_rec))

            if normalize:
                if prevalence >= 1.0:
                    auc = 0.0
                else:
                    auc = (auc - prevalence) / (1.0 - prevalence)

            results[(s, d)] = auc

    return _aggregate_results(results, sample_indices, dim_indices, average)


def nac_score(
    attributions: np.ndarray,
    dataset: Dict,
    sample_indices: Optional[List[int]] = None,
    dim_indices: Optional[List[int]] = None,
    average: Optional[str] = "macro",
    ground_truth_only: bool = True,
) -> Union[float, Dict[int, float], Dict[Tuple[int, int], float]]:
    """Normalized Attribution Correspondence (z-score at ground truth).

    Standardizes attributions (z-score), then takes mean at ground truth
    locations. Positive = attributions elevated at features. Negative = inverse.

    Also known as Normalised Scanpath Saliency (NSS) in eye-tracking literature.

    Args:
        attributions: Attribution values, shape (n_samples, n_timesteps, n_dims).
        dataset: Dataset dictionary from TimeSeriesBuilder.build().
        sample_indices: Which samples to evaluate. Defaults to all.
        dim_indices: Which dimensions to evaluate. Defaults to all.
        average: Aggregation method:
            - 'macro': Mean across all samples and dimensions -> float
            - 'per_sample': Mean per sample across dimensions -> Dict[sample_idx, score]
            - 'per_dimension': Mean per dimension across samples -> Dict[dim_idx, score]
            - None: No aggregation -> Dict[(sample_idx, dim_idx), score]
        ground_truth_only: If True, evaluate at mask locations. If False,
            evaluate at non-mask locations (useful for checking background).

    Returns:
        Score(s) with no fixed range. Positive = good, negative = inverted.

    References:
        Peters et al. (2005). Components of bottom-up gaze allocation in
        natural images. Vision Research, 45(18), 2397-2416.
    """
    attr, masks, sample_indices, dim_indices = _prepare_inputs(
        attributions, dataset, sample_indices, dim_indices
    )

    results = {}
    for i, s in enumerate(sample_indices):
        for j, d in enumerate(dim_indices):
            a, m = attr[i, :, j], masks[i, :, j]

            region = m if ground_truth_only else ~m

            if not np.any(region):
                results[(s, d)] = 0.0
            else:
                std = np.std(a)
                if std == 0:
                    results[(s, d)] = 0.0
                else:
                    z = (a - np.mean(a)) / std
                    results[(s, d)] = float(np.mean(z[region]))

    return _aggregate_results(results, sample_indices, dim_indices, average)


def mean_absolute_error(
    attributions: np.ndarray,
    dataset: Dict,
    sample_indices: Optional[List[int]] = None,
    dim_indices: Optional[List[int]] = None,
    average: Optional[str] = "macro",
) -> Union[float, Dict[int, float], Dict[Tuple[int, int], float]]:
    """Mean Absolute Error between attributions and ground truth mask.

    Treats the binary mask as target (1 at features, 0 elsewhere) and
    computes MAE. Lower is better (0.0 = perfect match).

    Note: Attributions should be normalized to [0, 1] for meaningful results.

    Args:
        attributions: Attribution values, shape (n_samples, n_timesteps, n_dims).
            Should be normalized to [0, 1].
        dataset: Dataset dictionary from TimeSeriesBuilder.build().
        sample_indices: Which samples to evaluate. Defaults to all.
        dim_indices: Which dimensions to evaluate. Defaults to all.
        average: Aggregation method:
            - 'macro': Mean across all samples and dimensions -> float
            - 'per_sample': Mean per sample across dimensions -> Dict[sample_idx, score]
            - 'per_dimension': Mean per dimension across samples -> Dict[dim_idx, score]
            - None: No aggregation -> Dict[(sample_idx, dim_idx), score]

    Returns:
        Score(s) >= 0. Lower is better.
    """
    attr, masks, sample_indices, dim_indices = _prepare_inputs(
        attributions, dataset, sample_indices, dim_indices
    )

    results = {}
    for i, s in enumerate(sample_indices):
        for j, d in enumerate(dim_indices):
            a, m = attr[i, :, j], masks[i, :, j].astype(float)
            results[(s, d)] = float(np.mean(np.abs(a - m)))

    return _aggregate_results(results, sample_indices, dim_indices, average)


def mean_squared_error(
    attributions: np.ndarray,
    dataset: Dict,
    sample_indices: Optional[List[int]] = None,
    dim_indices: Optional[List[int]] = None,
    average: Optional[str] = "macro",
) -> Union[float, Dict[int, float], Dict[Tuple[int, int], float]]:
    """Mean Squared Error between attributions and ground truth mask.

    Treats the binary mask as target (1 at features, 0 elsewhere) and
    computes MSE. Penalizes large errors more than MAE. Lower is better.

    Note: Attributions should be normalized to [0, 1] for meaningful results.

    Args:
        attributions: Attribution values, shape (n_samples, n_timesteps, n_dims).
            Should be normalized to [0, 1].
        dataset: Dataset dictionary from TimeSeriesBuilder.build().
        sample_indices: Which samples to evaluate. Defaults to all.
        dim_indices: Which dimensions to evaluate. Defaults to all.
        average: Aggregation method:
            - 'macro': Mean across all samples and dimensions -> float
            - 'per_sample': Mean per sample across dimensions -> Dict[sample_idx, score]
            - 'per_dimension': Mean per dimension across samples -> Dict[dim_idx, score]
            - None: No aggregation -> Dict[(sample_idx, dim_idx), score]

    Returns:
        Score(s) >= 0. Lower is better.
    """
    attr, masks, sample_indices, dim_indices = _prepare_inputs(
        attributions, dataset, sample_indices, dim_indices
    )

    results = {}
    for i, s in enumerate(sample_indices):
        for j, d in enumerate(dim_indices):
            a, m = attr[i, :, j], masks[i, :, j].astype(float)
            results[(s, d)] = float(np.mean((a - m) ** 2))

    return _aggregate_results(results, sample_indices, dim_indices, average)
