"""Metrics for evaluating feature attributions against ground truth."""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np


def _binarize_attributions(attributions: np.ndarray, threshold: float) -> np.ndarray:
    """Binarize attribution values using a threshold.

    Args:
        attributions (np.ndarray): Feature attribution values.
        threshold (float): Threshold for binarization.

    Returns:
        np.ndarray: Binarized attribution values (boolean array).
    """
    if np.issubdtype(attributions.dtype, np.bool_):
        return attributions  # Already boolean

    return attributions >= threshold


def _extract_masks_for_class(
    dataset: Dict,
    class_label: Optional[int] = None,
    dim_indices: Optional[List[int]] = None,
) -> Dict[str, np.ndarray]:
    """Extract feature masks for the specified class and dimensions.

    Args:
        dataset (Dict): Dataset dictionary returned by TimeSeriesBuilder.build().
        class_label (Optional[int]): Class label to extract masks for. If None, uses
            masks from all classes.
        dim_indices (Optional[List[int]]): Dimension indices to extract masks for.
            If None, uses all dimensions.

    Returns:
        Dict[str, np.ndarray]: Dictionary of feature masks filtered by class and dimension.
    """
    if "feature_masks" not in dataset:
        raise ValueError(
            "Dataset does not contain feature masks. Cannot calculate metrics."
        )

    feature_masks = dataset["feature_masks"]
    filtered_masks = {}

    for key, mask in feature_masks.items():
        # Filter by class if specified
        if class_label is not None:
            class_prefix = f"class_{class_label}_"
            if not key.startswith(class_prefix):
                continue

        # Filter by dimension if specified
        if dim_indices is not None:
            # Extract dimension from the mask key (format: class_X_feature_Y_type_dimZ)
            if "_dim" not in key:
                # Handle the case where no dimension is specified (assume dim0)
                if 0 not in dim_indices:
                    continue
            else:
                # Extract the dimension number - handle potential parsing issues
                try:
                    dim_part = key.split("_dim")[-1]
                    dim = int(dim_part)
                    if dim not in dim_indices:
                        continue
                except (ValueError, IndexError):
                    # If we can't parse the dimension, skip this mask
                    continue

        filtered_masks[key] = mask

    return filtered_masks


def _combine_masks(masks: Dict[str, np.ndarray]) -> np.ndarray:
    """Combine multiple feature masks into a single mask.

    If a timestep is part of any feature, it will be True in the combined mask.

    Args:
        masks (Dict[str, np.ndarray]): Dictionary of feature masks.

    Returns:
        np.ndarray: Combined boolean mask.
    """
    if not masks:
        return np.array([])

    # All masks should have the same shape (n_samples, n_timesteps)
    mask_shape = next(iter(masks.values())).shape
    combined = np.zeros(mask_shape, dtype=bool)

    # Combine masks with logical OR
    for mask in masks.values():
        combined = np.logical_or(combined, mask)

    return combined


def _validate_and_prepare_inputs(
    attributions: np.ndarray,
    dataset: Dict,
    sample_indices: Optional[List[int]] = None,
    dim_indices: Optional[List[int]] = None,
    threshold: Optional[float] = None,
    class_label: Optional[int] = None,
) -> Tuple[np.ndarray, Dict[int, np.ndarray], List[int], List[int]]:
    """Validate and prepare inputs for precision and recall calculations.

    Args:
        attributions (np.ndarray): Feature attribution values.
        dataset (Dict): Dataset dictionary returned by TimeSeriesBuilder.build().
        sample_indices (Optional[List[int]]): Sample indices to include.
        dim_indices (Optional[List[int]]): Dimension indices to include.
        threshold (Optional[float]): Threshold for binarizing attribution values.
        class_label (Optional[int]): Class label to calculate metrics for.

    Returns:
        Tuple containing:
            - np.ndarray: Binarized attributions
            - Dict[int, np.ndarray]: Ground truth masks by dimension
            - List[int]: Sample indices
            - List[int]: Dimension indices

    Raises:
        ValueError: If inputs have incompatible shapes or dimensions.
    """
    # Verify dataset has required metadata
    if "metadata" not in dataset:
        raise ValueError("Dataset does not contain metadata. Cannot calculate metrics.")

    if "feature_masks" not in dataset or not dataset["feature_masks"]:
        raise ValueError(
            "Dataset does not contain feature masks. Cannot calculate metrics."
        )

    # Get dataset dimensions from metadata
    n_samples = dataset["metadata"]["n_samples"]
    n_timesteps = dataset["metadata"]["n_timesteps"]
    n_dimensions = dataset["metadata"]["n_dimensions"]

    # Validate attribution shape
    attribution_shape = attributions.shape

    # Case 1: attributions is (n_timesteps,) - single sample, single dimension
    if len(attribution_shape) == 1:
        if attribution_shape[0] != n_timesteps:
            raise ValueError(
                f"Attribution length ({attribution_shape[0]}) does not match dataset timesteps ({n_timesteps}). "
                f"The time dimension of attributions must match the dataset."
            )
        # Reshape to (1, n_timesteps, 1) for consistent processing
        attributions = attributions.reshape(1, n_timesteps, 1)

        # When given 1D attributions, we need explicit sample_indices
        if sample_indices is None:
            raise ValueError(
                "When providing 1D attributions (single sample, single dimension), "
                "you must explicitly specify sample_indices to identify which sample "
                "to compare against."
            )
        if dim_indices is None:
            dim_indices = [0]  # Default to dimension 0 for 1D attributions

    # Case 2: attributions is (n_timesteps, n_dimensions) - single sample, multiple dimensions
    elif len(attribution_shape) == 2:
        if attribution_shape[0] != n_timesteps:
            raise ValueError(
                f"Attribution first dimension ({attribution_shape[0]}) does not match dataset timesteps ({n_timesteps}). "
                f"The time dimension of attributions must match the dataset."
            )
        # Reshape to (1, n_timesteps, n_dimensions) for consistent processing
        attributions = attributions.reshape(1, n_timesteps, attribution_shape[1])

        # When given 2D attributions, we need explicit sample_indices
        if sample_indices is None:
            raise ValueError(
                "When providing 2D attributions (single sample, multiple dimensions), "
                "you must explicitly specify sample_indices to identify which sample "
                "to compare against."
            )

    # Case 3: attributions is (n_samples, n_timesteps, n_dimensions) - already in correct format
    elif len(attribution_shape) == 3:
        if attribution_shape[1] != n_timesteps:
            raise ValueError(
                f"Attribution middle dimension ({attribution_shape[1]}) does not match dataset timesteps ({n_timesteps}). "
                f"The time dimension of attributions must match the dataset."
            )

        # For 3D attributions, we can use default sample_indices
        if sample_indices is None:
            # If attributions has fewer samples than the dataset, only use that many
            if attribution_shape[0] < n_samples:
                sample_indices = list(range(attribution_shape[0]))
                print(
                    f"Warning: Using only the first {attribution_shape[0]} samples from the dataset "
                    f"to match attribution array shape."
                )
            else:
                sample_indices = list(range(n_samples))

    else:
        raise ValueError(
            f"Unsupported attribution shape: {attribution_shape}. "
            f"Expected 1D (n_timesteps,), 2D (n_timesteps, n_dimensions), or "
            f"3D (n_samples, n_timesteps, n_dimensions)."
        )

    # Set default dimension indices if not provided
    if dim_indices is None:
        # Use dimensions available in the attribution array
        dim_indices = list(range(min(n_dimensions, attributions.shape[2])))

    # Validate that sample_indices are within range
    if max(sample_indices) >= n_samples:
        raise ValueError(
            f"Sample index {max(sample_indices)} out of range (0 to {n_samples - 1}). "
            f"Please provide valid sample indices within the dataset range."
        )

    # Validate that dim_indices are within range of the attributions array
    if attributions.shape[2] <= max(dim_indices):
        raise ValueError(
            f"Dimension index {max(dim_indices)} out of range for attributions "
            f"(0 to {attributions.shape[2] - 1}). Please provide valid dimension indices."
        )

    # Extract relevant masks for the specified class and dimensions
    masks = _extract_masks_for_class(dataset, class_label, dim_indices)

    # If no masks were found, raise an informative error
    if not masks:
        class_str = f"class {class_label}" if class_label is not None else "any class"
        dim_str = (
            f"dimensions {dim_indices}" if dim_indices is not None else "any dimension"
        )
        raise ValueError(
            f"No feature masks found for {class_str} and {dim_str}. "
            f"Make sure the dataset contains features for the specified class and dimensions."
        )

    # Validate feature mask shapes
    for key, mask in masks.items():
        if mask.shape[1] != n_timesteps:
            raise ValueError(
                f"Feature mask '{key}' has {mask.shape[1]} timesteps, but dataset has {n_timesteps}. "
                f"All feature masks must match the dataset's time dimension."
            )

    # FIXED: Organize ground truth masks by dimension before combining
    # This ensures each dimension's masks are combined separately
    ground_truth_by_dim = {
        dim_idx: np.zeros((len(sample_indices), n_timesteps), dtype=bool)
        for dim_idx in dim_indices
    }

    for key, mask in masks.items():
        # Determine which dimension this mask belongs to
        if "_dim" not in key:
            # No dimension specified, assume dim0
            dim_idx = 0
        else:
            try:
                dim_part = key.split("_dim")[-1]
                dim_idx = int(dim_part)
            except (ValueError, IndexError):
                # If we can't parse the dimension, skip this mask
                continue

        if dim_idx in dim_indices:
            # Extract the relevant samples and combine this mask with existing ones for this dimension
            mask_samples = mask[sample_indices]
            ground_truth_by_dim[dim_idx] = np.logical_or(
                ground_truth_by_dim[dim_idx], mask_samples
            )

    # Validate that we have valid masks for each dimension
    for dim_idx in dim_indices:
        if not np.any(ground_truth_by_dim[dim_idx]):
            print(
                f"Warning: No feature masks found for dimension {dim_idx}. This may affect metric calculations."
            )

    # Binarize attributions if needed
    if threshold is not None:
        attributions = _binarize_attributions(attributions, threshold)
    elif not np.issubdtype(attributions.dtype, np.bool_):
        raise ValueError(
            "Attributions are not boolean and no threshold was provided for binarization. "
            "Please provide a threshold value to binarize continuous attribution values."
        )

    # Validate attributions against samples
    if attributions.shape[0] < len(sample_indices):
        raise ValueError(
            f"Attribution array has {attributions.shape[0]} samples, but {len(sample_indices)} "
            f"sample indices were provided. The attribution array must have at least as many "
            f"samples as the number of sample indices."
        )

    # Select only the specified sample indices from the attributions
    attributions = attributions[: len(sample_indices)]

    return attributions, ground_truth_by_dim, sample_indices, dim_indices


def precision_score(
    attributions: np.ndarray,
    dataset: Dict,
    sample_indices: Optional[List[int]] = None,
    dim_indices: Optional[List[int]] = None,
    threshold: Optional[float] = None,
    class_label: Optional[int] = None,
    average: Optional[str] = "macro",
) -> Union[float, Dict[int, float], Dict[Tuple[int, int], float]]:
    """Calculate precision score for feature attributions.

    Precision = TP / (TP + FP)

    Args:
        attributions (np.ndarray): Feature attribution values. Can be:
            - 1D array (n_timesteps,): Single sample, single dimension
            - 2D array (n_timesteps, n_dimensions): Single sample, multiple dimensions
            - 3D array (n_samples, n_timesteps, n_dimensions): Multiple samples, multiple dimensions
        dataset (Dict): Dataset dictionary returned by TimeSeriesBuilder.build().
        sample_indices (Optional[List[int]]): Sample indices to include.
            For 1D or 2D attributions (single sample), you must specify which sample
            to compare against. For 3D attributions, defaults to all samples.
        dim_indices (Optional[List[int]]): Dimension indices to include.
            If None, uses all dimensions available in the attribution array.
        threshold (Optional[float]): Threshold for binarizing attribution values.
            Required if attributions are not already boolean.
        class_label (Optional[int]): Class label to calculate precision for.
            If None, uses all feature masks regardless of class.
        average (Optional[str]): Method for averaging:
            - 'macro': Average precision across samples and dimensions
            - 'per_sample': Return precision for each sample (averaged across dimensions)
            - 'per_dimension': Return precision for each dimension (averaged across samples)
            - 'per_sample_dimension': Return precision for each sample-dimension pair
            - None: Return overall precision (all samples and dimensions combined)

    Returns:
        Union[float, Dict[int, float], Dict[Tuple[int, int], float]]:
            Precision score(s) depending on the averaging method.

    Raises:
        ValueError: If inputs have incompatible shapes or dimensions.
    """
    # Validate and prepare inputs
    attributions, ground_truth_by_dim, sample_indices, dim_indices = (
        _validate_and_prepare_inputs(
            attributions, dataset, sample_indices, dim_indices, threshold, class_label
        )
    )

    # Extract needed dimensions
    n_samples = len(sample_indices)
    n_dimensions = len(dim_indices)

    # Initialize results container based on average method
    if average == "per_sample_dimension":
        results = {}
    elif average == "per_sample":
        results = {sample_idx: 0.0 for sample_idx in sample_indices}
    elif average == "per_dimension":
        results = {dim_idx: 0.0 for dim_idx in dim_indices}
    else:
        # For 'macro' or None, initialize a single result
        results = 0.0

    # Calculate precision for each sample and dimension
    for i, sample_idx in enumerate(sample_indices):
        for j, dim_idx in enumerate(dim_indices):
            # Get attribution and ground truth for this sample and dimension
            attribution = attributions[i, :, j]
            mask = ground_truth_by_dim[dim_idx][i]

            # Calculate true positives and false positives
            tp = np.sum(attribution & mask)
            fp = np.sum(attribution & ~mask)

            # Calculate precision (handle division by zero)
            if tp + fp == 0:
                precision = 0.0  # No positive predictions
            else:
                precision = tp / (tp + fp)

            # Store result based on average method
            if average == "per_sample_dimension":
                results[(sample_idx, dim_idx)] = precision
            elif average == "per_sample":
                results[sample_idx] += precision / n_dimensions
            elif average == "per_dimension":
                results[dim_idx] += precision / n_samples
            else:
                results += precision / (n_samples * n_dimensions)

    return results


def recall_score(
    attributions: np.ndarray,
    dataset: Dict,
    sample_indices: Optional[List[int]] = None,
    dim_indices: Optional[List[int]] = None,
    threshold: Optional[float] = None,
    class_label: Optional[int] = None,
    average: Optional[str] = "macro",
) -> Union[float, Dict[int, float], Dict[Tuple[int, int], float]]:
    """Calculate recall score for feature attributions.

    Recall = TP / (TP + FN)

    Args:
        attributions (np.ndarray): Feature attribution values. Can be:
            - 1D array (n_timesteps,): Single sample, single dimension
            - 2D array (n_timesteps, n_dimensions): Single sample, multiple dimensions
            - 3D array (n_samples, n_timesteps, n_dimensions): Multiple samples, multiple dimensions
        dataset (Dict): Dataset dictionary returned by TimeSeriesBuilder.build().
        sample_indices (Optional[List[int]]): Sample indices to include.
            For 1D or 2D attributions (single sample), you must specify which sample
            to compare against. For 3D attributions, defaults to all samples.
        dim_indices (Optional[List[int]]): Dimension indices to include.
            If None, uses all dimensions available in the attribution array.
        threshold (Optional[float]): Threshold for binarizing attribution values.
            Required if attributions are not already boolean.
        class_label (Optional[int]): Class label to calculate recall for.
            If None, uses all feature masks regardless of class.
        average (Optional[str]): Method for averaging:
            - 'macro': Average recall across samples and dimensions
            - 'per_sample': Return recall for each sample (averaged across dimensions)
            - 'per_dimension': Return recall for each dimension (averaged across samples)
            - 'per_sample_dimension': Return recall for each sample-dimension pair
            - None: Return overall recall (all samples and dimensions combined)

    Returns:
        Union[float, Dict[int, float], Dict[Tuple[int, int], float]]:
            Recall score(s) depending on the averaging method.

    Raises:
        ValueError: If inputs have incompatible shapes or dimensions.
    """
    # Validate and prepare inputs
    attributions, ground_truth_by_dim, sample_indices, dim_indices = (
        _validate_and_prepare_inputs(
            attributions, dataset, sample_indices, dim_indices, threshold, class_label
        )
    )

    # Extract needed dimensions
    n_samples = len(sample_indices)
    n_dimensions = len(dim_indices)

    # Initialize results container based on average method
    if average == "per_sample_dimension":
        results = {}
    elif average == "per_sample":
        results = {sample_idx: 0.0 for sample_idx in sample_indices}
    elif average == "per_dimension":
        results = {dim_idx: 0.0 for dim_idx in dim_indices}
    else:
        # For 'macro' or None, initialize a single result
        results = 0.0

    # Calculate recall for each sample and dimension
    for i, sample_idx in enumerate(sample_indices):
        for j, dim_idx in enumerate(dim_indices):
            # Get attribution and ground truth for this sample and dimension
            attribution = attributions[i, :, j]
            mask = ground_truth_by_dim[dim_idx][i]

            # Calculate true positives and false negatives
            tp = np.sum(attribution & mask)
            fn = np.sum(~attribution & mask)

            # Calculate recall (handle division by zero)
            if tp + fn == 0:
                recall = 1.0  # No actual positives - perfect recall
            else:
                recall = tp / (tp + fn)

            # Store result based on average method
            if average == "per_sample_dimension":
                results[(sample_idx, dim_idx)] = recall
            elif average == "per_sample":
                results[sample_idx] += recall / n_dimensions
            elif average == "per_dimension":
                results[dim_idx] += recall / n_samples
            else:
                results += recall / (n_samples * n_dimensions)

    return results


def f1_score(
    attributions: np.ndarray,
    dataset: Dict,
    sample_indices: Optional[List[int]] = None,
    dim_indices: Optional[List[int]] = None,
    threshold: Optional[float] = None,
    class_label: Optional[int] = None,
    average: Optional[str] = "macro",
) -> Union[float, Dict[int, float], Dict[Tuple[int, int], float]]:
    """Calculate F1 score (harmonic mean of precision and recall) for feature attributions.

    F1 = 2 * (precision * recall) / (precision + recall)

    Args:
        attributions (np.ndarray): Feature attribution values. Can be:
            - 1D array (n_timesteps,): Single sample, single dimension
            - 2D array (n_timesteps, n_dimensions): Single sample, multiple dimensions
            - 3D array (n_samples, n_timesteps, n_dimensions): Multiple samples, multiple dimensions
        dataset (Dict): Dataset dictionary returned by TimeSeriesBuilder.build().
        sample_indices (Optional[List[int]]): Sample indices to include.
            For 1D or 2D attributions (single sample), you must specify which sample
            to compare against. For 3D attributions, defaults to all samples.
        dim_indices (Optional[List[int]]): Dimension indices to include.
            If None, uses all dimensions available in the attribution array.
        threshold (Optional[float]): Threshold for binarizing attribution values.
            Required if attributions are not already boolean.
        class_label (Optional[int]): Class label to calculate F1 for.
            If None, uses all feature masks regardless of class.
        average (Optional[str]): Method for averaging:
            - 'macro': Average F1 across samples and dimensions
            - 'per_sample': Return F1 for each sample (averaged across dimensions)
            - 'per_dimension': Return F1 for each dimension (averaged across samples)
            - 'per_sample_dimension': Return F1 for each sample-dimension pair
            - None: Return overall F1 (all samples and dimensions combined)

    Returns:
        Union[float, Dict[int, float], Dict[Tuple[int, int], float]]:
            F1 score(s) depending on the averaging method.

    Raises:
        ValueError: If inputs have incompatible shapes or dimensions.
    """
    # Get precision and recall scores
    precision = precision_score(
        attributions,
        dataset,
        sample_indices,
        dim_indices,
        threshold,
        class_label,
        average,
    )

    recall = recall_score(
        attributions,
        dataset,
        sample_indices,
        dim_indices,
        threshold,
        class_label,
        average,
    )

    # Calculate F1 score
    if isinstance(precision, dict):
        f1 = {}
        for key in precision:
            p = precision[key]
            r = recall[key]
            if p + r == 0:
                f1[key] = 0.0
            else:
                f1[key] = 2 * (p * r) / (p + r)
        return f1
    else:
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)
