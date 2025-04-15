"""Metrics for evaluating feature attributions against ground truth."""

from typing import Any, Dict, List, Optional, Tuple, Union

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


def _validate_and_prepare_inputs(
    attributions: np.ndarray,
    dataset: Dict[str, Any],
    sample_indices: Optional[List[int]] = None,
    dim_indices: Optional[List[int]] = None,
    threshold: Optional[float] = None,
    class_label: Optional[int] = None,
    allow_continuous: bool = False,
) -> Tuple[np.ndarray, Dict[int, np.ndarray], List[int], List[int]]:
    """Validate and prepare inputs for precision and recall calculations.

    Args:
        attributions (np.ndarray): Feature attribution values.
        dataset (Dict[str, Any]): Dataset dictionary returned by TimeSeriesBuilder.build().
        sample_indices (Optional[List[int]]): Sample indices to include.
        dim_indices (Optional[List[int]]): Dimension indices to include.
        threshold (Optional[float]): Threshold for binarizing attribution values.
        class_label (Optional[int]): Class label to calculate metrics for.
        allow_continuous (bool): If True, don't require binarization for continuous attributions.
            Used for metrics that work with raw attribution values like NAC and AUC-ROC.

    Returns:
        Tuple containing:
            - np.ndarray: Attributions (binarized or continuous based on parameters)
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

    # Organize ground truth masks by dimension before combining
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
    elif not allow_continuous and not np.issubdtype(attributions.dtype, np.bool_):
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


def _validate_attribution_and_extract_feature_masks(
    attribution, dataset, sample_indices=None, class_label=None, threshold=None
):
    """Validate attribution and extract relevant feature masks.

    Helper function to avoid code duplication between different metrics.

    Args:
        attribution (np.ndarray): Attribution array with shape [batch_size, time_steps, channels]
            or [batch_size, channels, time_steps] depending on the data_format.
            Values can be boolean or continuous (if threshold is provided).
        dataset (dict): Dataset dictionary from TimeSeriesBuilder.build().
        sample_indices (list, optional): Indices of samples to evaluate.
            If None, all samples are used.
        class_label (int, optional): Class label to consider for evaluation.
            If None, uses the actual class labels from dataset["y"].
        threshold (float, optional): Threshold for converting continuous attributions
            to binary. If None, attribution must already be binary (boolean).

    Returns:
        tuple:
            - Binary attribution array normalized to channels_last format for processing
            - Dictionary of relevant feature masks
            - List of validated sample indices
            - Data format from dataset metadata

    Raises:
        ValueError: If attribution has wrong shape or is not binary when threshold=None.
    """
    # Extract data format from metadata or assume default
    data_format = dataset.get("metadata", {}).get("data_format", "channels_last")

    # Handle sample indices
    if sample_indices is None:
        sample_indices = np.arange(len(dataset["y"]))
    else:
        # Validate sample indices
        max_idx = len(dataset["y"]) - 1
        for idx in sample_indices:
            if idx < 0 or idx > max_idx:
                raise ValueError(f"Sample index {idx} out of range (0 to {max_idx}).")

    # Get metadata
    n_timesteps = dataset["metadata"]["n_timesteps"]
    n_dimensions = dataset["metadata"]["n_dimensions"]

    # Instead of validating strictly, try to infer the format of the attribution
    # and convert if necessary

    # Check if the attribution could be in channels_last format [batch, time, channels]
    if (
        len(attribution.shape) == 3
        and attribution.shape[1] == n_timesteps
        and attribution.shape[2] == n_dimensions
    ):
        # Attribution is already in channels_last format
        attribution_internal = attribution

    # Check if the attribution could be in channels_first format [batch, channels, time]
    elif (
        len(attribution.shape) == 3
        and attribution.shape[1] == n_dimensions
        and attribution.shape[2] == n_timesteps
    ):
        # Need to transpose to channels_last for internal processing
        attribution_internal = np.transpose(attribution, (0, 2, 1))

    # If either dimension is 1, try to reshape
    elif len(attribution.shape) == 3 and (
        attribution.shape[1] == 1 or attribution.shape[2] == 1
    ):
        if attribution.shape[1] == 1:  # [batch, 1, any]
            # Reshape to [batch, any, 1]
            attribution_internal = attribution.transpose(0, 2, 1)
        elif attribution.shape[2] == 1:  # [batch, any, 1]
            # Keep as is
            attribution_internal = attribution

        # Check if the resulting shape matches expected dimensions
        if (
            attribution_internal.shape[1] == n_timesteps
            and attribution_internal.shape[2] == n_dimensions
        ):
            pass  # Shape is now correct
        elif (
            attribution_internal.shape[1] == n_dimensions
            and attribution_internal.shape[2] == n_timesteps
        ):
            # Need to transpose again
            attribution_internal = np.transpose(attribution_internal, (0, 2, 1))
        else:
            raise ValueError(
                f"Attribution shape {attribution.shape} cannot be reshaped to match "
                f"dataset with dimensions {n_dimensions} and timesteps {n_timesteps}."
            )
    else:
        raise ValueError(
            f"Attribution shape {attribution.shape} doesn't match dataset with "
            f"{n_dimensions} dimensions and {n_timesteps} timesteps. "
            f"Expected either [batch, {n_timesteps}, {n_dimensions}] (channels_last) or "
            f"[batch, {n_dimensions}, {n_timesteps}] (channels_first)."
        )

    # Convert continuous attributions to binary if threshold provided
    if threshold is not None:
        binary_attribution = attribution_internal >= threshold
    else:
        # Ensure attribution is binary
        if not np.issubdtype(attribution_internal.dtype, np.bool_):
            raise ValueError(
                "Attribution must be boolean type when no threshold was provided."
            )
        binary_attribution = attribution_internal

    # Extract feature masks for relevant samples
    feature_masks = {}
    for key, mask in dataset["feature_masks"].items():
        # Only include masks for relevant class
        if class_label is not None:
            class_str = f"class_{class_label}_"
            if not key.startswith(class_str):
                continue

        # Only include samples from sample_indices
        feature_masks[key] = mask[sample_indices]

    return binary_attribution, feature_masks, sample_indices, data_format


def precision_score(
    attribution,
    dataset,
    sample_indices=None,
    class_label=None,
    threshold=None,
    average="macro",
):
    """Calculate precision score for attribution against ground truth.

    Measures how many of the attributed timesteps are actually within ground truth features.
    Precision = TP / (TP + FP)

    Args:
        attribution (np.ndarray): Attribution array with shape depending on data_format:
            - 'channels_last': [batch_size, time_steps, channels]
            - 'channels_first': [batch_size, channels, time_steps]
            Values can be boolean or continuous (if threshold is provided).
        dataset (dict): Dataset dictionary from TimeSeriesBuilder.build().
        sample_indices (list, optional): Indices of samples to evaluate.
            If None, all samples are used.
        class_label (int, optional): Class label to consider for evaluation.
            If None, uses the actual class labels from dataset["y"].
        threshold (float, optional): Threshold for converting continuous attributions
            to binary. If None, attribution must already be binary (boolean).
        average (str): Averaging method for multi-sample/multi-dimension results:
            - "macro": Average over samples and dimensions
            - "per_sample": Return score for each sample (avg across dimensions)
            - "per_dimension": Return score for each dimension (avg across samples)
            - "per_sample_dimension": Return score for each sample-dimension pair

    Returns:
        float or dict: Precision score(s) based on the specified averaging method.
    """
    # Validate and prepare data
    binary_attribution, feature_masks, sample_indices, data_format = (
        _validate_attribution_and_extract_feature_masks(
            attribution, dataset, sample_indices, class_label, threshold
        )
    )

    # Calculate precision for each sample-dimension pair
    results = {}
    for i, sample_idx in enumerate(sample_indices):
        # Iterate through dimensions
        for dim in range(dataset["metadata"]["n_dimensions"]):
            # Combine all feature masks for this sample & dimension
            combined_mask = np.zeros(dataset["metadata"]["n_timesteps"], dtype=bool)

            for key, masks in feature_masks.items():
                # Check if mask is relevant for this dimension
                if f"_dim{dim}" in key or (
                    dim == 0 and "_dim" not in key
                ):  # Support legacy format
                    combined_mask |= masks[i]

            # Calculate true positives: attribution AND ground truth
            true_positives = np.sum(binary_attribution[i, :, dim] & combined_mask)

            # Calculate false positives: attribution AND NOT ground truth
            false_positives = np.sum(binary_attribution[i, :, dim] & ~combined_mask)

            # Calculate precision
            if (true_positives + false_positives) > 0:
                precision = true_positives / (true_positives + false_positives)
            else:
                precision = 0  # Undefined precision, set to 0 by convention

            # Store result for this sample-dimension pair
            results[(sample_idx, dim)] = precision

    # Average results based on the specified method
    if average == "macro":
        return np.mean(list(results.values()))
    elif average == "per_sample":
        sample_results = {}
        for sample_idx in sample_indices:
            sample_dims = [(s, d) for (s, d) in results.keys() if s == sample_idx]
            sample_results[sample_idx] = np.mean([results[k] for k in sample_dims])
        return sample_results
    elif average == "per_dimension":
        dim_results = {}
        for dim in range(dataset["metadata"]["n_dimensions"]):
            dim_samples = [(s, d) for (s, d) in results.keys() if d == dim]
            dim_results[dim] = np.mean([results[k] for k in dim_samples])
        return list(dim_results.values()) if len(dim_results) > 1 else dim_results[0]
    elif average == "per_sample_dimension":
        return results
    else:
        raise ValueError(f"Invalid average method: {average}")


def recall_score(
    attribution,
    dataset,
    sample_indices=None,
    class_label=None,
    threshold=None,
    average="macro",
):
    """Calculate recall score for attribution against ground truth.

    Measures how much of the ground truth feature is captured by the attribution.
    Recall = TP / (TP + FN)

    Args:
        attribution (np.ndarray): Attribution array with shape depending on data_format:
            - 'channels_last': [batch_size, time_steps, channels]
            - 'channels_first': [batch_size, channels, time_steps]
            Values can be boolean or continuous (if threshold is provided).
        dataset (dict): Dataset dictionary from TimeSeriesBuilder.build().
        sample_indices (list, optional): Indices of samples to evaluate.
            If None, all samples are used.
        class_label (int, optional): Class label to consider for evaluation.
            If None, uses the actual class labels from dataset["y"].
        threshold (float, optional): Threshold for converting continuous attributions
            to binary. If None, attribution must already be binary (boolean).
        average (str): Averaging method for multi-sample/multi-dimension results:
            - "macro": Average over samples and dimensions
            - "per_sample": Return score for each sample (avg across dimensions)
            - "per_dimension": Return score for each dimension (avg across samples)
            - "per_sample_dimension": Return score for each sample-dimension pair

    Returns:
        float or dict: Recall score(s) based on the specified averaging method.
    """
    # Validate and prepare data
    binary_attribution, feature_masks, sample_indices, data_format = (
        _validate_attribution_and_extract_feature_masks(
            attribution, dataset, sample_indices, class_label, threshold
        )
    )

    # Calculate recall for each sample-dimension pair
    results = {}
    for i, sample_idx in enumerate(sample_indices):
        # Iterate through dimensions
        for dim in range(dataset["metadata"]["n_dimensions"]):
            # Combine all feature masks for this sample & dimension
            combined_mask = np.zeros(dataset["metadata"]["n_timesteps"], dtype=bool)

            for key, masks in feature_masks.items():
                # Check if mask is relevant for this dimension
                if f"_dim{dim}" in key or (
                    dim == 0 and "_dim" not in key
                ):  # Support legacy format
                    combined_mask |= masks[i]

            # Calculate true positives: attribution AND ground truth
            true_positives = np.sum(binary_attribution[i, :, dim] & combined_mask)

            # Calculate false negatives: NOT attribution AND ground truth
            false_negatives = np.sum((~binary_attribution[i, :, dim]) & combined_mask)

            # Calculate recall
            if (true_positives + false_negatives) > 0:
                recall = true_positives / (true_positives + false_negatives)
            else:
                recall = 0  # No ground truth, set to 0 by convention

            # Store result for this sample-dimension pair
            results[(sample_idx, dim)] = recall

    # Average results based on the specified method
    if average == "macro":
        return np.mean(list(results.values()))
    elif average == "per_sample":
        sample_results = {}
        for sample_idx in sample_indices:
            sample_dims = [(s, d) for (s, d) in results.keys() if s == sample_idx]
            sample_results[sample_idx] = np.mean([results[k] for k in sample_dims])
        return sample_results
    elif average == "per_dimension":
        dim_results = {}
        for dim in range(dataset["metadata"]["n_dimensions"]):
            dim_samples = [(s, d) for (s, d) in results.keys() if d == dim]
            dim_results[dim] = np.mean([results[k] for k in dim_samples])
        return list(dim_results.values()) if len(dim_results) > 1 else dim_results[0]
    elif average == "per_sample_dimension":
        return results
    else:
        raise ValueError(f"Invalid average method: {average}")


def f1_score(
    attribution,
    dataset,
    sample_indices=None,
    class_label=None,
    threshold=None,
    average="macro",
):
    """Calculate F1 score for attribution against ground truth.

    F1 score is the harmonic mean of precision and recall:
    F1 = 2 * (precision * recall) / (precision + recall)

    Args:
        attribution (np.ndarray): Attribution array with shape depending on data_format:
            - 'channels_last': [batch_size, time_steps, channels]
            - 'channels_first': [batch_size, channels, time_steps]
            Values can be boolean or continuous (if threshold is provided).
        dataset (dict): Dataset dictionary from TimeSeriesBuilder.build().
        sample_indices (list, optional): Indices of samples to evaluate.
            If None, all samples are used.
        class_label (int, optional): Class label to consider for evaluation.
            If None, uses the actual class labels from dataset["y"].
        threshold (float, optional): Threshold for converting continuous attributions
            to binary. If None, attribution must already be binary (boolean).
        average (str): Averaging method for multi-sample/multi-dimension results:
            - "macro": Average over samples and dimensions
            - "per_sample": Return score for each sample (avg across dimensions)
            - "per_dimension": Return score for each dimension (avg across samples)
            - "per_sample_dimension": Return score for each sample-dimension pair

    Returns:
        float or dict: F1 score(s) based on the specified averaging method.
    """
    # Calculate precision and recall first
    precision_results = precision_score(
        attribution,
        dataset,
        sample_indices,
        class_label,
        threshold,
        "per_sample_dimension",
    )
    recall_results = recall_score(
        attribution,
        dataset,
        sample_indices,
        class_label,
        threshold,
        "per_sample_dimension",
    )

    # Calculate F1 for each sample-dimension pair
    results = {}
    for key in precision_results.keys():
        precision_val = precision_results[key]
        recall_val = recall_results[key]

        # Calculate F1 score
        if precision_val + recall_val > 0:
            f1 = 2 * (precision_val * recall_val) / (precision_val + recall_val)
        else:
            f1 = 0  # Both precision and recall are 0

        results[key] = f1

    # Average results based on the specified method
    if average == "macro":
        return np.mean(list(results.values()))
    elif average == "per_sample":
        sample_results = {}
        sample_indices = set(s for (s, _) in results.keys())
        for sample_idx in sample_indices:
            sample_dims = [(s, d) for (s, d) in results.keys() if s == sample_idx]
            sample_results[sample_idx] = np.mean([results[k] for k in sample_dims])
        return sample_results
    elif average == "per_dimension":
        dim_results = {}
        n_dimensions = max(d for (_, d) in results.keys()) + 1
        for dim in range(n_dimensions):
            dim_samples = [(s, d) for (s, d) in results.keys() if d == dim]
            if dim_samples:
                dim_results[dim] = np.mean([results[k] for k in dim_samples])
        return list(dim_results.values()) if len(dim_results) > 1 else dim_results[0]
    elif average == "per_sample_dimension":
        return results
    else:
        raise ValueError(f"Invalid average method: {average}")


def nac_score(
    attributions: np.ndarray,
    dataset: Dict,
    sample_indices: Optional[List[int]] = None,
    dim_indices: Optional[List[int]] = None,
    class_label: Optional[int] = None,
    average: Optional[str] = "macro",
    ground_truth_only: bool = True,
) -> Union[float, Dict[int, float], Dict[Tuple[int, int], float]]:
    """Calculate Normalized Attribution Correspondence (NAC) score for feature attributions.

    NAC measures how well attribution values correspond with ground truth by standardizing
    the attribution values and taking the mean at specified locations. It evaluates whether
    important features receive significantly higher attribution scores than would be expected
    by chance.

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
        class_label (Optional[int]): Class label to calculate NAC for.
            If None, uses all feature masks regardless of class.
        average (Optional[str]): Method for averaging:
            - 'macro': Average NAC across samples and dimensions
            - 'per_sample': Return NAC for each sample (averaged across dimensions)
            - 'per_dimension': Return NAC for each dimension (averaged across samples)
            - 'per_sample_dimension': Return NAC for each sample-dimension pair
            - None: Return overall NAC (all samples and dimensions combined)
        ground_truth_only (bool): If True, calculate NAC only for timesteps with ground truth
            features. If False, calculate for timesteps without ground truth features.

    Returns:
        Union[float, Dict[int, float], Dict[Tuple[int, int], float]]:
            NAC score(s) depending on the averaging method.
    """
    # Validate and prepare inputs (pass None for threshold since we don't binarize here)
    attributions, ground_truth_by_dim, sample_indices, dim_indices = (
        _validate_and_prepare_inputs(
            attributions,
            dataset,
            sample_indices,
            dim_indices,
            None,
            class_label,
            allow_continuous=True,
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

    # Calculate NAC for each sample and dimension
    for i, sample_idx in enumerate(sample_indices):
        for j, dim_idx in enumerate(dim_indices):
            # Get attribution and ground truth for this sample and dimension
            attribution = attributions[i, :, j]
            mask = ground_truth_by_dim[dim_idx][i]

            # Select regions based on ground_truth_only parameter
            if ground_truth_only:
                region_mask = mask
            else:
                region_mask = ~mask

            # Skip if no relevant regions (mask is all False)
            if not np.any(region_mask):
                nac = 0.0  # Default value when no regions to evaluate
            else:
                # Standardize attribution values (z-score normalization)
                attribution_mean = np.mean(attribution)
                attribution_std = np.std(attribution)

                # Handle case where standard deviation is 0
                if attribution_std == 0:
                    standardized_attribution = np.zeros_like(attribution)
                else:
                    standardized_attribution = (
                        attribution - attribution_mean
                    ) / attribution_std

                # Calculate NAC: mean of standardized values at selected regions
                nac = np.mean(standardized_attribution[region_mask])

            # Store result based on average method
            if average == "per_sample_dimension":
                results[(sample_idx, dim_idx)] = nac
            elif average == "per_sample":
                results[sample_idx] += nac / n_dimensions
            elif average == "per_dimension":
                results[dim_idx] += nac / n_samples
            else:
                results += nac / (n_samples * n_dimensions)

    return results


def auc_roc_score(
    attributions: np.ndarray,
    dataset: Dict,
    sample_indices: Optional[List[int]] = None,
    dim_indices: Optional[List[int]] = None,
    class_label: Optional[int] = None,
    average: Optional[str] = "macro",
) -> Union[float, Dict[int, float], Dict[Tuple[int, int], float]]:
    """Calculate AUC-ROC score for feature attributions.

    AUC-ROC measures how well the attribution values discriminate between
    important and non-important features across all possible threshold values.
    A score of 0.5 indicates random performance, while 1.0 indicates perfect
    discrimination.

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
        class_label (Optional[int]): Class label to calculate AUC-ROC for.
            If None, uses all feature masks regardless of class.
        average (Optional[str]): Method for averaging:
            - 'macro': Average AUC-ROC across samples and dimensions
            - 'per_sample': Return AUC-ROC for each sample (averaged across dimensions)
            - 'per_dimension': Return AUC-ROC for each dimension (averaged across samples)
            - 'per_sample_dimension': Return AUC-ROC for each sample-dimension pair
            - None: Return overall AUC-ROC (all samples and dimensions combined)

    Returns:
        Union[float, Dict[int, float], Dict[Tuple[int, int], float]]:
            AUC-ROC score(s) depending on the averaging method.
    """
    # Validate and prepare inputs (pass None for threshold since we don't binarize here)
    attributions, ground_truth_by_dim, sample_indices, dim_indices = (
        _validate_and_prepare_inputs(
            attributions,
            dataset,
            sample_indices,
            dim_indices,
            None,
            class_label,
            allow_continuous=True,
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

    # Calculate AUC-ROC for each sample and dimension
    for i, sample_idx in enumerate(sample_indices):
        for j, dim_idx in enumerate(dim_indices):
            # Get attribution and ground truth for this sample and dimension
            attribution = attributions[i, :, j]
            mask = ground_truth_by_dim[dim_idx][i]

            # Skip if all ground truth values are the same (AUC-ROC undefined)
            if np.all(mask) or not np.any(mask):
                auc = 0.5  # Default value when all ground truth is the same
            else:
                # Calculate TPR and FPR at each possible threshold
                # Use unique attribution values as thresholds for efficiency
                thresholds = np.unique(attribution)

                # Add one threshold above the maximum to ensure we get (0,0) point
                if thresholds.size > 0:
                    thresholds = np.append(thresholds, thresholds.max() + 1)

                n_pos = np.sum(mask)
                n_neg = mask.size - n_pos

                tpr_values = []
                fpr_values = []

                for threshold in sorted(thresholds, reverse=True):
                    pred_pos = attribution >= threshold
                    tp = np.sum(pred_pos & mask)
                    fp = np.sum(pred_pos & ~mask)

                    tpr = tp / n_pos if n_pos > 0 else 0
                    fpr = fp / n_neg if n_neg > 0 else 0

                    tpr_values.append(tpr)
                    fpr_values.append(fpr)

                # Convert to numpy arrays for calculation
                tpr_values = np.array(tpr_values)
                fpr_values = np.array(fpr_values)

                # Calculate AUC using the trapezoidal rule
                auc = np.trapz(
                    tpr_values, fpr_values
                )  # TODO: replace with np.trapezoid once tsxai uses numpy>=2.0.0

                # Handle special case where fpr_values might not be monotonically increasing
                if np.any(np.diff(fpr_values) < 0):
                    # Sort points by fpr
                    sort_idx = np.argsort(fpr_values)
                    fpr_sorted = fpr_values[sort_idx]
                    tpr_sorted = tpr_values[sort_idx]

                    # Remove duplicate fpr values (keep max tpr for each fpr)
                    unique_fprs, unique_indices = np.unique(
                        fpr_sorted, return_index=True
                    )
                    if len(unique_fprs) > 1:
                        unique_tprs = np.zeros_like(unique_fprs)

                        for k, fpr in enumerate(unique_fprs):
                            mask = fpr_sorted == fpr
                            unique_tprs[k] = np.max(tpr_sorted[mask])

                        auc = np.trapz(
                            unique_tprs, unique_fprs
                        )  # TODO: replace with np.trapezoid once tsxai uses numpy>=2.0.0

            # Store result based on average method
            if average == "per_sample_dimension":
                results[(sample_idx, dim_idx)] = auc
            elif average == "per_sample":
                results[sample_idx] += auc / n_dimensions
            elif average == "per_dimension":
                results[dim_idx] += auc / n_samples
            else:
                results += auc / (n_samples * n_dimensions)

    return results
