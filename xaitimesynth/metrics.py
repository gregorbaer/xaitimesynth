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

    # Case 3: attributions is (n_samples, n_timesteps, n_dimensions) or (n_samples, n_dimensions, n_timesteps)
    elif len(attribution_shape) == 3:
        # We need to infer the format based on the attribution shape, NOT the dataset format
        # This is because attributions might be provided in a different format than the dataset

        # Check if attribution has correct shape for channels_last format [batch, time, channels]
        if attribution_shape[1] == n_timesteps and attribution_shape[2] <= n_dimensions:
            # Attribution is in channels_last format - no conversion needed
            pass
        # Check if attribution has correct shape for channels_first format [batch, channels, time]
        elif (
            attribution_shape[2] == n_timesteps and attribution_shape[1] <= n_dimensions
        ):
            # Convert from channels_first to channels_last for internal processing
            attributions = np.transpose(attributions, (0, 2, 1))
        else:
            raise ValueError(
                f"Attribution shape {attribution_shape} doesn't match dataset dimensions. "
                f"Expected either [batch, {n_timesteps}, {n_dimensions}] (channels_last) or "
                f"[batch, {n_dimensions}, {n_timesteps}] (channels_first)."
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
            f"3D (n_samples, n_timesteps, n_dimensions) or (n_samples, n_dimensions, n_timesteps) with appropriate data_format."
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
    attribution: np.ndarray,
    dataset: Dict[str, Any],
    sample_indices: Optional[List[int]] = None,
    dim_indices: Optional[List[int]] = None,
    class_label: Optional[int] = None,
    threshold: Optional[float] = None,
    average: str = "macro",
) -> Union[float, Dict[int, float], Dict[Tuple[int, int], float]]:
    """Calculate precision score for attribution against ground truth.

    Measures how many of the attributed timesteps are actually within ground truth features.
    Precision = TP / (TP + FP)

    Args:
        attribution (np.ndarray): Attribution array. Can be:
            - 1D array (n_timesteps,): Single sample, single dimension
            - 2D array (n_timesteps, n_dimensions): Single sample, multiple dimensions
            - 3D array (n_samples, n_timesteps, n_dimensions): Multiple samples, multiple dimensions
        dataset (Dict[str, Any]): Dataset dictionary returned by TimeSeriesBuilder.build().
        sample_indices (Optional[List[int]]): Sample indices to include.
            For 1D or 2D attributions (single sample), you must specify which sample
            to compare against. For 3D attributions, defaults to all samples.
        dim_indices (Optional[List[int]]): Dimension indices to include.
            If None, uses all dimensions available in the attribution array.
        class_label (Optional[int]): Class label to calculate precision for.
            If None, uses all feature masks regardless of class.
        threshold (Optional[float]): Threshold for binarizing attribution values.
            If None, attribution must already be binary (boolean).
        average (str): Averaging method for multi-sample/multi-dimension results:
            - "macro": Average over samples and dimensions
            - "per_sample": Return score for each sample (avg across dimensions)
            - "per_dimension": Return score for each dimension (avg across samples)
            - "per_sample_dimension": Return score for each sample-dimension pair

    Returns:
        Union[float, Dict[int, float], Dict[Tuple[int, int], float]]:
            Precision score(s) based on the specified averaging method.
    """
    # Extract data using the enhanced helper function
    data = _extract_feature_data(
        attribution,
        dataset,
        sample_indices,
        dim_indices,
        class_label,
        threshold,
        needs_feature_values=False,
    )

    attributions = data["attributions"]
    masks = data["masks"]
    sample_indices = data["sample_indices"]
    dim_indices = data["dim_indices"]

    # Calculate precision for each sample-dimension pair
    results = {}
    for i, sample_idx in enumerate(sample_indices):
        for j, dim_idx in enumerate(dim_indices):
            # Get attribution and mask for this sample and dimension
            attr_values = attributions[i, :, j]
            mask_values = masks[i, :, j]

            # Ensure attr_values is boolean for bitwise operations
            if not np.issubdtype(attr_values.dtype, np.bool_):
                if threshold is not None:
                    attr_values = attr_values >= threshold
                else:
                    raise ValueError(
                        "Attribution values must be boolean when no threshold is provided. "
                        "Please provide a threshold or convert to boolean values."
                    )

            # Calculate true positives: attribution AND ground truth
            true_positives = np.sum(attr_values & mask_values)

            # Calculate false positives: attribution AND NOT ground truth
            false_positives = np.sum(attr_values & ~mask_values)

            # Calculate precision
            if (true_positives + false_positives) > 0:
                precision = true_positives / (true_positives + false_positives)
            else:
                precision = 0  # Undefined precision, set to 0 by convention

            # Store result for this sample-dimension pair
            results[(sample_idx, dim_idx)] = precision

    # Average results based on the specified method
    if average == "macro":
        return np.mean(list(results.values())) if results else 0.0
    elif average == "per_sample":
        sample_results = {}
        for sample_idx in sample_indices:
            sample_dims = [(s, d) for (s, d) in results.keys() if s == sample_idx]
            sample_results[sample_idx] = (
                np.mean([results[k] for k in sample_dims]) if sample_dims else 0.0
            )
        return sample_results
    elif average == "per_dimension":
        dim_results = {}
        for dim_idx in dim_indices:
            dim_samples = [(s, d) for (s, d) in results.keys() if d == dim_idx]
            dim_results[dim_idx] = (
                np.mean([results[k] for k in dim_samples]) if dim_samples else 0.0
            )
        if len(dim_results) > 1:
            return list(dim_results.values())
        elif dim_results:
            return next(iter(dim_results.values()))
        else:
            return 0.0
    elif average == "per_sample_dimension":
        return results
    else:
        raise ValueError(f"Invalid average method: {average}")


def recall_score(
    attribution: np.ndarray,
    dataset: Dict[str, Any],
    sample_indices: Optional[List[int]] = None,
    dim_indices: Optional[List[int]] = None,
    class_label: Optional[int] = None,
    threshold: Optional[float] = None,
    average: str = "macro",
) -> Union[float, Dict[int, float], Dict[Tuple[int, int], float]]:
    """Calculate recall score for attribution against ground truth.

    Measures how much of the ground truth feature is captured by the attribution.
    Recall = TP / (TP + FN)

    Args:
        attribution (np.ndarray): Attribution array. Can be:
            - 1D array (n_timesteps,): Single sample, single dimension
            - 2D array (n_timesteps, n_dimensions): Single sample, multiple dimensions
            - 3D array (n_samples, n_timesteps, n_dimensions): Multiple samples, multiple dimensions
        dataset (Dict[str, Any]): Dataset dictionary returned by TimeSeriesBuilder.build().
        sample_indices (Optional[List[int]]): Sample indices to include.
            For 1D or 2D attributions (single sample), you must specify which sample
            to compare against. For 3D attributions, defaults to all samples.
        dim_indices (Optional[List[int]]): Dimension indices to include.
            If None, uses all dimensions available in the attribution array.
        class_label (Optional[int]): Class label to calculate recall for.
            If None, uses all feature masks regardless of class.
        threshold (Optional[float]): Threshold for binarizing attribution values.
            If None, attribution must already be binary (boolean).
        average (str): Averaging method for multi-sample/multi-dimension results:
            - "macro": Average over samples and dimensions
            - "per_sample": Return score for each sample (avg across dimensions)
            - "per_dimension": Return score for each dimension (avg across samples)
            - "per_sample_dimension": Return score for each sample-dimension pair

    Returns:
        Union[float, Dict[int, float], Dict[Tuple[int, int], float]]:
            Recall score(s) based on the specified averaging method.
    """
    # Extract data using the enhanced helper function
    data = _extract_feature_data(
        attribution,
        dataset,
        sample_indices,
        dim_indices,
        class_label,
        threshold,
        needs_feature_values=False,
    )

    attributions = data["attributions"]
    masks = data["masks"]
    sample_indices = data["sample_indices"]
    dim_indices = data["dim_indices"]

    # Calculate recall for each sample-dimension pair
    results = {}
    for i, sample_idx in enumerate(sample_indices):
        for j, dim_idx in enumerate(dim_indices):
            # Get attribution and mask for this sample and dimension
            attr_values = attributions[i, :, j]
            mask_values = masks[i, :, j]

            # Ensure attr_values is boolean for bitwise operations
            if not np.issubdtype(attr_values.dtype, np.bool_):
                if threshold is not None:
                    attr_values = attr_values >= threshold
                else:
                    raise ValueError(
                        "Attribution values must be boolean when no threshold is provided. "
                        "Please provide a threshold or convert to boolean values."
                    )

            # Calculate true positives: attribution AND ground truth
            true_positives = np.sum(attr_values & mask_values)

            # Calculate false negatives: NOT attribution AND ground truth
            false_negatives = np.sum((~attr_values) & mask_values)

            # Calculate recall
            if (true_positives + false_negatives) > 0:
                recall = true_positives / (true_positives + false_negatives)
            else:
                recall = 0  # No ground truth, set to 0 by convention

            # Store result for this sample-dimension pair
            results[(sample_idx, dim_idx)] = recall

    # Average results based on the specified method
    if average == "macro":
        return np.mean(list(results.values())) if results else 0.0
    elif average == "per_sample":
        sample_results = {}
        for sample_idx in sample_indices:
            sample_dims = [(s, d) for (s, d) in results.keys() if s == sample_idx]
            sample_results[sample_idx] = (
                np.mean([results[k] for k in sample_dims]) if sample_dims else 0.0
            )
        return sample_results
    elif average == "per_dimension":
        dim_results = {}
        for dim_idx in dim_indices:
            dim_samples = [(s, d) for (s, d) in results.keys() if d == dim_idx]
            dim_results[dim_idx] = (
                np.mean([results[k] for k in dim_samples]) if dim_samples else 0.0
            )
        if len(dim_results) > 1:
            return list(dim_results.values())
        elif dim_results:
            return next(iter(dim_results.values()))
        else:
            return 0.0
    elif average == "per_sample_dimension":
        return results
    else:
        raise ValueError(f"Invalid average method: {average}")


def f1_score(
    attribution: np.ndarray,
    dataset: Dict[str, Any],
    sample_indices: Optional[List[int]] = None,
    dim_indices: Optional[List[int]] = None,
    class_label: Optional[int] = None,
    threshold: Optional[float] = None,
    average: str = "macro",
) -> Union[float, Dict[int, float], Dict[Tuple[int, int], float]]:
    """Calculate F1 score for attribution against ground truth.

    F1 score is the harmonic mean of precision and recall:
    F1 = 2 * (precision * recall) / (precision + recall)

    Args:
        attribution (np.ndarray): Attribution array. Can be:
            - 1D array (n_timesteps,): Single sample, single dimension
            - 2D array (n_timesteps, n_dimensions): Single sample, multiple dimensions
            - 3D array (n_samples, n_timesteps, n_dimensions): Multiple samples, multiple dimensions
        dataset (Dict[str, Any]): Dataset dictionary returned by TimeSeriesBuilder.build().
        sample_indices (Optional[List[int]]): Sample indices to include.
            For 1D or 2D attributions (single sample), you must specify which sample
            to compare against. For 3D attributions, defaults to all samples.
        dim_indices (Optional[List[int]]): Dimension indices to include.
            If None, uses all dimensions available in the attribution array.
        class_label (Optional[int]): Class label to calculate F1 score for.
            If None, uses all feature masks regardless of class.
        threshold (Optional[float]): Threshold for binarizing attribution values.
            If None, attribution must already be binary (boolean).
        average (str): Averaging method for multi-sample/multi-dimension results:
            - "macro": Average over samples and dimensions
            - "per_sample": Return score for each sample (avg across dimensions)
            - "per_dimension": Return score for each dimension (avg across samples)
            - "per_sample_dimension": Return score for each sample-dimension pair

    Returns:
        Union[float, Dict[int, float], Dict[Tuple[int, int], float]]:
            F1 score(s) based on the specified averaging method.
    """
    # Extract data using the enhanced helper function
    data = _extract_feature_data(
        attribution,
        dataset,
        sample_indices,
        dim_indices,
        class_label,
        threshold,
        needs_feature_values=False,
    )

    attributions = data["attributions"]
    masks = data["masks"]
    sample_indices = data["sample_indices"]
    dim_indices = data["dim_indices"]

    # Calculate F1 score for each sample-dimension pair
    results = {}
    for i, sample_idx in enumerate(sample_indices):
        for j, dim_idx in enumerate(dim_indices):
            # Get attribution and mask for this sample and dimension
            attr_values = attributions[i, :, j]
            mask_values = masks[i, :, j]

            # Ensure attr_values is boolean for bitwise operations
            if not np.issubdtype(attr_values.dtype, np.bool_):
                if threshold is not None:
                    attr_values = attr_values >= threshold
                else:
                    raise ValueError(
                        "Attribution values must be boolean when no threshold is provided. "
                        "Please provide a threshold or convert to boolean values."
                    )

            # Calculate true positives, false positives, and false negatives
            true_positives = np.sum(attr_values & mask_values)
            false_positives = np.sum(attr_values & ~mask_values)
            false_negatives = np.sum((~attr_values) & mask_values)

            # Calculate precision and recall
            if (true_positives + false_positives) > 0:
                precision = true_positives / (true_positives + false_positives)
            else:
                precision = 0.0

            if (true_positives + false_negatives) > 0:
                recall = true_positives / (true_positives + false_negatives)
            else:
                recall = 0.0

            # Calculate F1 score
            if precision + recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
            else:
                f1 = 0.0  # Both precision and recall are 0

            # Store result for this sample-dimension pair
            results[(sample_idx, dim_idx)] = f1

    # Average results based on the specified method
    if average == "macro":
        return np.mean(list(results.values())) if results else 0.0
    elif average == "per_sample":
        sample_results = {}
        for sample_idx in sample_indices:
            sample_dims = [(s, d) for (s, d) in results.keys() if s == sample_idx]
            sample_results[sample_idx] = (
                np.mean([results[k] for k in sample_dims]) if sample_dims else 0.0
            )
        return sample_results
    elif average == "per_dimension":
        dim_results = {}
        for dim_idx in dim_indices:
            dim_samples = [(s, d) for (s, d) in results.keys() if d == dim_idx]
            dim_results[dim_idx] = (
                np.mean([results[k] for k in dim_samples]) if dim_samples else 0.0
            )
        if len(dim_results) > 1:
            return list(dim_results.values())
        elif dim_results:
            return next(iter(dim_results.values()))
        else:
            return 0.0
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
    # Extract data using the enhanced helper function - NAC needs continuous values
    data = _extract_feature_data(
        attributions,
        dataset,
        sample_indices,
        dim_indices,
        class_label,
        threshold=None,  # No thresholding for NAC
        needs_feature_values=False,  # NAC only needs masks, not feature values
        allow_continuous=True,
    )

    attributions = data["attributions"]
    masks = data["masks"]
    sample_indices = data["sample_indices"]
    dim_indices = data["dim_indices"]

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
            # Get attribution and mask for this sample and dimension
            attribution = attributions[i, :, j]
            mask = masks[i, :, j]

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
    # Extract data using the enhanced helper function - AUC-ROC needs continuous values
    data = _extract_feature_data(
        attributions,
        dataset,
        sample_indices,
        dim_indices,
        class_label,
        threshold=None,  # No thresholding for AUC-ROC
        needs_feature_values=False,  # AUC-ROC only needs masks, not feature values
        allow_continuous=True,
    )

    attributions = data["attributions"]
    masks = data["masks"]
    sample_indices = data["sample_indices"]
    dim_indices = data["dim_indices"]

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
            # Get attribution and mask for this sample and dimension
            attribution = attributions[i, :, j]
            mask = masks[i, :, j]

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
                )  # TODO: replace with np.trapezoidal if downstream integration permits
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
                        )  # TODO: replace with np.trapezoidal if downstream integration permits
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


def auc_pr_score(
    attributions: np.ndarray,
    dataset: Dict,
    sample_indices: Optional[List[int]] = None,
    dim_indices: Optional[List[int]] = None,
    class_label: Optional[int] = None,
    average: Optional[str] = "macro",
) -> Union[float, Dict[int, float], Dict[Tuple[int, int], float]]:
    """Calculate AUC-PR score (Area Under the Precision-Recall Curve) for feature attributions.

    AUC-PR measures the area under the precision-recall curve, which shows the
    trade-off between precision and recall at different attribution thresholds.
    This metric is particularly useful for imbalanced data where the positive class
    (ground truth features) is sparse. A score of 1.0 indicates perfect ranking.

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
        class_label (Optional[int]): Class label to calculate AUC-PR for.
            If None, uses all feature masks regardless of class.
        average (Optional[str]): Method for averaging:
            - 'macro': Average AUC-PR across samples and dimensions
            - 'per_sample': Return AUC-PR for each sample (averaged across dimensions)
            - 'per_dimension': Return AUC-PR for each dimension (averaged across samples)
            - 'per_sample_dimension': Return AUC-PR for each sample-dimension pair
            - None: Return overall AUC-PR (all samples and dimensions combined)

    Returns:
        Union[float, Dict[int, float], Dict[Tuple[int, int], float]]:
            AUC-PR score(s) depending on the averaging method.
    """
    # Extract data using the enhanced helper function - AUC-PR needs continuous values
    data = _extract_feature_data(
        attributions,
        dataset,
        sample_indices,
        dim_indices,
        class_label,
        threshold=None,  # No thresholding for AUC-PR
        needs_feature_values=False,  # AUC-PR only needs masks, not feature values
        allow_continuous=True,
    )

    attributions = data["attributions"]
    masks = data["masks"]
    sample_indices = data["sample_indices"]
    dim_indices = data["dim_indices"]

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

    # Calculate AUC-PR for each sample and dimension
    for i, sample_idx in enumerate(sample_indices):
        for j, dim_idx in enumerate(dim_indices):
            # Get attribution and ground truth for this sample and dimension
            attribution = attributions[i, :, j]
            mask = masks[i, :, j]

            # Skip if all ground truth values are the same (AUC-PR undefined)
            if np.all(mask) or not np.any(mask):
                # For AUC-PR, if all ground truth is True, PR is perfect (1.0)
                # If no ground truth, the baseline is 0.0
                auc = 1.0 if np.all(mask) else 0.0
            else:
                # Calculate precision and recall at each possible threshold
                # Use unique attribution values as thresholds for efficiency
                thresholds = np.unique(attribution)

                # Add one threshold above the maximum to ensure we capture all points
                if thresholds.size > 0:
                    thresholds = np.append(thresholds, thresholds.max() + 1)

                # Count total positive examples in ground truth
                n_pos = np.sum(mask)

                # Initialize lists for precision and recall values
                precision_values = []
                recall_values = []

                # Calculate precision-recall pairs at each threshold
                for threshold in sorted(thresholds, reverse=True):
                    # Get predictions at this threshold
                    pred_pos = attribution >= threshold

                    # Calculate TP, FP
                    tp = np.sum(pred_pos & mask)
                    fp = np.sum(pred_pos & ~mask)

                    # Calculate precision and recall
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
                    recall = tp / n_pos if n_pos > 0 else 0.0

                    precision_values.append(precision)
                    recall_values.append(recall)

                # Convert to numpy arrays
                precision_values = np.array(precision_values)
                recall_values = np.array(recall_values)

                # Sort points by recall (for proper AUC calculation)
                sort_idx = np.argsort(recall_values)
                recall_sorted = recall_values[sort_idx]
                precision_sorted = precision_values[sort_idx]

                # AUC-PR needs to handle duplicate recall values
                # When multiple precision values exist for a recall value, keep the max
                if len(recall_sorted) > 1:
                    unique_recalls, unique_indices = np.unique(
                        recall_sorted, return_index=True
                    )
                    unique_precisions = np.zeros_like(unique_recalls)

                    for k, recall in enumerate(unique_recalls):
                        mask = recall_sorted == recall
                        unique_precisions[k] = np.max(precision_sorted[mask])

                    # Add (0, 1) point if not present (precision=1 at recall=0)
                    if unique_recalls[0] != 0:
                        unique_recalls = np.append([0], unique_recalls)
                        unique_precisions = np.append([1.0], unique_precisions)

                    # Sort by recall to ensure correct AUC calculation
                    sort_idx = np.argsort(unique_recalls)
                    unique_recalls = unique_recalls[sort_idx]
                    unique_precisions = unique_precisions[sort_idx]

                    # Calculate AUC using the trapezoidal rule
                    auc = np.trapz(
                        unique_precisions, unique_recalls
                    )  # TODO: replace with np.trapezoidal if downstream integration permits
                else:
                    # Handle edge case with only one threshold
                    auc = precision_sorted[0]

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


def correlation_score(
    attributions: np.ndarray,
    dataset: Dict,
    sample_indices: Optional[List[int]] = None,
    dim_indices: Optional[List[int]] = None,
    class_label: Optional[int] = None,
    average: Optional[str] = "macro",
    feature_source: str = "isolated",
    absolute: bool = True,
) -> Union[float, Dict[int, float], Dict[Tuple[int, int], float]]:
    """Calculate correlation coefficient between attribution values and ground truth features.

    This metric measures how well attribution values correlate with ground truth feature values
    in regions where ground truth features exist. Higher correlation coefficients indicate that
    the attribution values follow a similar pattern to the feature values.

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
        class_label (Optional[int]): Class label to calculate correlation for.
            If None, uses all feature masks regardless of class.
        average (Optional[str]): Method for averaging:
            - 'macro': Average correlation across samples and dimensions
            - 'per_sample': Return correlation for each sample (averaged across dimensions)
            - 'per_dimension': Return correlation for each dimension (averaged across samples)
            - 'per_sample_dimension': Return correlation for each sample-dimension pair
        feature_source (str): Source of feature values to correlate against:
            - 'isolated': Use values from isolated feature components (dataset["components"][i].features)
            - 'aggregated': Use values from the aggregated time series (dataset["X"])
        absolute (bool): If True, returns the absolute value of the correlation coefficient,
            measuring strength of correlation regardless of direction. If False, returns
            the raw correlation coefficient with sign. Default is True.

    Returns:
        Union[float, Dict[int, float], Dict[Tuple[int, int], float]]:
            Correlation coefficient(s) depending on the averaging method.
    """
    # Extract all necessary data - correlation_score requires feature values
    data = _extract_feature_data(
        attributions,
        dataset,
        sample_indices,
        dim_indices,
        class_label,
        threshold=None,  # No thresholding for correlation
        feature_source=feature_source,
        needs_feature_values=True,  # Correlation needs actual feature values
        allow_continuous=True,
    )

    attr = data["attributions"]
    feat_vals = data["feature_values"]
    masks = data["masks"]
    sample_indices = data["sample_indices"]
    dim_indices = data["dim_indices"]

    n_samples = len(sample_indices)
    n_dimensions = len(dim_indices)

    # Calculate correlations for each sample-dimension pair
    raw_correlations = {}

    for i, sample_idx in enumerate(sample_indices):
        for j, dim_idx in enumerate(dim_indices):
            mask = masks[i, :, j]

            # Skip if no ground truth regions
            if not np.any(mask):
                raw_correlations[(sample_idx, dim_idx)] = 0.0
                continue

            # Get values at mask positions
            attr_values = attr[i, mask, j]
            feat_values = feat_vals[i, mask, j]

            # Skip NaN values
            valid = ~np.isnan(feat_values)
            if np.sum(valid) < 2:  # Need at least 2 points
                raw_correlations[(sample_idx, dim_idx)] = 0.0
                continue

            attr_valid = attr_values[valid]
            feat_valid = feat_values[valid]

            # Calculate correlation
            attr_std = np.std(attr_valid)
            feat_std = np.std(feat_valid)

            if attr_std == 0 and feat_std == 0:
                # Both constant arrays
                if np.allclose(attr_valid[0], feat_valid[0]):
                    raw_correlations[(sample_idx, dim_idx)] = 1.0
                elif np.allclose(attr_valid[0], -feat_valid[0]):
                    raw_correlations[(sample_idx, dim_idx)] = -1.0
                else:
                    raw_correlations[(sample_idx, dim_idx)] = 0.0
            elif attr_std == 0 or feat_std == 0:
                # One array is constant
                raw_correlations[(sample_idx, dim_idx)] = 0.0
            else:
                # Calculate normalized correlation
                attr_norm = (attr_valid - np.mean(attr_valid)) / attr_std
                feat_norm = (feat_valid - np.mean(feat_valid)) / feat_std
                raw_correlations[(sample_idx, dim_idx)] = np.mean(attr_norm * feat_norm)

    # Apply averaging according to the specified method
    results = {}

    if average == "per_sample_dimension":
        for key, corr in raw_correlations.items():
            results[key] = abs(corr) if absolute else corr
    elif average == "per_sample":
        for sample_idx in sample_indices:
            total_corr = 0.0
            for dim_idx in dim_indices:
                corr = raw_correlations.get((sample_idx, dim_idx), 0.0)
                total_corr += abs(corr) if absolute else corr
            results[sample_idx] = total_corr / n_dimensions
    elif average == "per_dimension":
        for dim_idx in dim_indices:
            total_corr = 0.0
            for sample_idx in sample_indices:
                corr = raw_correlations.get((sample_idx, dim_idx), 0.0)
                total_corr += abs(corr) if absolute else corr
            results[dim_idx] = total_corr / n_samples
    else:  # macro
        total_corr = 0.0
        for corr in raw_correlations.values():
            total_corr += abs(corr) if absolute else corr
        results = total_corr / (n_samples * n_dimensions)

    return results


def _extract_feature_data(
    attributions: np.ndarray,
    dataset: Dict,
    sample_indices: Optional[List[int]] = None,
    dim_indices: Optional[List[int]] = None,
    class_label: Optional[int] = None,
    threshold: Optional[float] = None,
    feature_source: str = "isolated",
    needs_feature_values: bool = False,
    allow_continuous: bool = False,
) -> Dict[str, Any]:
    """Extract attribution values, feature values, and masks for metric calculations.

    This helper function prepares all necessary data for feature attribution metrics,
    serving as a central extraction point for all metrics in the module.

    Args:
        attributions (np.ndarray): Feature attribution values.
        dataset (Dict): Dataset dictionary returned by TimeSeriesBuilder.build().
        sample_indices (Optional[List[int]]): Sample indices to include.
        dim_indices (Optional[List[int]]): Dimension indices to include.
        class_label (Optional[int]): Class label to consider for evaluation.
        threshold (Optional[float]): Threshold for binarizing attribution values.
        feature_source (str): Source of feature values to extract:
            - "isolated": Use values from isolated feature components.
            - "aggregated": Use values from the aggregated time series.
        needs_feature_values (bool): If True, extract actual feature values from the dataset.
            If False, only extract binary masks (more efficient for most metrics).
        allow_continuous (bool): If True, allow continuous attribution values even without a threshold.
            Used for metrics like AUC-ROC, AUC-PR, and NAC that work with continuous values.

    Returns:
        Dict[str, Any]: Dictionary containing:
            - "attributions": Feature attribution values [n_samples, n_timesteps, n_dimensions]
            - "feature_values": Ground truth feature values if needs_feature_values=True,
                               otherwise None
            - "masks": Ground truth masks [n_samples, n_timesteps, n_dimensions]
            - "sample_indices": List of sample indices
            - "dim_indices": List of dimension indices
    """
    # We need to set allow_continuous=True here to prevent validation errors
    # for continuous attributions in _validate_and_prepare_inputs
    allow_continuous_validation = True
    attributions, ground_truth_by_dim, sample_indices, dim_indices = (
        _validate_and_prepare_inputs(
            attributions,
            dataset,
            sample_indices,
            dim_indices,
            None,  # Don't binarize in _validate_and_prepare_inputs
            class_label,
            allow_continuous=allow_continuous_validation,
        )
    )

    n_samples = len(sample_indices)
    n_dimensions = len(dim_indices)
    n_timesteps = attributions.shape[1]

    # Create output arrays
    masks = np.zeros((n_samples, n_timesteps, n_dimensions), dtype=bool)

    # Fill mask array from ground_truth_by_dim
    for j, dim_idx in enumerate(dim_indices):
        dim_mask = ground_truth_by_dim[dim_idx]
        for i in range(n_samples):
            masks[i, :, j] = dim_mask[i]

    # Binarize attributions if threshold is provided
    if threshold is not None:
        # Create a binary copy for metrics that need boolean input
        attributions = _binarize_attributions(attributions, threshold)
    # For metrics that need boolean values (precision, recall, f1) but don't allow continuous values
    elif not (allow_continuous or needs_feature_values) and not np.issubdtype(
        attributions.dtype, np.bool_
    ):
        # For metrics that need boolean values but continuous values were provided
        # without a threshold, raise an error with the original error message pattern
        # to match existing tests
        raise ValueError(
            "Attribution must be boolean type when no threshold was provided."
        )

    # Only extract feature values if needed
    feature_values = None
    if needs_feature_values:
        feature_values = np.full((n_samples, n_timesteps, n_dimensions), np.nan)

        # Fill feature_values based on the source
        if feature_source == "aggregated":
            # Get data format from metadata
            data_format = dataset.get("metadata", {}).get(
                "data_format", "channels_first"
            )
            X = dataset["X"]

            # Ensure X is in channels_last format
            if data_format == "channels_first":
                X = np.transpose(X, (0, 2, 1))

            # Extract values for each sample and dimension
            for i, sample_idx in enumerate(sample_indices):
                for j, dim_idx in enumerate(dim_indices):
                    feature_values[i, :, j] = X[sample_idx, :, dim_idx]

        elif (
            feature_source == "isolated"
            and "components" in dataset
            and len(dataset["components"]) > 0
        ):
            # Extract from isolated components
            for i, sample_idx in enumerate(sample_indices):
                sample_components = dataset["components"][sample_idx]
                if (
                    hasattr(sample_components, "features")
                    and sample_components.features
                ):
                    # Process each dimension
                    for j, dim_idx in enumerate(dim_indices):
                        # Initialize with zeros where features will be added
                        feature_mask = masks[i, :, j]
                        if np.any(feature_mask):
                            feature_values[i, feature_mask, j] = 0.0

                        # Find and combine all features for this dimension
                        for (
                            feature_name,
                            feature_vals,
                        ) in sample_components.features.items():
                            dim_match = f"_dim{dim_idx}" in feature_name or (
                                dim_idx == 0 and "_dim" not in feature_name
                            )
                            if dim_match:
                                # Copy only non-NaN values
                                valid = ~np.isnan(feature_vals)
                                if np.any(valid):
                                    # Add feature values (treating NaN as 0)
                                    temp_vals = feature_vals.copy()
                                    temp_vals[~valid] = 0
                                    feature_values[i, valid, j] += temp_vals[valid]

    return {
        "attributions": attributions,
        "feature_values": feature_values,
        "masks": masks,
        "sample_indices": sample_indices,
        "dim_indices": dim_indices,
    }
