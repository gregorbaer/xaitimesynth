"""Metrics for evaluating feature attributions against ground truth."""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


# TODO: check whether we need binarizations and tresholds after stablizing metrics
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

    Intuition: Measures if important regions receive statistically higher attribution values than would be expected by chance.
    Answers: Are my attribution values significantly elevated in the regions that matter?

    Note: NAC is another name for Normalised Scanpath Saliency (NSS) [1], which is used in
    another context to compare binary ground truth masks of where people look in an image (obtained via eye-tracking)
    to the predicted saliency map of where we think they are most likely to focus their
    visual attention. We use the name NAC here to match the domain of XAI validation instead of eye-tracking.

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

    References:
        [1] Peters, R. J., Iyer, A., Itti, L., & Koch, C. (2005). Components of bottom-up gaze allocation in natural images. Vision research, 45(18), 2397-2416.
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
    normalize: bool = False,
) -> Union[float, Dict[int, float], Dict[Tuple[int, int], float]]:
    """Calculate AUC-ROC score for feature attributions.

    AUC-ROC measures how well the attribution values discriminate between
    important and non-important features across all possible threshold values.
    A score of 0.5 indicates random performance, while 1.0 indicates perfect
    discrimination.

    Intuition: Measures discrimination ability across all possible threshold values, with 0.5 indicating random performance.
    Answers: How well can my attributions separate important from unimportant timesteps?

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
        normalize (bool): If True, normalize the score to represent relative improvement
            over random (0.5). Calculated as (AUC-ROC - 0.5) / 0.5. Default is False.

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
                auc = np.trapezoid(tpr_values, fpr_values)
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

                        auc = np.trapezoid(unique_tprs, unique_fprs)

            # Normalize if requested
            if normalize:
                # Normalize relative to random baseline (0.5)
                # (AUC - 0.5) / (1.0 - 0.5) = (AUC - 0.5) / 0.5
                # This handles the case auc == 0.5 correctly (results in 0.0)
                auc = (auc - 0.5) / 0.5

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
    normalize: bool = False,
) -> Union[float, Dict[int, float], Dict[Tuple[int, int], float]]:
    """Calculate AUC-PR score (Area Under the Precision-Recall Curve) for feature attributions.

    AUC-PR measures the area under the precision-recall curve, which shows the
    trade-off between precision and recall at different attribution thresholds.
    This metric is particularly useful for imbalanced data where the positive class
    (ground truth features) is sparse. A score of 1.0 indicates perfect ranking.
    The baseline (random) score is equal to the prevalence (fraction of timesteps
    covered by the ground truth mask) of the positive class.

    Intuition: Measures precision-recall trade-off across thresholds, particularly useful for imbalanced data with sparse features.
    Answers: How well do my attributions maintain precision while finding all important timesteps?

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
        normalize (bool): If True, normalize the score to represent relative improvement
            over random (prevalence). Calculated as (AUC-PR - prevalence) / (1 - prevalence).
            Returns 0 if prevalence is 1 (to avoid division by zero). Default is False.

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

            # Calculate prevalence (positive class fraction)
            n_pos = np.sum(mask)
            n_total = mask.size
            prevalence = n_pos / n_total if n_total > 0 else 0.0

            # Skip if all ground truth values are the same (AUC-PR undefined)
            if n_pos == n_total or n_pos == 0:
                # If all ground truth is True, raw AUC-PR is 1.0
                # If no ground truth, raw AUC-PR is 0.0 (or undefined, treat as 0)
                auc = 1.0 if n_pos == n_total else 0.0
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
                    auc = np.trapezoid(unique_precisions, unique_recalls)
                else:
                    # Handle edge case with only one threshold
                    auc = precision_sorted[0]

            # Normalize if requested
            if normalize:
                # Avoid division by zero; max improvement is not well-defined
                if prevalence == 1.0:
                    auc = 0.0
                else:
                    # Normalize relative to random baseline (prevalence)
                    # (AUC - prevalence) / (1.0 - prevalence)
                    # This handles auc == prevalence correctly (results in 0.0)
                    auc = (auc - prevalence) / (1.0 - prevalence)

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


def relevance_mass_accuracy(
    attributions: np.ndarray,
    dataset: Dict,
    sample_indices: Optional[List[int]] = None,
    dim_indices: Optional[List[int]] = None,
    class_label: Optional[int] = None,
    average: str = "macro",
) -> Union[float, Dict[int, float], Dict[Tuple[int, int], float]]:
    """Compute Relevance Mass Accuracy for feature attributions.

    The relevance mass accuracy[1] is the ratio of the sum of attributions within the ground truth mask
    to the sum of all attributions, for each sample and dimension. This measures how much "mass"
    the explanation method gives to pixels (timesteps) within the ground truth.

    Args:
        attributions (np.ndarray): Attribution values (any shape supported by _validate_and_prepare_inputs).
        dataset (Dict): Dataset dictionary returned by TimeSeriesBuilder.build().
        sample_indices (Optional[List[int]]): Sample indices to include.
        dim_indices (Optional[List[int]]): Dimension indices to include.
        class_label (Optional[int]): Class label to calculate the metric for.
        average (str): Averaging method: 'macro', 'per_sample', 'per_dimension', 'per_sample_dimension'.

    Returns:
        Union[float, Dict[int, float], Dict[Tuple[int, int], float]]: Relevance mass accuracy score(s).

    References:
        [1] Arras, L., Osman, A., & Samek, W. (2022). CLEVR-XAI: A benchmark dataset for the ground truth evaluation of neural network explanations. Information Fusion, 81, 14-40.
    """
    attributions, ground_truth_by_dim, sample_indices, dim_indices = (
        _validate_and_prepare_inputs(
            attributions,
            dataset,
            sample_indices,
            dim_indices,
            threshold=None,
            class_label=class_label,
            allow_continuous=True,
        )
    )

    n_samples = len(sample_indices)
    n_dimensions = len(dim_indices)
    n_timesteps = attributions.shape[1]

    # Prepare mask array: [n_samples, n_timesteps, n_dimensions]
    masks = np.zeros((n_samples, n_timesteps, n_dimensions), dtype=bool)
    for j, dim_idx in enumerate(dim_indices):
        dim_mask = ground_truth_by_dim[dim_idx]
        for i in range(n_samples):
            masks[i, :, j] = dim_mask[i]

    results = {}
    for i, sample_idx in enumerate(sample_indices):
        for j, dim_idx in enumerate(dim_indices):
            attr = attributions[i, :, j]
            mask = masks[i, :, j]
            r_within = np.sum(attr[mask])
            r_total = np.sum(attr)
            score = r_within / r_total if r_total > 0 else 0.0
            results[(sample_idx, dim_idx)] = score

    if average == "macro":
        return np.mean(list(results.values())) if results else 0.0
    elif average == "per_sample":
        sample_results = {}
        for sample_idx in sample_indices:
            vals = [results[(sample_idx, d)] for d in dim_indices]
            sample_results[sample_idx] = np.mean(vals) if vals else 0.0
        return sample_results
    elif average == "per_dimension":
        dim_results = {}
        for dim_idx in dim_indices:
            vals = [results[(s, dim_idx)] for s in sample_indices]
            dim_results[dim_idx] = np.mean(vals) if vals else 0.0
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


def relevance_rank_accuracy(
    attributions: np.ndarray,
    dataset: Dict,
    sample_indices: Optional[List[int]] = None,
    dim_indices: Optional[List[int]] = None,
    class_label: Optional[int] = None,
    average: str = "macro",
) -> Union[float, Dict[int, float], Dict[Tuple[int, int], float]]:
    """Compute Relevance Rank Accuracy for feature attributions.

    The relevance rank accuracy[1] measures the fraction of the K highest attribution values
    that fall within the ground truth mask, where K is the number of ground truth positives.

    Args:
        attributions (np.ndarray): Attribution values (any shape supported by _validate_and_prepare_inputs).
        dataset (Dict): Dataset dictionary returned by TimeSeriesBuilder.build().
        sample_indices (Optional[List[int]]): Sample indices to include.
        dim_indices (Optional[List[int]]): Dimension indices to include.
        class_label (Optional[int]): Class label to calculate the metric for.
        average (str): Averaging method: 'macro', 'per_sample', 'per_dimension', 'per_sample_dimension'.

    Returns:
        Union[float, Dict[int, float], Dict[Tuple[int, int], float]]: Relevance rank accuracy score(s).

    References:
        [1] Arras, L., Osman, A., & Samek, W. (2022). CLEVR-XAI: A benchmark dataset for the ground truth evaluation of neural network explanations. Information Fusion, 81, 14-40.
    """
    attributions, ground_truth_by_dim, sample_indices, dim_indices = (
        _validate_and_prepare_inputs(
            attributions,
            dataset,
            sample_indices,
            dim_indices,
            threshold=None,
            class_label=class_label,
            allow_continuous=True,
        )
    )

    n_samples = len(sample_indices)
    n_dimensions = len(dim_indices)
    n_timesteps = attributions.shape[1]

    # Prepare mask array: [n_samples, n_timesteps, n_dimensions]
    masks = np.zeros((n_samples, n_timesteps, n_dimensions), dtype=bool)
    for j, dim_idx in enumerate(dim_indices):
        dim_mask = ground_truth_by_dim[dim_idx]
        for i in range(n_samples):
            masks[i, :, j] = dim_mask[i]

    results = {}
    for i, sample_idx in enumerate(sample_indices):
        for j, dim_idx in enumerate(dim_indices):
            attr = attributions[i, :, j]
            mask = masks[i, :, j]
            K = int(np.sum(mask))
            if K == 0:
                score = 0.0
            else:
                # Get indices of top-K attributions
                topk_indices = np.argpartition(-attr, K - 1)[:K]
                # Count how many of these indices are in the ground truth mask
                n_in_mask = np.sum(mask[topk_indices])
                score = n_in_mask / K
            results[(sample_idx, dim_idx)] = score

    if average == "macro":
        return np.mean(list(results.values())) if results else 0.0
    elif average == "per_sample":
        sample_results = {}
        for sample_idx in sample_indices:
            vals = [results[(sample_idx, d)] for d in dim_indices]
            sample_results[sample_idx] = np.mean(vals) if vals else 0.0
        return sample_results
    elif average == "per_dimension":
        dim_results = {}
        for dim_idx in dim_indices:
            vals = [results[(s, dim_idx)] for s in sample_indices]
            dim_results[dim_idx] = np.mean(vals) if vals else 0.0
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


def pointing_game(
    attributions: np.ndarray,
    dataset: Dict,
    sample_indices: Optional[List[int]] = None,
    dim_indices: Optional[List[int]] = None,
    class_label: Optional[int] = None,
    average: str = "macro",
) -> Union[float, Dict[int, float], Dict[Tuple[int, int], float]]:
    """Compute Pointing Game accuracy for feature attributions.

    The Pointing Game[1] evaluates whether the point of maximal attribution falls
    within the ground truth mask. For each sample, it returns 1 if the highest
    attributed timestep is within the ground truth region, 0 otherwise.

    This is a simple but intuitive metric: does the explanation correctly identify
    at least one important location as most salient?

    Intuition: Binary check whether the highest-attributed point is on target.
    Answers: Does my explanation point to the right place?

    Args:
        attributions (np.ndarray): Attribution values (any shape supported by _validate_and_prepare_inputs).
        dataset (Dict): Dataset dictionary returned by TimeSeriesBuilder.build().
        sample_indices (Optional[List[int]]): Sample indices to include.
        dim_indices (Optional[List[int]]): Dimension indices to include.
        class_label (Optional[int]): Class label to calculate the metric for.
        average (str): Averaging method: 'macro', 'per_sample', 'per_dimension', 'per_sample_dimension'.

    Returns:
        Union[float, Dict[int, float], Dict[Tuple[int, int], float]]:
            Pointing game accuracy (0 or 1 per sample-dimension, averaged according to `average`).

    References:
        [1] Zhang, J., Bargal, S. A., Lin, Z., Brber, J., Shen, X., & Sclaroff, S. (2018).
            Top-down neural attention by excitation backprop. International Journal of
            Computer Vision, 126(10), 1084-1102.
    """
    attributions, ground_truth_by_dim, sample_indices, dim_indices = (
        _validate_and_prepare_inputs(
            attributions,
            dataset,
            sample_indices,
            dim_indices,
            threshold=None,
            class_label=class_label,
            allow_continuous=True,
        )
    )

    n_samples = len(sample_indices)
    n_dimensions = len(dim_indices)
    n_timesteps = attributions.shape[1]

    # Prepare mask array: [n_samples, n_timesteps, n_dimensions]
    masks = np.zeros((n_samples, n_timesteps, n_dimensions), dtype=bool)
    for j, dim_idx in enumerate(dim_indices):
        dim_mask = ground_truth_by_dim[dim_idx]
        for i in range(n_samples):
            masks[i, :, j] = dim_mask[i]

    results = {}
    for i, sample_idx in enumerate(sample_indices):
        for j, dim_idx in enumerate(dim_indices):
            attr = attributions[i, :, j]
            mask = masks[i, :, j]

            # Find index of maximum attribution
            max_idx = np.argmax(attr)

            # Check if max attribution is within ground truth mask
            score = 1.0 if mask[max_idx] else 0.0
            results[(sample_idx, dim_idx)] = score

    if average == "macro":
        return np.mean(list(results.values())) if results else 0.0
    elif average == "per_sample":
        sample_results = {}
        for sample_idx in sample_indices:
            vals = [results[(sample_idx, d)] for d in dim_indices]
            sample_results[sample_idx] = np.mean(vals) if vals else 0.0
        return sample_results
    elif average == "per_dimension":
        dim_results = {}
        for dim_idx in dim_indices:
            vals = [results[(s, dim_idx)] for s in sample_indices]
            dim_results[dim_idx] = np.mean(vals) if vals else 0.0
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


def mean_absolute_error(
    attributions: np.ndarray,
    dataset: Dict,
    sample_indices: Optional[List[int]] = None,
    dim_indices: Optional[List[int]] = None,
    class_label: Optional[int] = None,
    average: str = "macro",
) -> Union[float, Dict[int, float], Dict[Tuple[int, int], float]]:
    """Compute Mean Absolute Error between attributions and ground truth mask.

    Treats the binary ground truth mask as a continuous target (0 for non-feature,
    1 for feature timesteps) and computes the mean absolute error between the
    attributions and this target. Lower values indicate better alignment.

    Note: For meaningful comparison, attributions should be normalized to [0, 1] range
    before calling this function.

    Intuition: Measures average absolute deviation from ideal attribution (1 at features, 0 elsewhere).
    Answers: How far are my attributions from the perfect localization on average?

    Args:
        attributions (np.ndarray): Attribution values (any shape supported by _validate_and_prepare_inputs).
            Should be normalized to [0, 1] for meaningful interpretation.
        dataset (Dict): Dataset dictionary returned by TimeSeriesBuilder.build().
        sample_indices (Optional[List[int]]): Sample indices to include.
        dim_indices (Optional[List[int]]): Dimension indices to include.
        class_label (Optional[int]): Class label to calculate the metric for.
        average (str): Averaging method: 'macro', 'per_sample', 'per_dimension', 'per_sample_dimension'.

    Returns:
        Union[float, Dict[int, float], Dict[Tuple[int, int], float]]:
            Mean absolute error (lower is better).
    """
    attributions, ground_truth_by_dim, sample_indices, dim_indices = (
        _validate_and_prepare_inputs(
            attributions,
            dataset,
            sample_indices,
            dim_indices,
            threshold=None,
            class_label=class_label,
            allow_continuous=True,
        )
    )

    n_samples = len(sample_indices)
    n_dimensions = len(dim_indices)
    n_timesteps = attributions.shape[1]

    # Prepare mask array: [n_samples, n_timesteps, n_dimensions]
    masks = np.zeros((n_samples, n_timesteps, n_dimensions), dtype=bool)
    for j, dim_idx in enumerate(dim_indices):
        dim_mask = ground_truth_by_dim[dim_idx]
        for i in range(n_samples):
            masks[i, :, j] = dim_mask[i]

    results = {}
    for i, sample_idx in enumerate(sample_indices):
        for j, dim_idx in enumerate(dim_indices):
            attr = attributions[i, :, j]
            mask = masks[i, :, j].astype(float)  # Convert to 0.0/1.0

            # Compute mean absolute error
            score = np.mean(np.abs(attr - mask))
            results[(sample_idx, dim_idx)] = score

    if average == "macro":
        return np.mean(list(results.values())) if results else 0.0
    elif average == "per_sample":
        sample_results = {}
        for sample_idx in sample_indices:
            vals = [results[(sample_idx, d)] for d in dim_indices]
            sample_results[sample_idx] = np.mean(vals) if vals else 0.0
        return sample_results
    elif average == "per_dimension":
        dim_results = {}
        for dim_idx in dim_indices:
            vals = [results[(s, dim_idx)] for s in sample_indices]
            dim_results[dim_idx] = np.mean(vals) if vals else 0.0
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


def mean_squared_error(
    attributions: np.ndarray,
    dataset: Dict,
    sample_indices: Optional[List[int]] = None,
    dim_indices: Optional[List[int]] = None,
    class_label: Optional[int] = None,
    average: str = "macro",
) -> Union[float, Dict[int, float], Dict[Tuple[int, int], float]]:
    """Compute Mean Squared Error between attributions and ground truth mask.

    Treats the binary ground truth mask as a continuous target (0 for non-feature,
    1 for feature timesteps) and computes the mean squared error between the
    attributions and this target. Lower values indicate better alignment.
    Compared to MAE, MSE penalizes large deviations more heavily.

    Note: For meaningful comparison, attributions should be normalized to [0, 1] range
    before calling this function.

    Intuition: Measures average squared deviation from ideal attribution, penalizing large errors more.
    Answers: How far are my attributions from perfect localization, with emphasis on large mistakes?

    Args:
        attributions (np.ndarray): Attribution values (any shape supported by _validate_and_prepare_inputs).
            Should be normalized to [0, 1] for meaningful interpretation.
        dataset (Dict): Dataset dictionary returned by TimeSeriesBuilder.build().
        sample_indices (Optional[List[int]]): Sample indices to include.
        dim_indices (Optional[List[int]]): Dimension indices to include.
        class_label (Optional[int]): Class label to calculate the metric for.
        average (str): Averaging method: 'macro', 'per_sample', 'per_dimension', 'per_sample_dimension'.

    Returns:
        Union[float, Dict[int, float], Dict[Tuple[int, int], float]]:
            Mean squared error (lower is better).
    """
    attributions, ground_truth_by_dim, sample_indices, dim_indices = (
        _validate_and_prepare_inputs(
            attributions,
            dataset,
            sample_indices,
            dim_indices,
            threshold=None,
            class_label=class_label,
            allow_continuous=True,
        )
    )

    n_samples = len(sample_indices)
    n_dimensions = len(dim_indices)
    n_timesteps = attributions.shape[1]

    # Prepare mask array: [n_samples, n_timesteps, n_dimensions]
    masks = np.zeros((n_samples, n_timesteps, n_dimensions), dtype=bool)
    for j, dim_idx in enumerate(dim_indices):
        dim_mask = ground_truth_by_dim[dim_idx]
        for i in range(n_samples):
            masks[i, :, j] = dim_mask[i]

    results = {}
    for i, sample_idx in enumerate(sample_indices):
        for j, dim_idx in enumerate(dim_indices):
            attr = attributions[i, :, j]
            mask = masks[i, :, j].astype(float)  # Convert to 0.0/1.0

            # Compute mean squared error
            score = np.mean((attr - mask) ** 2)
            results[(sample_idx, dim_idx)] = score

    if average == "macro":
        return np.mean(list(results.values())) if results else 0.0
    elif average == "per_sample":
        sample_results = {}
        for sample_idx in sample_indices:
            vals = [results[(sample_idx, d)] for d in dim_indices]
            sample_results[sample_idx] = np.mean(vals) if vals else 0.0
        return sample_results
    elif average == "per_dimension":
        dim_results = {}
        for dim_idx in dim_indices:
            vals = [results[(s, dim_idx)] for s in sample_indices]
            dim_results[dim_idx] = np.mean(vals) if vals else 0.0
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
