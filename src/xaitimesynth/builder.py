from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .data_structures import TimeSeriesComponents
from .functions import normalize
from .generators import generate_component


class TimeSeriesBuilder:
    """Builder for synthetic time series datasets.

    This class provides a fluent API for building synthetic time series
    datasets with known ground truth features for XAI evaluation.

    The builder uses different fill values for different component types:
    - Features: Typically filled with NaN where the feature doesn't exist
    - Foundation: Typically filled with zeros where no foundation component exists
    - Noise: Typically filled with zeros where no noise component exists

    These fill values control how components are visualized and combined,
    with NaN values being ignored during addition operations.

    Attributes:
        n_timesteps: Length of each time series.
        n_samples: Total number of samples to generate.
        n_dimensions: Number of dimensions in each time series (for multivariate series).
        normalization: Normalization method for the final time series.
        random_state: Random seed for reproducibility.
        rng: Random number generator.
        class_definitions: List of class definitions.
        current_class: Current class being configured.
        feature_fill_value: Value used for non-existent features (default: np.nan).
        foundation_fill_value: Value used for foundation when none exists (default: 0.0).
        noise_fill_value: Value used for noise when none exists (default: 0.0).
    """

    def __init__(
        self,
        n_timesteps: int = 100,
        n_samples: int = 1000,
        n_dimensions: int = 1,
        normalization: str = "zscore",
        random_state: Optional[int] = None,
        normalization_kwargs: Optional[Dict[str, Any]] = {},
        feature_fill_value: Any = np.nan,
        foundation_fill_value: Any = 0.0,
        noise_fill_value: Any = 0.0,
    ):
        """Initialize the time series builder.

        Args:
            n_timesteps: Length of each time series.
            n_samples: Total number of samples to generate.
            n_dimensions: Number of dimensions in each time series. Default is 1 (univariate).
            normalization: Normalization method for the final time series.
                Options: "zscore", "minmax", or "none". Default is "zscore".
            random_state: Random seed for reproducibility.
            normalization_kwargs: Additional parameters for normalization.
            feature_fill_value: Value used for non-existent features (default: np.nan).
                This represents points where a feature doesn't exist. Using NaN makes
                features only appear where they're defined in visualizations.
            foundation_fill_value: Value used for foundation when none exists (default: 0.0).
                Foundation typically affects the entire time series, so zeros represent
                "no contribution" rather than "doesn't exist".
            noise_fill_value: Value used for noise when none exists (default: 0.0).
                Similar to foundation, zeros indicate "no contribution".
        """
        self.n_timesteps = n_timesteps
        self.n_samples = n_samples
        self.n_dimensions = n_dimensions

        # Validate n_dimensions
        if n_dimensions < 1:
            raise ValueError("n_dimensions must be at least 1")

        self.normalization = normalization
        self.normalization_kwargs = normalization_kwargs or {}
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)
        self.feature_fill_value = feature_fill_value
        self.foundation_fill_value = foundation_fill_value
        self.noise_fill_value = noise_fill_value

        # Initialize class definitions and the current class
        self.class_definitions = []
        self.current_class = None

    def for_class(self, class_label: int, weight: float = 1.0) -> "TimeSeriesBuilder":
        """Set the current class for component assignment.

        Args:
            class_label: Label for the class.
            weight: Relative weight of this class in the dataset.

        Returns:
            self for method chaining.
        """
        # Create a new class definition
        class_def = {
            "label": class_label,
            "weight": weight,
            "components": {"foundation": [], "noise": [], "features": []},
        }

        self.class_definitions.append(class_def)
        self.current_class = class_def

        return self

    def _validate_dimensions(self, dimensions: List[int]) -> None:
        """Validate dimension indices against n_dimensions.

        Args:
            dimensions: List of dimension indices to validate.

        Raises:
            ValueError: If any dimension is out of range.
        """
        for d in dimensions:
            if not 0 <= d < self.n_dimensions:
                raise ValueError(
                    f"Dimension {d} is out of range. Valid dimensions are 0 to {self.n_dimensions - 1}."
                )

    def add_signal(
        self,
        component: Dict[str, Any],
        role: str = "foundation",
        dim: Optional[List[int]] = None,
        shared_randomness: bool = False,
    ) -> "TimeSeriesBuilder":
        """Add a signal component to the current class.

        Args:
            component: Component definition dictionary.
            role: Role of the component (foundation, noise).
            dim: List of dimension indices where this signal should be applied.
                 Default is [0] (→ univariate time series if all signals have dim=[0]).
            shared_randomness: If True, the same random pattern will be used across all dimensions.
                               If False, each dimension gets its own random pattern (for stochastic components).

        Returns:
            self for method chaining.
        """
        if self.current_class is None:
            raise ValueError("No class selected. Call for_class() first.")

        if role not in ("foundation", "noise"):
            raise ValueError(f"Invalid role: {role}. Must be 'foundation' or 'noise'.")

        # Default to dimension 0 if not specified (for backward compatibility)
        if dim is None:
            dim = [0]

        # Validate dimensions
        self._validate_dimensions(dim)

        # If shared_randomness is True or only one dimension, store a single component
        if shared_randomness or len(dim) == 1:
            # Store dimension information with the component
            component_with_dim = component.copy()
            component_with_dim["dimensions"] = dim
            component_with_dim["shared_randomness"] = shared_randomness

            self.current_class["components"][role].append(component_with_dim)
        else:
            # For multiple dimensions with different randomness,
            # create separate component entries for each dimension
            for d in dim:
                component_with_dim = component.copy()
                component_with_dim["dimensions"] = [d]  # Single dimension
                component_with_dim["shared_randomness"] = shared_randomness

                self.current_class["components"][role].append(component_with_dim)

        return self

    def add_feature(
        self,
        component: Dict[str, Any],
        start_pct: Optional[float] = None,
        end_pct: Optional[float] = None,
        length_pct: Optional[float] = None,
        random_location: bool = False,
        dim: Optional[List[int]] = None,
        shared_location: bool = True,
        shared_randomness: bool = False,
    ) -> "TimeSeriesBuilder":
        """Add a feature component to the current class.

        Args:
            component: Component definition dictionary.
            start_pct: Start position as percentage of time series length (0-1).
            end_pct: End position as percentage of time series length (0-1).
            length_pct: Length of feature as percentage of time series length (0-1).
            random_location: Whether to place the feature at a random location.
            dim: List of dimension indices where this feature should be applied.
                 Default is [0] for backward compatibility (univariate case).
            shared_location: If True and random_location is True, the same random
                             location will be used across all dimensions.
                             If False, each dimension gets its own random location.
            shared_randomness: If True, the same random pattern will be used across all dimensions.
                               If False, each dimension gets its own random pattern (for stochastic components).

        Returns:
            self for method chaining.
        """
        if self.current_class is None:
            raise ValueError("No class selected. Call for_class() first.")

        # Default to dimension 0 if not specified
        if dim is None:
            dim = [0]

        # Validate dimensions
        self._validate_dimensions(dim)

        # Create feature definition
        feature_def = component.copy()

        # Add location parameters
        if random_location:
            if length_pct is None:
                raise ValueError(
                    "length_pct must be provided when random_location is True"
                )
            if not (0 < length_pct <= 1):
                raise ValueError("length_pct must be between 0 and 1")

            feature_def["random_location"] = True
            feature_def["length_pct"] = length_pct
        else:
            if start_pct is None or end_pct is None:
                raise ValueError(
                    "start_pct and end_pct must be provided when random_location is False"
                )
            if not (0 <= start_pct < 1 and 0 < end_pct <= 1 and start_pct < end_pct):
                raise ValueError(
                    "Invalid start_pct or end_pct. Must be between 0 and 1, with start_pct < end_pct"
                )

            feature_def["random_location"] = False
            feature_def["start_pct"] = start_pct
            feature_def["end_pct"] = end_pct

        # If shared_randomness is True or only one dimension, add a single feature
        if shared_randomness or len(dim) == 1:
            # Add dimension information
            feature_def["dimensions"] = dim
            feature_def["shared_location"] = shared_location
            feature_def["shared_randomness"] = shared_randomness

            self.current_class["components"]["features"].append(feature_def)
        else:
            # For multiple dimensions with different randomness,
            # create separate feature entries for each dimension
            for d in dim:
                feature_single_dim = feature_def.copy()
                feature_single_dim["dimensions"] = [d]  # Single dimension
                feature_single_dim["shared_location"] = (
                    shared_location  # Preserve shared_location setting
                )
                feature_single_dim["shared_randomness"] = shared_randomness

                self.current_class["components"]["features"].append(feature_single_dim)

        return self

    def _generate_component_vector(
        self, component_def: Dict[str, Any], feature_length: Optional[int] = None
    ) -> np.ndarray:
        """Generate a component vector based on its definition.

        Args:
            component_def: Component definition dictionary.
            feature_length: Length of the feature in timesteps.
                Only used for feature components.

        Returns:
            Component vector.
        """
        component_type = component_def["type"]
        component_params = component_def.copy()
        component_params.pop("type")

        # Remove dimension information if present
        component_params.pop("dimensions", None)
        component_params.pop("shared_location", None)
        component_params.pop("shared_randomness", None)

        # If it's a feature, add the feature_length parameter
        if feature_length is not None:
            component_params["length"] = feature_length

        return generate_component(
            component_type, self.n_timesteps, self.rng, **component_params
        )

    def _generate_feature_vector(
        self, feature_def: Dict[str, Any], dim_index: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a feature vector and its mask.

        Args:
            feature_def: Feature definition dictionary.
            dim_index: The index in the dimensions list to use for location
                       determination. Only used when shared_location is False.

        Returns:
            Tuple of (feature vector, boolean mask).
        """
        # Initialize with feature_fill_value instead of fill_value
        feature = np.full(self.n_timesteps, self.feature_fill_value)
        mask = np.zeros(self.n_timesteps, dtype=bool)

        # Determine feature location
        if feature_def["random_location"]:
            length_pct = feature_def["length_pct"]
            feature_length = max(1, int(length_pct * self.n_timesteps))

            # Generate random start position
            # If dim_index is provided and shared_location is False, use different
            # random locations for each dimension
            if dim_index is not None and not feature_def["shared_location"]:
                # Use dim_index to get a different random seed for each dimension
                dim_rng = np.random.RandomState(self.rng.randint(0, 2**32 - 1))
                max_start = self.n_timesteps - feature_length
                start_idx = dim_rng.randint(0, max_start + 1)
            else:
                max_start = self.n_timesteps - feature_length
                start_idx = self.rng.randint(0, max_start + 1)

            end_idx = start_idx + feature_length
        else:
            start_pct = feature_def["start_pct"]
            end_pct = feature_def["end_pct"]

            start_idx = int(start_pct * self.n_timesteps)
            end_idx = int(end_pct * self.n_timesteps)

            # Ensure at least one timestep is selected
            if start_idx == end_idx:
                end_idx = start_idx + 1

        # Mark the feature region
        mask[start_idx:end_idx] = True

        # Generate the feature vector
        feature_params = feature_def.copy()
        feature_type = feature_params.pop("type")

        # Remove location parameters
        feature_params.pop("random_location", None)
        feature_params.pop("start_pct", None)
        feature_params.pop("end_pct", None)
        feature_params.pop("length_pct", None)
        feature_params.pop("dimensions", None)
        feature_params.pop("shared_location", None)

        # Generate the component for the feature length
        feature_length = end_idx - start_idx
        feature_values = generate_component(
            feature_type,
            self.n_timesteps,
            self.rng,
            length=feature_length,
            **feature_params,
        )

        # Place the feature in the correct location
        feature[start_idx:end_idx] = feature_values

        return feature, mask

    def build(
        self, return_components: bool = True, train_test_split: Optional[float] = None
    ) -> Dict[str, Any]:
        """Build the dataset based on the class definitions.

        Args:
            return_components: Whether to return the component vectors.
            train_test_split: If not None, fraction of data to use for training.

        Returns:
            Dictionary containing the generated dataset.
        """
        # Validate class definitions
        if not self.class_definitions:
            raise ValueError(
                "No class definitions provided. Call for_class() at least once."
            )

        # Normalize class weights
        weights = np.array([cd["weight"] for cd in self.class_definitions])
        weights = weights / weights.sum()

        # Determine class distribution
        class_counts = self.rng.multinomial(self.n_samples, weights)

        # Initialize arrays - now with n_dimensions
        X = np.zeros((self.n_samples, self.n_timesteps, self.n_dimensions))
        y = np.zeros(self.n_samples, dtype=int)
        all_components = []
        feature_masks = {}

        # Generate data for each class
        sample_idx = 0
        for class_def, count in zip(self.class_definitions, class_counts):
            class_label = class_def["label"]

            for _ in range(count):
                # Initialize arrays for this sample with appropriate fill values per dimension
                foundation = np.full(
                    (self.n_timesteps, self.n_dimensions), self.foundation_fill_value
                )
                noise = np.full(
                    (self.n_timesteps, self.n_dimensions), self.noise_fill_value
                )
                features_dict = {}
                feature_masks_dict = {}

                # Add base structure components
                for base_def in class_def["components"]["foundation"]:
                    base_vector = self._generate_component_vector(base_def)

                    # Apply to specified dimensions
                    for dim_idx in base_def["dimensions"]:
                        foundation[:, dim_idx] = self._add_vector_handling_nans(
                            foundation[:, dim_idx], base_vector
                        )

                # Add noise components
                for noise_def in class_def["components"]["noise"]:
                    noise_vector = self._generate_component_vector(noise_def)

                    # Apply to specified dimensions
                    for dim_idx in noise_def["dimensions"]:
                        noise[:, dim_idx] = self._add_vector_handling_nans(
                            noise[:, dim_idx], noise_vector
                        )

                # Initialize aggregated time series
                aggregated = foundation.copy()

                # Add features
                for feature_idx, feature_def in enumerate(
                    class_def["components"]["features"]
                ):
                    # Get dimensions for this feature
                    feature_dims = feature_def["dimensions"]

                    # For each dimension in the feature
                    for i, dim_idx in enumerate(feature_dims):
                        # Generate feature vector - if shared_location is True, all dimensions
                        # will use the same location, otherwise pass the dimension index
                        dim_index = None if feature_def["shared_location"] else i
                        feature, mask = self._generate_feature_vector(
                            feature_def, dim_index
                        )

                        # Add to aggregated series for this dimension
                        aggregated[:, dim_idx] = self._add_vector_handling_nans(
                            aggregated[:, dim_idx], feature
                        )

                        # Store components
                        feature_name = (
                            f"feature_{feature_idx}_{feature_def['type']}_dim{dim_idx}"
                        )
                        if feature_name not in features_dict:
                            features_dict[feature_name] = feature
                            feature_masks_dict[feature_name] = mask

                        # Add to global feature masks
                        feature_key = f"class_{class_label}_{feature_name}"
                        if feature_key not in feature_masks:
                            feature_masks[feature_key] = np.zeros(
                                (self.n_samples, self.n_timesteps), dtype=bool
                            )

                        feature_masks[feature_key][sample_idx] = mask

                # Add noise to aggregated series (each dimension separately)
                for dim_idx in range(self.n_dimensions):
                    aggregated[:, dim_idx] = self._add_vector_handling_nans(
                        aggregated[:, dim_idx], noise[:, dim_idx]
                    )

                # Normalize if required (apply to each dimension separately)
                for dim_idx in range(self.n_dimensions):
                    aggregated[:, dim_idx] = normalize(
                        aggregated[:, dim_idx],
                        method=self.normalization,
                        **self.normalization_kwargs,
                    )

                # Store the result
                X[sample_idx] = aggregated
                y[sample_idx] = class_label

                # Store components if needed
                if return_components:
                    all_components.append(
                        TimeSeriesComponents(
                            foundation=foundation,
                            noise=noise,
                            features=features_dict,
                            feature_masks=feature_masks_dict,
                            aggregated=aggregated,
                        )
                    )

                sample_idx += 1

        # Prepare result dictionary
        result = {
            "X": X,
            "y": y,
            "feature_masks": feature_masks,
            "metadata": {
                "n_samples": self.n_samples,
                "n_timesteps": self.n_timesteps,
                "n_dimensions": self.n_dimensions,
                "class_definitions": self.class_definitions,
                "normalize": self.normalization,
                "normalization_kwargs": self.normalization_kwargs,
                "random_state": self.random_state,
            },
        }

        if return_components:
            result["components"] = all_components

        # Split into train and test if requested
        if train_test_split is not None:
            # Shuffle indices
            indices = np.arange(self.n_samples)
            self.rng.shuffle(indices)

            # Split point
            split_idx = int(self.n_samples * train_test_split)

            # Train indices
            train_indices = indices[:split_idx]
            result["X_train"] = X[train_indices]
            result["y_train"] = y[train_indices]

            # Test indices
            test_indices = indices[split_idx:]
            result["X_test"] = X[test_indices]
            result["y_test"] = y[test_indices]

            # Split feature masks
            for key in feature_masks:
                result[f"feature_masks_train_{key}"] = feature_masks[key][train_indices]
                result[f"feature_masks_test_{key}"] = feature_masks[key][test_indices]

            # Split components if needed
            if return_components:
                result["components_train"] = [all_components[i] for i in train_indices]
                result["components_test"] = [all_components[i] for i in test_indices]

        return result

    def to_df(
        self,
        dataset: Dict[str, Any],
        samples: Optional[List[int]] = None,
        classes: Optional[List[int]] = None,
        components: Optional[List[str]] = None,
        dimensions: Optional[List[int]] = None,
        format_classes: bool = False,
    ) -> pd.DataFrame:
        """Convert time series dataset to a long-format pandas DataFrame.

        This method creates a DataFrame with one row per timestep per component per sample per dimension,
        suitable for detailed analysis and visualization.

        Args:
            dataset: Dataset returned by build().
            samples: List of sample indices to include. If None, includes all samples.
            classes: List of class labels to include. If None, includes all classes.
            components: List of component types to include. Default includes all:
                ["aggregated", "foundation", "noise", "features"]
            dimensions: List of dimension indices to include. If None, includes all dimensions.
            format_classes: If True, format class labels as "Class X".
                Otherwise use numeric labels.

        Returns:
            pd.DataFrame: Long-format DataFrame with columns for timesteps, values,
                class labels, sample indices, component types, and dimensions.
        """
        # Default components to include (use programming-friendly names)
        default_components = ["aggregated", "foundation", "noise", "features"]
        components_to_include = (
            components if components is not None else default_components
        )

        # Get number of dimensions from metadata or infer from data shape
        n_dims = dataset.get("metadata", {}).get("n_dimensions", 1)
        if n_dims == 1 and len(dataset["X"].shape) == 3:
            n_dims = dataset["X"].shape[2]

        # Default dimensions to include
        if dimensions is None:
            dimensions = list(range(n_dims))
        else:
            # Validate dimensions
            for d in dimensions:
                if not 0 <= d < n_dims:
                    raise ValueError(
                        f"Dimension {d} is out of range (0 to {n_dims - 1})."
                    )

        # Filter by class if specified
        if classes is not None:
            class_indices = np.where(np.isin(dataset["y"], classes))[0]
        else:
            class_indices = np.arange(len(dataset["y"]))

        # Filter by sample if specified
        if samples is not None:
            sample_indices = np.array(samples)
            # Ensure sample indices are within class_indices
            sample_indices = np.intersect1d(sample_indices, class_indices)
        else:
            sample_indices = class_indices

        # Initialize list to hold DataFrames
        dfs = []

        # Process aggregated time series (formerly "Complete Series")
        if "aggregated" in components_to_include:
            # Get all selected samples at once
            X_selected = dataset["X"][sample_indices]
            n_samples = len(sample_indices)
            n_timesteps = X_selected.shape[1]

            # For each dimension
            for dim_idx in dimensions:
                # Create time indices for all samples
                times = np.arange(n_timesteps)

                # Create sample indices repeated for each timestep
                sample_idx_rep = np.repeat(sample_indices, n_timesteps)
                time_idx_rep = np.tile(times, n_samples)

                # Create values array for this dimension
                if len(X_selected.shape) == 3:  # Multivariate case
                    values = X_selected[:, :, dim_idx].flatten()
                else:  # Univariate case (backward compatibility)
                    values = X_selected.flatten()

                # Get class labels
                classes_rep = np.repeat(dataset["y"][sample_indices], n_timesteps)
                if format_classes:
                    class_labels = np.array([f"Class {c}" for c in classes_rep])
                else:
                    class_labels = classes_rep

                # Create DataFrame
                df_agg = pd.DataFrame(
                    {
                        "time": time_idx_rep,
                        "value": values,
                        "class": class_labels,
                        "sample": sample_idx_rep,
                        "component": "aggregated",
                        "feature": None,
                        "dim": dim_idx,
                    }
                )

                dfs.append(df_agg)

        # Process components if available
        if "components" in dataset:
            for component_name in ["foundation", "noise"]:
                if component_name in components_to_include:
                    for dim_idx in dimensions:
                        comp_data = []
                        valid_samples = []

                        # Collect data from all samples
                        for i, idx in enumerate(sample_indices):
                            comp = dataset["components"][idx]
                            if (
                                hasattr(comp, component_name)
                                and getattr(comp, component_name) is not None
                            ):
                                comp_array = getattr(comp, component_name)
                                # Check if component has dimension data
                                if (
                                    len(comp_array.shape) == 2
                                    and comp_array.shape[1] > dim_idx
                                ):
                                    comp_data.append(comp_array[:, dim_idx])
                                    valid_samples.append(idx)
                                elif len(comp_array.shape) == 1 and dim_idx == 0:
                                    # Backward compatibility - 1D array for univariate case
                                    comp_data.append(comp_array)
                                    valid_samples.append(idx)

                        if comp_data:
                            # Stack component data
                            comp_array = np.vstack(comp_data)
                            n_valid = len(valid_samples)
                            n_timesteps = comp_array.shape[1]

                            # Create indices
                            sample_idx_rep = np.repeat(valid_samples, n_timesteps)
                            time_idx_rep = np.tile(np.arange(n_timesteps), n_valid)

                            # Get class labels
                            classes_rep = np.repeat(
                                dataset["y"][valid_samples], n_timesteps
                            )
                            if format_classes:
                                class_labels = np.array(
                                    [f"Class {c}" for c in classes_rep]
                                )
                            else:
                                class_labels = classes_rep

                            # Create DataFrame
                            df_comp = pd.DataFrame(
                                {
                                    "time": time_idx_rep,
                                    "value": comp_array.flatten(),
                                    "class": class_labels,
                                    "sample": sample_idx_rep,
                                    "component": component_name,
                                    "feature": None,
                                    "dim": dim_idx,
                                }
                            )

                            dfs.append(df_comp)

            # Process features - features need special handling since they're stored in a dict
            if "features" in components_to_include:
                feature_dfs = []

                for idx in sample_indices:
                    comp = dataset["components"][idx]
                    if hasattr(comp, "features") and comp.features:
                        for feature_name, feature_values in comp.features.items():
                            # Extract dimension from feature name (if present)
                            if "_dim" in feature_name:
                                parts = feature_name.split("_dim")
                                dim_idx = int(parts[-1])
                                if dim_idx not in dimensions:
                                    continue
                            else:
                                # For backward compatibility, assume dimension 0
                                dim_idx = 0
                                if dim_idx not in dimensions:
                                    continue

                            # Get class label
                            class_label = dataset["y"][idx]
                            if format_classes:
                                class_str = f"Class {class_label}"
                            else:
                                class_str = class_label

                            # Create feature DataFrame
                            df_feature = pd.DataFrame(
                                {
                                    "time": np.arange(len(feature_values)),
                                    "value": feature_values,
                                    "class": class_str,
                                    "sample": idx,
                                    "component": "features",
                                    "feature": feature_name,
                                    "dim": dim_idx,
                                }
                            )

                            feature_dfs.append(df_feature)

                if feature_dfs:
                    dfs.append(pd.concat(feature_dfs, ignore_index=True))

        # Combine all DataFrames
        if not dfs:
            return pd.DataFrame()

        df = pd.concat(dfs, ignore_index=True)

        # Set up categorical variables for ordered plotting
        components_present = [
            c for c in components_to_include if c in df["component"].unique()
        ]
        df["component"] = pd.Categorical(
            df["component"], categories=components_present, ordered=True
        )

        if format_classes:
            class_labels = sorted(
                df["class"].unique(), key=lambda x: int(x.split()[-1])
            )
            df["class"] = pd.Categorical(
                df["class"], categories=class_labels, ordered=True
            )

        return df

    def _add_vector_handling_nans(
        self, base: np.ndarray, to_add: np.ndarray
    ) -> np.ndarray:
        """Add two vectors while properly handling NaN values.

        When adding vectors that may contain NaN values:
        1. Where both vectors have values (not NaN): Normal addition
        2. Where one vector has NaN: Use the non-NaN value
        3. Where both have NaN: Result remains NaN

        This allows components to only contribute where they're defined.

        Args:
            base: Base vector to add to.
            to_add: Vector to add to the base.

        Returns:
            Combined vector with NaNs handled appropriately.
        """
        # Stack arrays and use nansum for element-wise addition that ignores NaNs
        result = np.nansum(np.stack([base, to_add]), axis=0)

        # Fix case where both values are NaN (nansum would return 0, but we want NaN)
        both_nan = np.isnan(base) & np.isnan(to_add)
        result[both_nan] = np.nan

        return result
