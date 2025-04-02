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

    def add_signal(
        self, component: Dict[str, Any], role: str = "foundation"
    ) -> "TimeSeriesBuilder":
        """Add a signal component to the current class.

        Args:
            component: Component definition dictionary.
            role: Role of the component (foundation, noise).

        Returns:
            self for method chaining.
        """
        if self.current_class is None:
            raise ValueError("No class selected. Call for_class() first.")

        if role not in ("foundation", "noise"):
            raise ValueError(f"Invalid role: {role}. Must be 'foundation' or 'noise'.")

        self.current_class["components"][role].append(component)

        return self

    def add_feature(
        self,
        component: Dict[str, Any],
        start_pct: Optional[float] = None,
        end_pct: Optional[float] = None,
        length_pct: Optional[float] = None,
        random_location: bool = False,
    ) -> "TimeSeriesBuilder":
        """Add a feature component to the current class.

        Args:
            component: Component definition dictionary.
            start_pct: Start position as percentage of time series length (0-1).
            end_pct: End position as percentage of time series length (0-1).
            length_pct: Length of feature as percentage of time series length (0-1).
            random_location: Whether to place the feature at a random location.

        Returns:
            self for method chaining.
        """
        if self.current_class is None:
            raise ValueError("No class selected. Call for_class() first.")

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

        self.current_class["components"]["features"].append(feature_def)

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

        # If it's a feature, add the feature_length parameter
        if feature_length is not None:
            component_params["length"] = feature_length

        return generate_component(
            component_type, self.n_timesteps, self.rng, **component_params
        )

    def _generate_feature_vector(
        self, feature_def: Dict[str, Any]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a feature vector and its mask.

        Args:
            feature_def: Feature definition dictionary.

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

        # Initialize arrays
        X = np.zeros((self.n_samples, self.n_timesteps))
        y = np.zeros(self.n_samples, dtype=int)
        all_components = []
        feature_masks = {}

        # Generate data for each class
        sample_idx = 0
        for class_def, count in zip(self.class_definitions, class_counts):
            class_label = class_def["label"]

            for _ in range(count):
                # Initialize arrays for this sample with appropriate fill values
                foundation = np.full(self.n_timesteps, self.foundation_fill_value)
                noise = np.full(self.n_timesteps, self.noise_fill_value)
                features_dict = {}
                feature_masks_dict = {}

                # Add base structure components
                for base_def in class_def["components"]["foundation"]:
                    base_vector = self._generate_component_vector(base_def)
                    foundation = self._add_vector_handling_nans(foundation, base_vector)

                # Add noise components
                for noise_def in class_def["components"]["noise"]:
                    noise_vector = self._generate_component_vector(noise_def)
                    noise = self._add_vector_handling_nans(noise, noise_vector)

                # Initialize aggregated time series
                aggregated = foundation.copy()

                # Add features
                for feature_idx, feature_def in enumerate(
                    class_def["components"]["features"]
                ):
                    feature, mask = self._generate_feature_vector(feature_def)

                    # Add to aggregated series using helper method
                    aggregated = self._add_vector_handling_nans(aggregated, feature)

                    # Store components
                    feature_name = f"feature_{feature_idx}_{feature_def['type']}"
                    features_dict[feature_name] = feature
                    feature_masks_dict[feature_name] = mask

                    # Add to global feature masks
                    feature_key = f"class_{class_label}_{feature_name}"
                    if feature_key not in feature_masks:
                        feature_masks[feature_key] = np.zeros(
                            (self.n_samples, self.n_timesteps), dtype=bool
                        )

                    feature_masks[feature_key][sample_idx] = mask

                # Add noise to aggregated series
                aggregated = self._add_vector_handling_nans(aggregated, noise)

                # Normalize if required
                aggregated = normalize(
                    aggregated, method=self.normalization, **self.normalization_kwargs
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
        format_classes: bool = True,
    ) -> pd.DataFrame:
        """Convert time series dataset to a long-format pandas DataFrame.

        This method creates a DataFrame with one row per timestep per component per sample,
        suitable for detailed analysis and visualization.

        Args:
            dataset: Dataset returned by build().
            samples: List of sample indices to include. If None, includes all samples.
            classes: List of class labels to include. If None, includes all classes.
            components: List of component types to include. Default includes all:
                ["aggregated", "foundation", "noise", "features"]
            format_classes: If True, format class labels as "Class X".
                Otherwise use numeric labels.

        Returns:
            pd.DataFrame: Long-format DataFrame with columns for timesteps, values,
                class labels, sample indices, and component types.
        """
        # Default components to include (use programming-friendly names)
        default_components = ["aggregated", "foundation", "noise", "features"]
        components_to_include = (
            components if components is not None else default_components
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

            # Create time indices for all samples
            times = np.arange(n_timesteps)

            # Create sample indices repeated for each timestep
            sample_idx_rep = np.repeat(sample_indices, n_timesteps)
            time_idx_rep = np.tile(times, n_samples)

            # Create values array
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
                }
            )

            dfs.append(df_agg)

        # Process components if available
        if "components" in dataset:
            for component_name in ["foundation", "noise"]:
                if component_name in components_to_include:
                    comp_data = []
                    valid_samples = []

                    # Collect data from all samples
                    for i, idx in enumerate(sample_indices):
                        comp = dataset["components"][idx]
                        if (
                            hasattr(comp, component_name)
                            and getattr(comp, component_name) is not None
                        ):
                            comp_data.append(getattr(comp, component_name))
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
                            class_labels = np.array([f"Class {c}" for c in classes_rep])
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
                            }
                        )

                        dfs.append(df_comp)

            # Process features
            if "features" in components_to_include:
                feature_dfs = []

                for idx in sample_indices:
                    comp = dataset["components"][idx]
                    if hasattr(comp, "features") and comp.features:
                        for feature_name, feature_values in comp.features.items():
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
