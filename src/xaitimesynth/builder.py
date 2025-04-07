from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .data_structures import TimeSeriesComponents
from .functions import normalize
from .generators import generate_component


class TimeSeriesBuilder:
    """Builder for synthetic time series datasets with known ground truth for XAI.

    This class provides a fluent API for building synthetic time series datasets with
    known ground truth features for explainable AI (XAI) evaluation.

    The builder creates time series by combining multiple components:
    - Foundation: The base structure of the time series (e.g., random walk, constant)
    - Noise: Random noise added to the time series
    - Features: Discriminative patterns for class separation (e.g., shapelet, peak)

    Terminology:
    - "Signals" refer to either foundation or noise components, added with add_signal()
    - Foundation and noise components differ mainly for visualization purposes
    - Features are components that distinguish between classes, added with add_feature()

    Component flexibility:
    - Component generators are not strictly limited to their registered role
    - A signal generator could be used as a feature or vice versa
    - Features can be localized in time or span the entire series
    - It's up to the user to ensure features actually create meaningful class differences

    Key capabilities:
    - Univariate and multivariate time series generation
    - Control over feature positions and randomness
    - Support for shared patterns across dimensions
    - Training/test splits with consistent class distributions
    - Built-in visualization and conversion utilities

    Example usage (univariate):
        ```python
        from xaitimesynth import (
            TimeSeriesBuilder, random_walk, gaussian, shapelet
        )

        # Create a simple binary classification dataset
        dataset = (
            TimeSeriesBuilder(n_timesteps=100, n_samples=200)
            # Class 0: Just random walk with noise
            .for_class(0)
            .add_signal(random_walk(step_size=0.2))
            .add_signal(gaussian(sigma=0.1), role="noise")
            # Class 1: Random walk with noise plus a shapelet feature
            .for_class(1)
            .add_signal(random_walk(step_size=0.2))
            .add_signal(gaussian(sigma=0.1), role="noise")
            .add_feature(shapelet(scale=1.0), start_pct=0.4, end_pct=0.6)
            .build(train_test_split=0.7)
        )
        ```

    Advanced usage:
    - Components can be configured with various parameters
    - Features can be positioned at fixed or random locations
    - For multivariate series, components can target specific dimensions
    - Shared randomness and locations can be controlled across dimensions

    When components are not registered, the builder uses default fill values:
    - Features: NaN where the feature doesn't exist
    - Foundation: zeros where no foundation component exists
    - Noise: zeros where no noise component exists

    Attributes:
        n_timesteps (int): Length of each time series.
        n_samples (int): Total number of samples to generate.
        n_dimensions (int): Number of dimensions in each time series.
        normalization (str): Normalization method for the final time series.
        normalization_kwargs (dict): Additional parameters for normalization.
        random_state (int): Random seed for reproducibility.
        rng (np.random.RandomState): Random number generator.
        feature_fill_value: Value used for non-existent features (default: np.nan).
        foundation_fill_value: Value used for foundation when none exists (default: 0.0).
        noise_fill_value: Value used for noise when none exists (default: 0.0).
        class_definitions (list): List of class definitions with components.
        current_class (dict): Current class being configured.
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
            n_timesteps (int): Length of each time series. Default is 100.
            n_samples (int): Total number of samples to generate. Default is 1000.
            n_dimensions (int): Number of dimensions for multivariate time series. Default is 1 (univariate).
            normalization (str): Normalization method for the final time series.
                Options: "zscore" (standardization), "minmax" (scale to 0-1), or "none". Default is "zscore".
            random_state (int, optional): Seed for random number generation to ensure reproducibility.
            normalization_kwargs (dict, optional): Additional parameters for normalization methods.
                For "minmax": can specify "feature_range" as tuple (min, max).
            feature_fill_value: Value used for non-existent features. Default is np.nan.
                Using NaN makes features only appear where they're defined in visualizations.
            foundation_fill_value: Value used for foundation when none exists. Default is 0.0.
                Foundation typically affects the entire time series, so zeros represent
                "no contribution" rather than "doesn't exist".
            noise_fill_value: Value used for noise when none exists. Default is 0.0.
                Similar to foundation, zeros indicate "no contribution".

        Raises:
            ValueError: If n_dimensions is less than 1.
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

        Creates a new class definition and makes it the target for subsequent component additions.
        Multiple calls create multiple classes for classification tasks.

        Args:
            class_label (int): Integer label for the class, used as the target value.
            weight (float): Relative weight of this class in the dataset. Controls the
                class distribution in the generated dataset. Default is 1.0.

        Returns:
            TimeSeriesBuilder: Self for method chaining.
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

        Ensures all provided dimension indices are within valid range for the configured
        number of dimensions in the builder.

        Args:
            dimensions (List[int]): List of dimension indices to validate.

        Raises:
            ValueError: If any dimension index is out of range (0 to n_dimensions-1).
        """
        for d in dimensions:
            if not 0 <= d < self.n_dimensions:
                raise ValueError(
                    f"Dimension {d} is out of range. "
                    f"Valid dimensions are 0 to {self.n_dimensions - 1}."
                )

    # TODO: add random time shift parameter over multiple channels
    def add_signal(
        self,
        component: Dict[str, Any],
        role: str = "foundation",
        dim: Optional[List[int]] = [0],
        shared_randomness: bool = False,
    ) -> "TimeSeriesBuilder":
        """Add a signal component to the current class.

        Signal components can be either foundation or noise. Foundation components form the
        base structure of the time series, while noise components add random variations.

        Args:
            component (Dict[str, Any]): Component definition dictionary with 'type' and parameters.
            role (str): Role of the component, either 'foundation' or 'noise'. Default is 'foundation'.
            dim (List[int]): List of dimension indices where this signal should be applied.
                Default is [0] (creates univariate time series if all components have dim=[0]).
            shared_randomness (bool): If True, the same random pattern will be used across all
                specified dimensions. If False, each dimension gets its own random pattern
                (for stochastic components). Default is False.

        Returns:
            TimeSeriesBuilder: Self for method chaining.

        Raises:
            ValueError: If no class is selected or if the role is invalid.
        """
        if self.current_class is None:
            raise ValueError("No class selected. Call for_class() first.")

        if role not in ("foundation", "noise"):
            raise ValueError(f"Invalid role: {role}. Must be 'foundation' or 'noise'.")

        self._validate_dimensions(dim)

        # If shared_randomness is True or only one dimension, store a single component
        if shared_randomness or len(dim) == 1:
            component_with_dim = component.copy()
            component_with_dim["dimensions"] = dim
            component_with_dim["shared_randomness"] = shared_randomness
            self.current_class["components"][role].append(component_with_dim)

        # For multiple dimensions with different randomness,
        # create separate component entries for each dimension
        else:
            for d in dim:
                component_with_dim = component.copy()
                component_with_dim["dimensions"] = [d]  # Single dimension
                component_with_dim["shared_randomness"] = shared_randomness
                self.current_class["components"][role].append(component_with_dim)

        return self

    # TODO: add random time shift parameter over multiple channels
    def add_signal_segment(
        self,
        component: Dict[str, Any],
        role: str = "foundation",
        dim: Optional[List[int]] = [0],
        shared_randomness: bool = False,
        start_pct: Optional[float] = None,
        end_pct: Optional[float] = None,
        length_pct: Optional[float] = None,
        random_location: bool = False,
        shared_location: bool = True,
    ) -> "TimeSeriesBuilder":
        """Add a signal component to the current class for a segment of the time series.

        Use this instead of add_signal() when you want to specify a time range for the signal.

        Signal components can be either foundation or noise. Foundation components form the
        base structure of the time series, while noise components add random variations.

        Args:
            component (Dict[str, Any]): Component definition dictionary with 'type' and parameters.
            role (str): Role of the component, either 'foundation' or 'noise'. Default is 'foundation'.
            dim (List[int]): List of dimension indices where this signal should be applied.
                Default is [0] (creates univariate time series if all components have dim=[0]).
            shared_randomness (bool): If True, the same random pattern will be used across all
                specified dimensions. If False, each dimension gets its own random pattern
                (for stochastic components). Default is False.
            start_pct (float, optional): Start position as percentage of time series length (0-1).
            end_pct (float, optional): End position as percentage of time series length (0-1).
            length_pct (float, optional): Length of signal as percentage of time series length (0-1).
            random_location (bool): Whether to place the signal at a random location.
                Default is False (applied to entire time series).
            shared_location (bool): If True and random_location is True, the same random
                location will be used across all dimensions. If False, each dimension gets
                its own random location. Default is True.

        Returns:
            TimeSeriesBuilder: Self for method chaining.

        Raises:
            ValueError: If no class is selected or if the role is invalid.
        """
        if self.current_class is None:
            raise ValueError("No class selected. Call for_class() first.")

        if role not in ("foundation", "noise"):
            raise ValueError(f"Invalid role: {role}. Must be 'foundation' or 'noise'.")

        self._validate_dimensions(dim)

        # If we have time range parameters, validate them
        has_time_range = (
            start_pct is not None
            or end_pct is not None
            or length_pct is not None
            or random_location
        )

        if has_time_range:
            # Add time range parameters to component
            component_with_time = component.copy()

            if random_location:
                if length_pct is None:
                    raise ValueError(
                        "length_pct must be provided when random_location is True"
                    )
                if not (0 < length_pct <= 1):
                    raise ValueError("length_pct must be between 0 and 1")

                component_with_time["random_location"] = True
                component_with_time["length_pct"] = length_pct
                component_with_time["shared_location"] = shared_location
            else:
                if start_pct is None or end_pct is None:
                    raise ValueError(
                        "start_pct and end_pct must be provided when random_location is False"
                    )
                if not (
                    0 <= start_pct < 1 and 0 < end_pct <= 1 and start_pct < end_pct
                ):
                    raise ValueError(
                        "Invalid start_pct or end_pct. Must be between 0 and 1, with start_pct < end_pct"
                    )

                component_with_time["random_location"] = False
                component_with_time["start_pct"] = start_pct
                component_with_time["end_pct"] = end_pct
        else:
            component_with_time = component.copy()

        # Add dimensions and randomness settings to a single component
        # when using shared location/randomness
        if shared_location and random_location or shared_randomness or len(dim) == 1:
            component_with_time["dimensions"] = dim
            component_with_time["shared_randomness"] = shared_randomness
            component_with_time["shared_location"] = shared_location
            self.current_class["components"][role].append(component_with_time)
        else:
            # For multiple dimensions without shared location/randomness,
            # create separate component entries for each dimension
            for d in dim:
                component_with_dim = component_with_time.copy()
                component_with_dim["dimensions"] = [d]  # Single dimension
                component_with_dim["shared_randomness"] = shared_randomness
                component_with_dim["shared_location"] = shared_location
                self.current_class["components"][role].append(component_with_dim)

        return self

    # TODO: add random time shift parameter over multiple channels
    def add_feature(
        self,
        component: Dict[str, Any],
        start_pct: Optional[float] = None,
        end_pct: Optional[float] = None,
        length_pct: Optional[float] = None,
        random_location: bool = False,
        dim: Optional[List[int]] = [0],
        shared_location: bool = True,
        shared_randomness: bool = False,
    ) -> "TimeSeriesBuilder":
        """Add a feature component to the current class.

        Features are distinctive patterns that can differentiate between classes.
        They can be placed at fixed or random locations within the time series.

        Args:
            component (Dict[str, Any]): Component definition dictionary with 'type' and parameters.
            start_pct (float, optional): Start position as percentage of time series length (0-1).
                Required when random_location is False.
            end_pct (float, optional): End position as percentage of time series length (0-1).
                Required when random_location is False.
            length_pct (float, optional): Length of feature as percentage of time series length (0-1).
                Required when random_location is True.
            random_location (bool): Whether to place the feature at a random location.
                Default is False (fixed position).
            dim (List[int]): List of dimension indices where this feature should be applied.
                Default is [0] (creates univariate time series if all components have dim=[0]).
            shared_location (bool): If True and random_location is True, the same random
                location will be used across all dimensions. If False, each dimension gets
                its own random location. Default is True.
            shared_randomness (bool): If True, the same random pattern will be used across
                all dimensions. If False, each dimension gets its own random pattern
                (for stochastic components). Default is False.

        Returns:
            TimeSeriesBuilder: Self for method chaining.

        Raises:
            ValueError: If no class is selected or if location parameters are invalid.
        """
        if self.current_class is None:
            raise ValueError("No class selected. Call for_class() first.")

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

        # Add to feature collection, ensuring the shared location logic is properly observed
        if shared_location and random_location or shared_randomness or len(dim) == 1:
            feature_def["dimensions"] = dim
            feature_def["shared_location"] = shared_location
            feature_def["shared_randomness"] = shared_randomness
            self.current_class["components"]["features"].append(feature_def)
        else:
            # Create separate feature entries for each dimension when not sharing
            for d in dim:
                feature_single_dim = feature_def.copy()
                feature_single_dim["dimensions"] = [d]  # Single dimension
                feature_single_dim["shared_location"] = shared_location
                feature_single_dim["shared_randomness"] = shared_randomness
                self.current_class["components"]["features"].append(feature_single_dim)

        return self

    def _generate_component_vector(
        self, component_def: Dict[str, Any], feature_length: Optional[int] = None
    ) -> np.ndarray:
        """Generate a component vector based on its definition.

        Calls the appropriate component generator based on the component type
        and parameters specified in the definition.

        Args:
            component_def (Dict[str, Any]): Component definition dictionary with 'type'
                and parameters for the generator.
            feature_length (Optional[int]): Length of the feature in timesteps.
                Only used for feature components.

        Returns:
            np.ndarray: Generated component vector with specified pattern.
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
        self,
        feature_def: Dict[str, Any],
        dim_index: Optional[int] = None,
        shared_location_cache: Optional[Tuple[int, int]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a feature vector and its corresponding mask.

        Creates a feature at the specified location (fixed or random) and returns
        both the vector and a boolean mask indicating the feature's position.

        Args:
            feature_def (Dict[str, Any]): Feature definition dictionary with 'type',
                location parameters, and generator parameters.
            dim_index (Optional[int]): The index in the dimensions list to use for location
                determination. Only used when shared_location is False.
            shared_location_cache (Optional[Tuple[int, int]]): Pre-calculated start and end
                indices for a shared location. Used to ensure consistency across dimensions.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple containing:
                - Feature vector with specified pattern at the determined location
                - Boolean mask indicating the feature's position (True where feature exists)
        """
        # Initialize with feature with fill value
        feature = np.full(self.n_timesteps, self.feature_fill_value)
        mask = np.zeros(self.n_timesteps, dtype=bool)

        # Determine feature location
        if feature_def["random_location"]:
            if shared_location_cache is not None:
                # Use the cached shared location
                start_idx, end_idx = shared_location_cache
            else:
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
        feature_params.pop("shared_randomness", None)

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
        """Build the dataset based on the configured class definitions.

        Generates time series data by combining all components for each class according
        to the specified parameters, with options to include component vectors and
        create a train/test split.

        Args:
            return_components (bool): Whether to return the individual component vectors.
                Useful for visualization and analysis. Default is True.
            train_test_split (Optional[float]): If provided, fraction of data to use for training
                (between 0 and 1). The dataset will be randomly split into train and test sets.
                If None, no split is performed. Default is None.

        Returns:
            Dict[str, Any]: Dictionary containing the generated dataset with keys:
                - 'X': Time series data with shape (n_samples, n_timesteps, n_dimensions)
                - 'y': Class labels for each sample
                - 'feature_masks': Boolean masks showing feature locations
                - 'metadata': Dataset configuration information
                - 'components': Individual component vectors (if return_components=True)
                If train_test_split is provided, also includes:
                - 'X_train', 'y_train': Training data
                - 'X_test', 'y_test': Testing data

        Raises:
            ValueError: If no class definitions have been provided.
        """
        if not self.class_definitions:
            raise ValueError(
                "No class definitions provided. Call for_class() at least once."
            )

        # Normalize class weights and determine class distribution
        weights = np.array([cd["weight"] for cd in self.class_definitions])
        weights = weights / weights.sum()
        class_counts = self.rng.multinomial(self.n_samples, weights)

        # Initialize arrays
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
                    # For signals with time range parameters, generate random location once if shared
                    if "random_location" in base_def and base_def["random_location"]:
                        # Determine signal length
                        length_pct = base_def["length_pct"]
                        signal_length = max(1, int(length_pct * self.n_timesteps))
                        max_start = self.n_timesteps - signal_length

                        # If shared_location is True, generate the location once for all dimensions
                        shared_location = base_def.get("shared_location", True)
                        if shared_location:
                            shared_start_idx = self.rng.randint(0, max_start + 1)
                            shared_end_idx = shared_start_idx + signal_length

                        # Apply to specified dimensions with appropriate location handling
                        for i, dim_idx in enumerate(base_def["dimensions"]):
                            # Create a full-length vector filled with the foundation fill value
                            base_vector = np.full(
                                self.n_timesteps, self.foundation_fill_value
                            )

                            # Determine signal location - possibly unique per dimension
                            if shared_location:
                                # Use the shared location for all dimensions
                                start_idx = shared_start_idx
                                end_idx = shared_end_idx
                            else:
                                # Create a unique location for each dimension
                                dim_rng = np.random.RandomState(
                                    self.rng.randint(0, 2**32 - 1)
                                )
                                start_idx = dim_rng.randint(0, max_start + 1)
                                end_idx = start_idx + signal_length

                            # Calculate the actual length of the signal segment
                            signal_length = end_idx - start_idx

                            # Prepare parameters for component generation
                            signal_params = base_def.copy()
                            signal_type = signal_params.pop("type")

                            # Remove location and dimension parameters
                            signal_params.pop("random_location", None)
                            signal_params.pop("length_pct", None)
                            signal_params.pop("shared_location", None)
                            signal_params.pop("dimensions", None)
                            signal_params.pop("shared_randomness", None)

                            # Generate the component only for the specified length
                            signal_values = generate_component(
                                signal_type, signal_length, self.rng, **signal_params
                            )

                            # Place the signal in the correct location
                            base_vector[start_idx:end_idx] = signal_values

                            # Add to foundation for this dimension
                            foundation[:, dim_idx] = self._add_vector_handling_nans(
                                foundation[:, dim_idx], base_vector
                            )
                    else:
                        # Handle non-random location signals (the original behavior)
                        if "random_location" in base_def:
                            # Fixed location signal
                            base_vector = np.full(
                                self.n_timesteps, self.foundation_fill_value
                            )

                            start_pct = base_def["start_pct"]
                            end_pct = base_def["end_pct"]
                            start_idx = int(start_pct * self.n_timesteps)
                            end_idx = int(end_pct * self.n_timesteps)

                            # Ensure at least one timestep is selected
                            if start_idx == end_idx:
                                end_idx = start_idx + 1

                            signal_length = end_idx - start_idx

                            # Generate the component only for the specified length
                            signal_params = base_def.copy()
                            signal_type = signal_params.pop("type")

                            # Remove location parameters
                            signal_params.pop("random_location", None)
                            signal_params.pop("start_pct", None)
                            signal_params.pop("end_pct", None)
                            signal_params.pop("dimensions", None)
                            signal_params.pop("shared_randomness", None)

                            signal_values = generate_component(
                                signal_type, signal_length, self.rng, **signal_params
                            )

                            base_vector[start_idx:end_idx] = signal_values
                        else:
                            # Full-length signal (original behavior)
                            base_vector = self._generate_component_vector(base_def)

                        # Apply to all specified dimensions with the same signal
                        for dim_idx in base_def["dimensions"]:
                            foundation[:, dim_idx] = self._add_vector_handling_nans(
                                foundation[:, dim_idx], base_vector
                            )

                # Add noise components - use the same approach as foundation components
                for noise_def in class_def["components"]["noise"]:
                    # For noise with random location parameters, generate random location once if shared
                    if "random_location" in noise_def and noise_def["random_location"]:
                        # Determine noise length
                        length_pct = noise_def["length_pct"]
                        noise_length = max(1, int(length_pct * self.n_timesteps))
                        max_start = self.n_timesteps - noise_length

                        # If shared_location is True, generate the location once for all dimensions
                        shared_location = noise_def.get("shared_location", True)
                        if shared_location:
                            shared_start_idx = self.rng.randint(0, max_start + 1)
                            shared_end_idx = shared_start_idx + noise_length

                        # Apply to specified dimensions with appropriate location handling
                        for i, dim_idx in enumerate(noise_def["dimensions"]):
                            # Create a full-length vector filled with the noise fill value
                            noise_vector = np.full(
                                self.n_timesteps, self.noise_fill_value
                            )

                            # Determine noise location - possibly unique per dimension
                            if shared_location:
                                # Use the shared location for all dimensions
                                start_idx = shared_start_idx
                                end_idx = shared_end_idx
                            else:
                                # Create a unique location for each dimension
                                dim_rng = np.random.RandomState(
                                    self.rng.randint(0, 2**32 - 1)
                                )
                                start_idx = dim_rng.randint(0, max_start + 1)
                                end_idx = start_idx + noise_length

                            # Calculate the actual length of the noise segment
                            noise_length = end_idx - start_idx

                            # Prepare parameters for component generation
                            noise_params = noise_def.copy()
                            noise_type = noise_params.pop("type")

                            # Remove location and dimension parameters
                            noise_params.pop("random_location", None)
                            noise_params.pop("length_pct", None)
                            noise_params.pop("shared_location", None)
                            noise_params.pop("dimensions", None)
                            noise_params.pop("shared_randomness", None)

                            # Generate the component only for the specified length
                            noise_values = generate_component(
                                noise_type, noise_length, self.rng, **noise_params
                            )

                            # Place the noise in the correct location
                            noise_vector[start_idx:end_idx] = noise_values

                            # Add to noise for this dimension
                            noise[:, dim_idx] = self._add_vector_handling_nans(
                                noise[:, dim_idx], noise_vector
                            )
                    else:
                        # Handle non-random location noise (the original behavior)
                        if "random_location" in noise_def:
                            # Fixed location noise
                            noise_vector = np.full(
                                self.n_timesteps, self.noise_fill_value
                            )

                            start_pct = noise_def["start_pct"]
                            end_pct = noise_def["end_pct"]
                            start_idx = int(start_pct * self.n_timesteps)
                            end_idx = int(end_pct * self.n_timesteps)

                            # Ensure at least one timestep is selected
                            if start_idx == end_idx:
                                end_idx = start_idx + 1

                            noise_length = end_idx - start_idx

                            # Generate the component only for the specified length
                            noise_params = noise_def.copy()
                            noise_type = noise_params.pop("type")

                            # Remove location parameters
                            noise_params.pop("random_location", None)
                            noise_params.pop("start_pct", None)
                            noise_params.pop("end_pct", None)
                            noise_params.pop("dimensions", None)
                            noise_params.pop("shared_randomness", None)

                            noise_values = generate_component(
                                noise_type, noise_length, self.rng, **noise_params
                            )

                            noise_vector[start_idx:end_idx] = noise_values
                        else:
                            # Full-length noise (original behavior)
                            noise_vector = self._generate_component_vector(noise_def)

                        # Apply to all specified dimensions with the same noise
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
                    # For each dimension in the feature
                    feature_dims = feature_def["dimensions"]

                    # Generate a shared random location once if needed
                    shared_location_cache = None
                    if feature_def.get("random_location", False) and feature_def.get(
                        "shared_location", True
                    ):
                        # Pre-calculate the shared location to ensure it's the same across dimensions
                        length_pct = feature_def["length_pct"]
                        feature_length = max(1, int(length_pct * self.n_timesteps))
                        max_start = self.n_timesteps - feature_length
                        shared_start_idx = self.rng.randint(0, max_start + 1)
                        shared_end_idx = shared_start_idx + feature_length
                        shared_location_cache = (shared_start_idx, shared_end_idx)

                    for i, dim_idx in enumerate(feature_dims):
                        # Generate feature vector - if shared_location is True and we have a cached location,
                        # pass it; otherwise pass the dimension index for unique locations
                        dim_index = (
                            None
                            if feature_def.get("shared_location", True)
                            else dim_idx
                        )
                        feature, mask = self._generate_feature_vector(
                            feature_def, dim_index, shared_location_cache
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
                # #TODO: check whether normalisation works as intended
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

        Creates a DataFrame with one row per timestep per component per sample per dimension,
        suitable for detailed analysis and visualization with libraries like Seaborn or Plotly.

        Args:
            dataset (Dict[str, Any]): Dataset dictionary returned by build().
            samples (Optional[List[int]]): List of sample indices to include.
                If None, includes all samples.
            classes (Optional[List[int]]): List of class labels to include.
                If None, includes all classes.
            components (Optional[List[str]]): List of component types to include.
                Default includes all: ["aggregated", "foundation", "noise", "features"]
            dimensions (Optional[List[int]]): List of dimension indices to include.
                If None, includes all dimensions.
            format_classes (bool): If True, format class labels as "Class X".
                Otherwise use numeric labels. Default is False.

        Returns:
            pd.DataFrame: Long-format DataFrame with columns:
                - time: Timestep index
                - value: Component value at that timestep
                - class: Class label (formatted if format_classes=True)
                - sample: Sample index
                - component: Component type
                - feature: Feature name (for feature components)
                - dim: Dimension index

        Raises:
            ValueError: If specified dimensions are out of range.
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

        Special handling of NaN values during vector addition:
        1. Where both vectors have values (not NaN): Normal addition
        2. Where one vector has NaN: Use the non-NaN value
        3. Where both have NaN: Result remains NaN

        This allows components to only contribute where they're defined.

        Args:
            base (np.ndarray): Base vector to add to.
            to_add (np.ndarray): Vector to add to the base.

        Returns:
            np.ndarray: Combined vector with NaNs handled according to the rules above.
        """
        # Stack arrays and use nansum for element-wise addition that ignores NaNs
        result = np.nansum(np.stack([base, to_add]), axis=0)

        # Fix case where both values are NaN (nansum would return 0, but we want NaN)
        both_nan = np.isnan(base) & np.isnan(to_add)
        result[both_nan] = np.nan

        return result
