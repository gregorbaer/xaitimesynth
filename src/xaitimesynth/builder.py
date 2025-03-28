from typing import Any, Dict, Optional, Tuple

import numpy as np

from .data_structures import TimeSeriesComponents
from .generators import generate_component


class TimeSeriesBuilder:
    """Builder for synthetic time series datasets.

    This class provides a fluent API for building synthetic time series
    datasets with known ground truth features for XAI evaluation.

    Attributes:
        n_timesteps: Length of each time series.
        n_samples: Total number of samples to generate.
        normalize: Whether to z-normalize the final time series.
        random_state: Random seed for reproducibility.
        rng: Random number generator.
        class_definitions: List of class definitions.
        current_class: Current class being configured.
    """

    def __init__(
        self,
        n_timesteps: int = 100,
        n_samples: int = 1000,
        normalize: bool = True,
        random_state: Optional[int] = None,
    ):
        """Initialize the time series builder.

        Args:
            n_timesteps: Length of each time series.
            n_samples: Total number of samples to generate.
            normalize: Whether to z-normalize the final time series.
            random_state: Random seed for reproducibility.
        """
        self.n_timesteps = n_timesteps
        self.n_samples = n_samples
        self.normalize = normalize
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)

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
        # Initialize
        feature = np.zeros(self.n_timesteps)
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
                # Initialize arrays for this sample
                foundation = np.zeros(self.n_timesteps)
                noise = np.zeros(self.n_timesteps)
                features_dict = {}
                feature_masks_dict = {}

                # Add base structure components
                for base_def in class_def["components"]["foundation"]:
                    base_vector = self._generate_component_vector(base_def)
                    foundation += base_vector

                # Add noise components
                for noise_def in class_def["components"]["noise"]:
                    noise_vector = self._generate_component_vector(noise_def)
                    noise += noise_vector

                # Initialize aggregated time series
                aggregated = foundation.copy()

                # Add features
                for feature_idx, feature_def in enumerate(
                    class_def["components"]["features"]
                ):
                    feature, mask = self._generate_feature_vector(feature_def)

                    # Add to aggregated series
                    aggregated += feature

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
                aggregated += noise

                # Normalize if required
                if self.normalize:
                    std = np.std(aggregated)
                    if std > 0:
                        aggregated = (aggregated - np.mean(aggregated)) / std

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
                "normalize": self.normalize,
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
