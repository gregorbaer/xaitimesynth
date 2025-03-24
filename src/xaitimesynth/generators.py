from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


@dataclass
class TimeSeriesComponents:
    """Stores the separate components of a generated time series.

    This dataclass holds the individual vectors that were used to create
    a synthetic time series, allowing for ground truth evaluation of
    feature attribution methods.

    Attributes:
        base_structure: Base structure component (e.g., constant, random walk).
        noise: Noise component added to the series.
        features: Dictionary mapping feature names to their vector representations.
        feature_masks: Dictionary of boolean masks indicating feature locations.
        aggregated: The final aggregated time series after combining components.
    """

    base_structure: np.ndarray
    noise: np.ndarray
    features: Dict[str, np.ndarray]
    feature_masks: Dict[str, np.ndarray]
    aggregated: np.ndarray


class BaseStructureType(Enum):
    """Types of base structures for synthetic time series."""

    CONSTANT = "constant"
    RANDOM_WALK = "random_walk"
    AR = "autoregressive"


class NoiseType(Enum):
    """Types of noise for synthetic time series."""

    GAUSSIAN = "gaussian"
    UNIFORM = "uniform"
    NONE = "none"


class FeatureType(Enum):
    """Types of localized features for synthetic time series."""

    SHAPELET = "shapelet"
    LEVEL_CHANGE = "level_change"
    TREND = "trend"
    PEAK = "peak"
    TROUGH = "trough"
    TIME_FREQUENCY = "time_frequency"


class TimeSeriesGenerator:
    """Generator for synthetic time series classification datasets with ground truth features.

    This class creates synthetic time series data with known ground truth
    discriminative features for evaluating XAI methods. It constructs time
    series by combining base structures, noise, and localized features.

    The generator supports binary classification with precisely known
    discriminative features, allowing for direct evaluation of feature
    attribution methods across different feature types and locations.
    """

    def __init__(
        self,
        n_timesteps: int = 100,
        n_samples: int = 1000,
        normalize: bool = True,
        random_state: Optional[int] = None,
    ):
        """Initialize the time series generator.

        Args:
            n_timesteps: Length of each time series.
            n_samples: Number of samples to generate.
            normalize: Whether to z-normalize the final time series.
            random_state: Random seed for reproducibility. If None, use system time.
        """
        self.n_timesteps = n_timesteps
        self.n_samples = n_samples
        self.normalize = normalize
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)

    def generate_base_structure(
        self,
        base_type: Union[str, BaseStructureType] = BaseStructureType.CONSTANT,
        **kwargs,
    ) -> np.ndarray:
        """Generate the base structure component.

        Args:
            base_type: Type of base structure to generate.
            **kwargs: Additional parameters for specific base structures.
                For CONSTANT: value (float, default=0)
                For RANDOM_WALK: step_size (float, default=0.1)
                For AR: coefficients (List[float]), sigma (float, default=1.0)

        Returns:
            np.ndarray: The generated base structure vector.
        """
        if isinstance(base_type, str):
            base_type = BaseStructureType(base_type)

        base = np.zeros(self.n_timesteps)

        if base_type == BaseStructureType.CONSTANT:
            value = kwargs.get("value", 0.0)
            base.fill(value)

        elif base_type == BaseStructureType.RANDOM_WALK:
            step_size = kwargs.get("step_size", 0.1)
            steps = self.rng.normal(0, step_size, self.n_timesteps)
            base = np.cumsum(steps)

        elif base_type == BaseStructureType.AR:
            coefficients = kwargs.get("coefficients", [0.8])
            sigma = kwargs.get("sigma", 1.0)
            p = len(coefficients)

            # Initialize with random values
            base[:p] = self.rng.normal(0, sigma, p)

            # Generate the autoregressive process
            for t in range(p, self.n_timesteps):
                base[t] = np.sum(
                    [coefficients[i] * base[t - i - 1] for i in range(p)]
                ) + self.rng.normal(0, sigma)

        return base

    def generate_noise(
        self, noise_type: Union[str, NoiseType] = NoiseType.GAUSSIAN, **kwargs
    ) -> np.ndarray:
        """Generate the noise component.

        Args:
            noise_type: Type of noise to generate.
            **kwargs: Additional parameters for specific noise types.
                For GAUSSIAN: mu (float, default=0), sigma (float, default=0.1)
                For UNIFORM: low (float, default=-0.1), high (float, default=0.1)

        Returns:
            np.ndarray: The generated noise vector.
        """
        if isinstance(noise_type, str):
            if noise_type.lower() == "none":
                noise_type = NoiseType.NONE
            else:
                noise_type = NoiseType(noise_type)

        if noise_type == NoiseType.NONE:
            return np.zeros(self.n_timesteps)

        if noise_type == NoiseType.GAUSSIAN:
            mu = kwargs.get("mu", 0.0)
            sigma = kwargs.get("sigma", 0.1)
            return self.rng.normal(mu, sigma, self.n_timesteps)

        if noise_type == NoiseType.UNIFORM:
            low = kwargs.get("low", -0.1)
            high = kwargs.get("high", 0.1)
            return self.rng.uniform(low, high, self.n_timesteps)

        raise ValueError(f"Unsupported noise type: {noise_type}")

    def generate_feature(
        self,
        feature_type: Union[str, FeatureType],
        start_pct: Optional[float] = None,
        end_pct: Optional[float] = None,
        length_pct: Optional[float] = None,
        random_location: bool = False,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a localized feature component.

        Args:
            feature_type: Type of feature to generate.
            start_pct: Start position as percentage of time series length (0-1).
                Required if random_location is False.
            end_pct: End position as percentage of time series length (0-1).
                Required if random_location is False.
            length_pct: Length of feature as percentage of time series length (0-1).
                Required if random_location is True.
            random_location: Whether to place the feature at a random location.
                If True, use length_pct instead of start_pct/end_pct.
            **kwargs: Additional parameters for specific feature types.
                For SHAPELET: pattern (np.ndarray), scale (float, default=1.0)
                For LEVEL_CHANGE: amplitude (float, default=1.0)
                For TREND: slope (float, default=0.1)
                For PEAK/TROUGH: amplitude (float, default=1.0), width (int, default=3)
                For TIME_FREQUENCY: frequency (float), amplitude (float, default=1.0)

        Returns:
            Tuple[np.ndarray, np.ndarray]: The generated feature vector and
                a boolean mask indicating feature locations.
        """
        if isinstance(feature_type, str):
            feature_type = FeatureType(feature_type)

        # Determine feature location
        if random_location:
            # Validate length percentage input
            if length_pct is None:
                raise ValueError(
                    "length_pct must be provided when random_location is True"
                )
            if not (0 < length_pct <= 1):
                raise ValueError("length_pct must be between 0 and 1")

            # Calculate feature length in time steps
            feature_length = max(1, int(length_pct * self.n_timesteps))

            # Generate random start position
            max_start = self.n_timesteps - feature_length
            start_idx = self.rng.randint(0, max_start + 1)
            end_idx = start_idx + feature_length
        else:
            # Validate fixed location inputs
            if start_pct is None or end_pct is None:
                raise ValueError(
                    "start_pct and end_pct must be provided when random_location is False"
                )
            if not (0 <= start_pct <= 1 and 0 <= end_pct <= 1 and start_pct <= end_pct):
                raise ValueError(
                    "Invalid start_pct or end_pct. Must be between 0 and 1, with start_pct <= end_pct"
                )

            # Calculate start and end indices
            start_idx = int(start_pct * self.n_timesteps)
            end_idx = int(end_pct * self.n_timesteps)

            # Ensure at least one timestep is selected
            if start_idx == end_idx:
                end_idx = start_idx + 1

        # Initialize feature vector and mask
        feature = np.zeros(self.n_timesteps)
        mask = np.zeros(self.n_timesteps, dtype=bool)
        mask[start_idx:end_idx] = True

        # Generate the specific feature
        if feature_type == FeatureType.SHAPELET:
            # Default shapelet is a bump
            if "pattern" in kwargs:
                pattern = kwargs["pattern"]
            else:
                t = np.linspace(-1, 1, end_idx - start_idx)
                pattern = np.exp(-5 * t**2)

            scale = kwargs.get("scale", 1.0)
            # Ensure pattern length matches feature length
            if len(pattern) != end_idx - start_idx:
                # Resample pattern to match feature length
                pattern = np.interp(
                    np.linspace(0, 1, end_idx - start_idx),
                    np.linspace(0, 1, len(pattern)),
                    pattern,
                )
            feature[start_idx:end_idx] = scale * pattern

        elif feature_type == FeatureType.LEVEL_CHANGE:
            amplitude = kwargs.get("amplitude", 1.0)
            feature[start_idx:end_idx] = amplitude

        elif feature_type == FeatureType.TREND:
            slope = kwargs.get("slope", 0.1)
            t = np.arange(end_idx - start_idx)
            feature[start_idx:end_idx] = slope * t

        elif feature_type in (FeatureType.PEAK, FeatureType.TROUGH):
            amplitude = kwargs.get("amplitude", 1.0)
            if feature_type == FeatureType.TROUGH:
                amplitude = -amplitude

            width = kwargs.get("width", min(3, end_idx - start_idx))

            # Generate a peak centered in the feature region
            center_idx = (start_idx + end_idx) // 2
            half_width = width // 2
            peak_start = max(center_idx - half_width, start_idx)
            peak_end = min(center_idx + half_width + 1, end_idx)

            feature[peak_start:peak_end] = amplitude

        elif feature_type == FeatureType.TIME_FREQUENCY:
            if "frequency" not in kwargs:
                raise ValueError(
                    "frequency parameter is required for TIME_FREQUENCY feature"
                )

            frequency = kwargs["frequency"]
            amplitude = kwargs.get("amplitude", 1.0)

            # Generate a sinusoid at the specified frequency
            t = np.arange(start_idx, end_idx)
            feature[start_idx:end_idx] = amplitude * np.sin(
                2 * np.pi * frequency * t / self.n_timesteps
            )

        else:
            raise ValueError(f"Unsupported feature type: {feature_type}")

        return feature, mask

    def generate_dataset(
        self,
        class_definitions: List[Dict[str, Any]],
        return_components: bool = True,
        train_test_split: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Generate a synthetic classification dataset.

        Args:
            class_definitions: List of dictionaries defining each class's components.
                Each dictionary should contain:
                - 'base_structure': Dict with 'type' and parameters
                - 'noise': Dict with 'type' and parameters
                - 'features': List of dicts with 'type', 'start_pct', 'end_pct', and parameters
                - 'weight': Relative weight of this class in the dataset (optional)
            return_components: Whether to return the component vectors.
            train_test_split: If not None, fraction of data to use for training.

        Returns:
            Dict containing generated dataset:
                - 'X': Time series data (n_samples, n_timesteps)
                - 'y': Class labels
                - 'components': TimeSeriesComponents for each sample (if return_components=True)
                - 'feature_masks': Dictionary of ground truth feature locations
                - 'metadata': Dictionary with generation parameters
                - 'X_train', 'X_test', 'y_train', 'y_test': If train_test_split is provided
        """
        # Validate input
        if not class_definitions:
            raise ValueError("At least one class definition is required")

        # Normalize class weights
        weights = np.array([cd.get("weight", 1.0) for cd in class_definitions])
        weights = weights / weights.sum()

        # Determine class distribution
        class_counts = self.rng.multinomial(self.n_samples, weights)

        # Initialize arrays
        X = np.zeros((self.n_samples, self.n_timesteps))
        y = np.zeros(self.n_samples, dtype=int)
        all_components = []
        feature_masks = {}
        metadata = {
            "n_samples": self.n_samples,
            "n_timesteps": self.n_timesteps,
            "class_definitions": class_definitions,
            "normalize": self.normalize,
            "random_state": self.random_state,
        }

        # Generate data for each class
        sample_idx = 0
        for class_idx, (class_def, count) in enumerate(
            zip(class_definitions, class_counts)
        ):
            for _ in range(count):
                # Generate base structure
                base_params = class_def.get(
                    "base_structure", {"type": "constant"}
                ).copy()
                base_type = base_params.pop("type", "constant")
                base = self.generate_base_structure(base_type, **base_params)

                # Generate noise
                noise_params = class_def.get("noise", {"type": "gaussian"}).copy()
                noise_type = noise_params.pop("type", "gaussian")
                noise = self.generate_noise(noise_type, **noise_params)

                # Initialize aggregated time series and component storage
                aggregated = base.copy()
                features_dict = {}
                feature_masks_dict = {}

                # Add features
                for feature_idx, feature_def in enumerate(
                    class_def.get("features", [])
                ):
                    feature_def = (
                        feature_def.copy()
                    )  # Create a copy to avoid modifying the original
                    feature_type = feature_def.pop("type")

                    # Handle different feature location specifications
                    random_location = feature_def.pop("random_location", False)

                    if random_location:
                        # Random location based on length
                        length_pct = feature_def.pop("length_pct")
                        feature, mask = self.generate_feature(
                            feature_type,
                            random_location=True,
                            length_pct=length_pct,
                            **feature_def,
                        )
                    else:
                        # Fixed location based on start/end percentages
                        start_pct = feature_def.pop("start_pct")
                        end_pct = feature_def.pop("end_pct")
                        feature, mask = self.generate_feature(
                            feature_type,
                            start_pct=start_pct,
                            end_pct=end_pct,
                            **feature_def,
                        )

                    # Add to aggregated series
                    aggregated += feature

                    # Store components
                    feature_name = f"feature_{feature_idx}_{feature_type}"
                    features_dict[feature_name] = feature
                    feature_masks_dict[feature_name] = mask

                    # Add to global feature masks
                    feature_key = f"class_{class_idx}_{feature_name}"
                    if feature_key not in feature_masks:
                        feature_masks[feature_key] = np.zeros(
                            (self.n_samples, self.n_timesteps), dtype=bool
                        )

                    feature_masks[feature_key][sample_idx] = mask

                # Add noise to aggregated series
                aggregated += noise

                # Normalize if required
                if self.normalize:
                    # Avoid division by zero
                    std = np.std(aggregated)
                    if std > 0:
                        aggregated = (aggregated - np.mean(aggregated)) / std

                # Store the result
                X[sample_idx] = aggregated
                y[sample_idx] = class_idx

                # Store components if needed
                if return_components:
                    all_components.append(
                        TimeSeriesComponents(
                            base_structure=base,
                            noise=noise,
                            features=features_dict,
                            feature_masks=feature_masks_dict,
                            aggregated=aggregated,
                        )
                    )

                sample_idx += 1

        # Prepare result dictionary
        result = {"X": X, "y": y, "feature_masks": feature_masks, "metadata": metadata}

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

    def plot_sample(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        feature_masks: Optional[Dict[str, np.ndarray]] = None,
        components: Optional[List[TimeSeriesComponents]] = None,
        sample_idx: int = 0,
        figsize: Tuple[int, int] = (12, 8),
    ) -> plt.Figure:
        """Plot a time series sample with its components and feature masks.

        Args:
            X: Time series data.
            y: Class labels.
            feature_masks: Dictionary of feature masks.
            components: List of TimeSeriesComponents objects.
            sample_idx: Index of the sample to plot.
            figsize: Figure size.

        Returns:
            plt.Figure: The created figure.
        """
        fig = plt.figure(figsize=figsize)

        n_plots = 1
        if components is not None:
            n_plots += 2 + len(components[sample_idx].features)

        # Plot the full time series
        ax1 = plt.subplot(n_plots, 1, 1)
        ax1.plot(X[sample_idx], "b-", label="Time Series")
        ax1.set_title(
            f"Sample {sample_idx}"
            + (f" (Class {y[sample_idx]})" if y is not None else "")
        )
        ax1.legend()

        # Highlight feature regions if masks are provided
        if feature_masks is not None:
            for key, mask in feature_masks.items():
                if not key.startswith("class_"):
                    continue

                # Check if this feature belongs to the sample's class
                class_str = f"class_{y[sample_idx]}_" if y is not None else ""
                if not class_str or key.startswith(class_str):
                    # Find contiguous feature regions
                    sample_mask = mask[sample_idx]
                    idx_ranges = []
                    start_idx = None

                    for i, val in enumerate(sample_mask):
                        if val and start_idx is None:
                            start_idx = i
                        elif not val and start_idx is not None:
                            idx_ranges.append((start_idx, i))
                            start_idx = None

                    # Add last range if mask ends with True
                    if start_idx is not None:
                        idx_ranges.append((start_idx, len(sample_mask)))

                    # Highlight each range
                    feature_name = key.replace(class_str, "")
                    for start, end in idx_ranges:
                        ax1.add_patch(
                            Rectangle(
                                (start, ax1.get_ylim()[0]),
                                end - start,
                                ax1.get_ylim()[1] - ax1.get_ylim()[0],
                                alpha=0.2,
                                color="r",
                                label=feature_name
                                if (start, end) == idx_ranges[0]
                                else None,
                            )
                        )

        # Plot individual components if available
        if components is not None:
            comp = components[sample_idx]

            # Plot base structure
            ax2 = plt.subplot(n_plots, 1, 2, sharex=ax1)
            ax2.plot(comp.base_structure, "g-")
            ax2.set_title("Base Structure")

            # Plot noise
            ax3 = plt.subplot(n_plots, 1, 3, sharex=ax1)
            ax3.plot(comp.noise, "r-")
            ax3.set_title("Noise")

            # Plot each feature
            for i, (name, feature) in enumerate(comp.features.items()):
                ax = plt.subplot(n_plots, 1, 4 + i, sharex=ax1)
                ax.plot(feature, "c-")
                ax.set_title(f"Feature: {name}")

                # Highlight the feature region
                if name in comp.feature_masks:
                    mask = comp.feature_masks[name]
                    # Find contiguous regions in the mask
                    changes = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
                    starts = np.where(changes == 1)[0]
                    ends = np.where(changes == -1)[0]

                    for start, end in zip(starts, ends):
                        ax.axvspan(start, end, alpha=0.2, color="y")

        plt.tight_layout()
        return fig
