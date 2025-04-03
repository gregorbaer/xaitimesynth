from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


@dataclass
class TimeSeriesComponents:
    """Stores the separate components of a generated time series.

    This dataclass is designed to hold the individual components that constitute
    a synthetic time series. By storing these components separately, it facilitates
    ground truth evaluation of XAI (Explainable AI) methods, allowing for a deeper
    understanding of how each component contributes to the final time series.

    Attributes:
        foundation (np.ndarray): Foundational signal, or base structure component (e.g., constant, random walk).
        noise (Optional[np.ndarray]): Noise component added to the series. Defaults to None.
        features (Optional[Dict[str, np.ndarray]]): Dictionary mapping feature names to their vector representations. Defaults to None.
        feature_masks (Optional[Dict[str, np.ndarray]]): Dictionary of boolean masks indicating feature locations. Defaults to None.
        aggregated (Optional[np.ndarray]): The final aggregated time series after combining components. Defaults to None.
    """

    foundation: np.ndarray
    noise: Optional[np.ndarray] = None
    features: Optional[Dict[str, np.ndarray]] = None
    feature_masks: Optional[Dict[str, np.ndarray]] = None
    aggregated: Optional[np.ndarray] = None

    def __post_init__(self):
        """Validate that all components have the same shape as the foundation."""
        expected_shape = self.foundation.shape

        # Check noise component
        if self.noise is not None and self.noise.shape != expected_shape:
            raise ValueError(
                f"The 'noise' component shape {self.noise.shape} doesn't match "
                f"foundation shape {expected_shape}."
            )

        # Check features components
        if self.features is not None:
            for feature_name, feature_data in self.features.items():
                if feature_data.shape != expected_shape:
                    raise ValueError(
                        f"The feature '{feature_name}' shape {feature_data.shape} doesn't match "
                        f"foundation shape {expected_shape}."
                    )

        # Check feature masks components
        if self.feature_masks is not None:
            for mask_name, mask_data in self.feature_masks.items():
                if mask_data.shape != expected_shape:
                    raise ValueError(
                        f"The feature mask '{mask_name}' shape {mask_data.shape} doesn't match "
                        f"foundation shape {expected_shape}."
                    )

        # Check aggregated component
        if self.aggregated is not None and self.aggregated.shape != expected_shape:
            raise ValueError(
                f"The 'aggregated' component shape {self.aggregated.shape} doesn't match "
                f"foundation shape {expected_shape}."
            )
