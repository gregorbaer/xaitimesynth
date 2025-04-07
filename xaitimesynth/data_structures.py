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
        """Validate that components have compatible shapes with the foundation."""
        expected_length = self.foundation.shape[0]  # Time dimension length

        # Check noise component
        if self.noise is not None:
            if self.noise.shape[0] != expected_length:
                raise ValueError(
                    f"The 'noise' component first dimension {self.noise.shape[0]} doesn't match "
                    f"foundation first dimension {expected_length}."
                )
            # For multivariate case, ensure other dimensions match if present
            if len(self.noise.shape) > 1 and len(self.foundation.shape) > 1:
                if self.noise.shape[1] != self.foundation.shape[1]:
                    raise ValueError(
                        f"The 'noise' component shape {self.noise.shape} doesn't match "
                        f"foundation shape {self.foundation.shape} in the second dimension."
                    )

        # Check features components
        if self.features is not None:
            for feature_name, feature_data in self.features.items():
                # For features, we only validate that the time dimension matches
                # This allows dimension-specific features to be 1D arrays
                if feature_data.shape[0] != expected_length:
                    raise ValueError(
                        f"The feature '{feature_name}' first dimension {feature_data.shape[0]} doesn't match "
                        f"foundation first dimension {expected_length}."
                    )

        # Check feature masks components
        if self.feature_masks is not None:
            for mask_name, mask_data in self.feature_masks.items():
                # Feature masks should also match at least in the time dimension
                if mask_data.shape[0] != expected_length:
                    raise ValueError(
                        f"The feature mask '{mask_name}' first dimension {mask_data.shape[0]} doesn't match "
                        f"foundation first dimension {expected_length}."
                    )

        # Check aggregated component
        if self.aggregated is not None:
            if self.aggregated.shape != self.foundation.shape:
                raise ValueError(
                    f"The 'aggregated' component shape {self.aggregated.shape} doesn't match "
                    f"foundation shape {self.foundation.shape}."
                )
