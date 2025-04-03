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
