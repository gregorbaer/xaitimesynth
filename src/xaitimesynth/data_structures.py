from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


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
    noise: Optional[np.ndarray] = None
    features: Optional[Dict[str, np.ndarray]] = None
    feature_masks: Optional[Dict[str, np.ndarray]] = None
    aggregated: Optional[np.ndarray] = None
