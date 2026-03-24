"""High-level dataset convenience functions.

This module provides ready-made dataset generators for well-known synthetic
time series benchmarks, implemented on top of TimeSeriesBuilder so that
ground-truth feature masks are automatically produced alongside the data.
"""

from typing import Any, Dict, List, Optional

import numpy as np

from .builder import TimeSeriesBuilder
from .components import gaussian_noise, manual


def generate_cylinder_bell_funnel(
    n_samples: int = 300,
    n_timesteps: int = 128,
    weights: Optional[List[float]] = None,
    random_state: Optional[int] = None,
    normalization: str = "none",
    data_format: str = "channels_first",
) -> Dict[str, Any]:
    """Generate a Cylinder-Bell-Funnel (CBF) dataset with ground-truth feature masks.

    Recreates the classic CBF time series benchmark (Saito, 2000) using
    xaitimesynth's builder, so each sample comes with a boolean ``feature_mask``
    that marks the exact window where the class-discriminating pattern lives.

    The three classes differ only inside a randomly placed window [a, b]::

        Cylinder (0):  constant plateau of amplitude (6 + η)
        Bell     (1):  linearly increasing ramp  0  → (6 + η)
        Funnel   (2):  linearly decreasing ramp (6 + η) → 0

    Outside [a, b] all classes share the same Gaussian noise background ε(t) ~ N(0,1).
    The amplitude noise η ~ N(0,1) is drawn fresh for every sample.

    **Approximation vs. original:**
    The original formulation draws ``a ~ Uniform[16, 32]`` (window never starts
    before timestep 16) and ``b - a ~ Uniform[32, 96]``.  This implementation
    samples the window *length* uniformly from [32, 96] timesteps
    (``length_pct=(0.25, 0.75)``) and places it at a *fully random* start
    position, so the window can begin at timestep 0.  The length distribution
    is faithful; the start distribution is wider.  For XAI benchmarking the
    ground-truth mask is what matters, so this difference is intentional.

    Args:
        n_samples (int): Total number of time series to generate. Default 300.
        n_timesteps (int): Length of each time series. Default 128.
        weights (list of float, optional): Sampling weight for each of the three
            classes ``[w_cylinder, w_bell, w_funnel]``. Must be positive and sum
            to 1 (or to any positive value — they are normalised internally).
            If ``None``, classes are balanced (weight 1/3 each). Default None.
        random_state (int, optional): Seed for reproducibility. Default None.
        normalization (str): Normalisation applied to each generated series.
            ``"none"`` preserves the raw CBF signal values (recommended for
            comparison with the original). Other options: ``"zscore"``,
            ``"minmax"``. Default ``"none"``.
        data_format (str): Output tensor layout.  ``"channels_first"`` gives
            ``X`` shape ``(n_samples, 1, n_timesteps)`` (PyTorch convention);
            ``"channels_last"`` gives ``(n_samples, n_timesteps, 1)``.
            Default ``"channels_first"``.

    Returns:
        dict: Standard xaitimesynth dataset dictionary with keys:

            - ``"X"``: numpy array of shape ``(n_samples, 1, n_timesteps)``
              (or channels-last equivalent).
            - ``"y"``: numpy array of shape ``(n_samples,)`` with class labels
              0 (Cylinder), 1 (Bell), 2 (Funnel).
            - ``"feature_masks"``: dict mapping feature name → boolean array of
              shape ``(n_samples, n_timesteps)``.  ``True`` where the
              class-discriminating window is located.
            - ``"metadata"``: generation metadata dict.
            - ``"components"``: per-sample component breakdown.

    Raises:
        ValueError: If ``weights`` has the wrong length or contains non-positive values.

    References:
        Saito, N. (2000). Local feature extraction and its applications using a
        library of bases. *Topics in Analysis and Its Applications: Selected
        Theses*, 269–451. World Scientific.

    Example:
        ```python
        dataset = generate_cylinder_bell_funnel(n_samples=90, random_state=42)
        X, y = dataset["X"], dataset["y"]
        X.shape
        # (90, 1, 128)
        np.bincount(y)
        # array([30, 30, 30])
        masks = dataset["feature_masks"]
        ```
    """
    # --- Validate and normalise weights -------------------------------------
    if weights is None:
        weights = [1 / 3, 1 / 3, 1 / 3]
    else:
        weights = list(weights)
        if len(weights) != 3:
            raise ValueError(
                f"weights must have 3 elements (one per class), got {len(weights)}"
            )
        if any(w <= 0 for w in weights):
            raise ValueError("All weights must be positive")
        total = sum(weights)
        weights = [w / total for w in weights]

    # --- Per-sample feature generators -------------------------------------
    # η ~ N(0,1) is drawn from rng so it varies across samples.

    def _cylinder(n_timesteps, rng, length, **kwargs):
        """Constant level shift of amplitude (6 + η)."""
        eta = rng.randn()
        return np.full(length, 6.0 + eta)

    def _bell(n_timesteps, rng, length, **kwargs):
        """Linearly increasing ramp from 0 to (6 + η)."""
        eta = rng.randn()
        return np.linspace(0, 6.0 + eta, length)

    def _funnel(n_timesteps, rng, length, **kwargs):
        """Linearly decreasing ramp from (6 + η) to 0."""
        eta = rng.randn()
        return np.linspace(6.0 + eta, 0, length)

    # --- Build dataset ------------------------------------------------------
    dataset = (
        TimeSeriesBuilder(
            n_timesteps=n_timesteps,
            n_samples=n_samples,
            normalization=normalization,
            random_state=random_state,
            data_format=data_format,
        )
        .for_class(0, weight=weights[0])  # Cylinder
        .add_signal(gaussian_noise(mu=0, sigma=1))
        .add_feature(
            manual(generator=_cylinder),
            random_location=True,
            length_pct=(0.25, 0.75),  # b-a ~ Uniform[32, 96] out of 128 timesteps
        )
        .for_class(1, weight=weights[1])  # Bell
        .add_signal(gaussian_noise(mu=0, sigma=1))
        .add_feature(
            manual(generator=_bell),
            random_location=True,
            length_pct=(0.25, 0.75),
        )
        .for_class(2, weight=weights[2])  # Funnel
        .add_signal(gaussian_noise(mu=0, sigma=1))
        .add_feature(
            manual(generator=_funnel),
            random_location=True,
            length_pct=(0.25, 0.75),
        )
        .build()
    )

    return dataset
