from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


def plot_sample(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    feature_masks: Optional[Dict[str, np.ndarray]] = None,
    components: Optional[List] = None,
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
        The created figure.
    """
    fig = plt.figure(figsize=figsize)

    n_plots = 1
    if components is not None:
        n_plots += 2 + len(components[sample_idx].features or {})

    # Plot the full time series
    ax1 = plt.subplot(n_plots, 1, 1)
    ax1.plot(X[sample_idx], "b-", label="Time Series")
    ax1.set_title(
        f"Sample {sample_idx}" + (f" (Class {y[sample_idx]})" if y is not None else "")
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

        # Plot noise if available
        if comp.noise is not None:
            ax3 = plt.subplot(n_plots, 1, 3, sharex=ax1)
            ax3.plot(comp.noise, "r-")
            ax3.set_title("Noise")

        # Plot each feature if available
        if comp.features:
            for i, (name, feature) in enumerate(comp.features.items()):
                ax = plt.subplot(n_plots, 1, 4 + i, sharex=ax1)
                ax.plot(feature, "c-")
                ax.set_title(f"Feature: {name}")

                # Highlight the feature region if masks are available
                if comp.feature_masks and name in comp.feature_masks:
                    mask = comp.feature_masks[name]
                    # Find contiguous regions in the mask
                    changes = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
                    starts = np.where(changes == 1)[0]
                    ends = np.where(changes == -1)[0]

                    for start, end in zip(starts, ends):
                        ax.axvspan(start, end, alpha=0.2, color="y")

    plt.tight_layout()
    return fig
