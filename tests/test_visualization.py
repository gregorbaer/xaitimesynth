"""Functional tests for the visualization module.

These tests verify that the visualization functions run correctly.
Since we cannot verify visual output, we focus on essential functionality
and error handling rather than exhaustive parameter coverage.
"""

from pathlib import Path

import numpy as np
import pytest
from lets_plot.export import ggsave

from xaitimesynth import (
    TimeSeriesBuilder,
    constant,
    random_walk,
)
from xaitimesynth.visualization import (
    plot_component,
    plot_components,
    prepare_plot_data,
    plot_sample,
)


@pytest.fixture
def univariate_dataset():
    """Create a simple univariate dataset with two classes."""
    return (
        TimeSeriesBuilder(n_samples=20, n_timesteps=50, random_state=42)
        .for_class(0)
        .add_signal(random_walk())
        .for_class(1)
        .add_signal(random_walk())
        .add_feature(constant(value=1.0), start_pct=0.4, end_pct=0.6)
        .build()
    )


@pytest.fixture
def multivariate_dataset():
    """Create a multivariate dataset with two classes and two dimensions."""
    return (
        TimeSeriesBuilder(n_samples=20, n_timesteps=50, n_dimensions=2, random_state=42)
        .for_class(0)
        .add_signal(random_walk(), dim=[0, 1])
        .for_class(1)
        .add_signal(random_walk(), dim=[0, 1])
        .add_feature(constant(value=1.0), start_pct=0.3, end_pct=0.5, dim=[0])
        .build()
    )


# ============================================================================
# plot_component tests
# ============================================================================


def test_plot_component_with_signal():
    """Test plot_component with a pre-generated signal array."""
    signal = np.sin(np.linspace(0, 4 * np.pi, 100))
    p = plot_component(signal=signal)
    assert p is not None


def test_plot_component_with_component_type():
    """Test plot_component by specifying a component type."""
    p = plot_component(component_type="random_walk", n_timesteps=100)
    assert p is not None


def test_plot_component_raises_without_signal_or_type():
    """Test that plot_component raises error when neither signal nor type provided."""
    with pytest.raises(ValueError, match="Either signal or component_type"):
        plot_component()


def test_plot_component_raises_for_unknown_type():
    """Test that plot_component raises error for unknown component type."""
    with pytest.raises(ValueError, match="Unknown component type"):
        plot_component(component_type="nonexistent_component")


# ============================================================================
# plot_components tests
# ============================================================================


def test_plot_components_univariate(univariate_dataset):
    """Test plot_components with univariate dataset and various options."""
    # Default
    p = plot_components(univariate_dataset)
    assert p is not None

    # With specific sample indices (dict format)
    class_0_idx = np.where(univariate_dataset["y"] == 0)[0][0]
    class_1_idx = np.where(univariate_dataset["y"] == 1)[0][0]
    p = plot_components(
        univariate_dataset, sample_indices={0: class_0_idx, 1: class_1_idx}
    )
    assert p is not None

    # With component filtering and no indicators
    p = plot_components(
        univariate_dataset, components=["aggregated"], show_indicators=False
    )
    assert p is not None


def test_plot_components_facet_component_order(univariate_dataset, tmp_path: Path):
    """Test component facets render in Background -> Features -> Aggregated order."""
    p = plot_components(univariate_dataset)
    output_path = tmp_path / "plot_components_facets.svg"
    ggsave(p, str(output_path))

    svg_text = output_path.read_text(encoding="utf-8")
    assert "Internal error: NoSuchElementException" not in svg_text

    background_pos = svg_text.find("Background")
    features_pos = svg_text.find("Features")
    aggregated_pos = svg_text.find("Aggregated")

    assert background_pos != -1
    assert features_pos != -1
    assert aggregated_pos != -1
    assert background_pos < features_pos < aggregated_pos


def test_plot_components_multivariate(multivariate_dataset):
    """Test plot_components with multivariate dataset returns list of plots."""
    result = plot_components(multivariate_dataset)
    assert isinstance(result, list)
    assert len(result) == 2  # Two dimensions

    # Single dimension returns single plot
    result = plot_components(multivariate_dataset, dimensions=[0])
    assert not isinstance(result, list)


def test_plot_components_raises_out_of_range(univariate_dataset):
    """Test plot_components raises IndexError for out of range sample index."""
    with pytest.raises(IndexError, match="out of range"):
        plot_components(univariate_dataset, sample_indices=999)


def test_plot_components_raises_wrong_class(univariate_dataset):
    """Test plot_components raises ValueError for mismatched class in dict."""
    class_0_idx = np.where(univariate_dataset["y"] == 0)[0][0]
    with pytest.raises(ValueError, match="has class"):
        plot_components(univariate_dataset, sample_indices={1: class_0_idx})


def _get_line_layer_mapping(plot_obj):
    """Extract the mapping dict for the first geom_line layer in a plot spec."""
    layers = plot_obj.as_dict().get("layers", [])
    line_layers = [layer for layer in layers if layer.get("geom") == "line"]
    assert line_layers, "Expected at least one geom_line layer in plot spec"
    return line_layers[0].get("mapping", {})


def test_plot_components_uses_line_group_for_multiple_features_univariate():
    """Test univariate plots group lines explicitly when multiple features exist."""
    dataset = (
        TimeSeriesBuilder(n_samples=20, n_timesteps=50, random_state=42)
        .for_class(0)
        .add_signal(random_walk())
        .for_class(1)
        .add_signal(random_walk())
        .add_feature(constant(value=1.0), start_pct=0.2, end_pct=0.4)
        .add_feature(constant(value=-0.8), start_pct=0.6, end_pct=0.8)
        .build()
    )

    p = plot_components(dataset)
    mapping = _get_line_layer_mapping(p)
    assert mapping.get("group") == "line_group"


def test_plot_components_uses_line_group_for_multiple_features_multivariate():
    """Test multivariate plots group lines explicitly when multiple features exist."""
    dataset = (
        TimeSeriesBuilder(n_samples=20, n_timesteps=50, n_dimensions=2, random_state=42)
        .for_class(0)
        .add_signal(random_walk(), dim=[0, 1])
        .for_class(1)
        .add_signal(random_walk(), dim=[0, 1])
        .add_feature(constant(value=1.0), start_pct=0.2, end_pct=0.4, dim=[0])
        .add_feature(constant(value=-0.8), start_pct=0.6, end_pct=0.8, dim=[0])
        .build()
    )

    result = plot_components(dataset)
    assert isinstance(result, list)
    assert len(result) == 2
    for p in result:
        mapping = _get_line_layer_mapping(p)
        assert mapping.get("group") == "line_group"


def test_prepare_plot_data_keeps_multiple_features_separate():
    """Test feature rows remain separate (not internally aggregated) in plot data."""
    dataset = (
        TimeSeriesBuilder(n_samples=20, n_timesteps=50, random_state=42)
        .for_class(0)
        .add_signal(random_walk())
        .for_class(1)
        .add_signal(random_walk())
        .add_feature(constant(value=1.0), start_pct=0.2, end_pct=0.4)
        .add_feature(constant(value=-0.8), start_pct=0.6, end_pct=0.8)
        .build()
    )

    class_1_idx = np.where(dataset["y"] == 1)[0][0]
    df = prepare_plot_data(
        dataset,
        sample_indices={1: class_1_idx},
        components_to_include=["features"],
        dimensions=[0],
    )

    feature_df = df[df["component"] == "features"]
    assert feature_df["feature"].nunique() > 1


# ============================================================================
# plot_sample tests
# ============================================================================


def test_plot_sample_basic(univariate_dataset):
    """Test plot_sample with dataset components."""
    p = plot_sample(
        X=univariate_dataset["X"],
        y=univariate_dataset["y"],
        components=univariate_dataset["components"],
        sample_idx=0,
    )
    assert p is not None


def test_plot_sample_without_y(univariate_dataset):
    """Test plot_sample without y labels (uses dummy labels)."""
    p = plot_sample(
        X=univariate_dataset["X"],
        components=univariate_dataset["components"],
        sample_idx=0,
    )
    assert p is not None


def test_plot_components_with_raw_array():
    """Test plotting raw array without components (aggregated only)."""
    X = np.random.randn(10, 1, 50)
    y = np.zeros(10, dtype=int)
    dataset = {"X": X, "y": y}

    p = plot_components(dataset, components=["aggregated"], show_indicators=False)
    assert p is not None
