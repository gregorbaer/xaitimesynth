"""Tests for the xaitimesynth.parser module.

This module contains tests for the configuration parsing functionality, particularly
the load_builders_from_config function which creates TimeSeriesBuilder instances from
various configuration sources (dictionary, file, or YAML string).
"""

from pathlib import Path
from typing import Dict

import numpy as np
import pytest
import yaml

from xaitimesynth.builder import TimeSeriesBuilder
from xaitimesynth.parser import (
    _create_single_builder_from_dict,
    load_builders_from_config,
)


@pytest.fixture
def minimal_config_dict() -> Dict:
    """Fixture providing a minimal dataset configuration dictionary."""
    return {
        "test_dataset": {
            "n_timesteps": 50,
            "n_samples": 20,
            "classes": [
                {
                    "id": 0,
                    "signals": [
                        {"function": "random_walk", "params": {"step_size": 0.1}}
                    ],
                }
            ],
        }
    }


@pytest.fixture
def complex_config_dict() -> Dict:
    """Fixture providing a more complex configuration with multiple datasets."""
    return {
        "dataset_a": {
            "n_timesteps": 100,
            "n_samples": 50,
            "n_dimensions": 2,
            "random_state": 42,
            "classes": [
                {
                    "id": 0,
                    "signals": [
                        {
                            "function": "random_walk",
                            "params": {"step_size": 0.1},
                            "dimensions": [0],
                        },
                        {
                            "function": "gaussian_noise",
                            "params": {"sigma": 0.05},
                        },
                    ],
                },
                {
                    "id": 1,
                    "weight": 1.5,
                    "signals": [
                        {
                            "function": "random_walk",
                            "params": {"step_size": 0.1},
                        }
                    ],
                    "features": [
                        {
                            "function": "peak",
                            "params": {"amplitude": 1.5, "width": 3},
                            "length_pct": 0.1,
                            "random_location": True,
                            "dimensions": [0],
                            "shared_location": False,
                        }
                    ],
                },
            ],
        },
        "dataset_b": {
            "n_timesteps": 200,
            "n_samples": 30,
            "n_dimensions": 1,
            "classes": [
                {
                    "id": 0,
                    "signals": [{"function": "seasonal", "params": {"period": 20}}],
                }
            ],
        },
        "nested": {
            "datasets": {
                "dataset_c": {
                    "n_timesteps": 80,
                    "n_samples": 25,
                    "classes": [
                        {
                            "id": 0,
                            "signals": [
                                {"function": "constant", "params": {"value": 1.0}}
                            ],
                        }
                    ],
                }
            }
        },
    }


@pytest.fixture
def temp_config_file(complex_config_dict, tmp_path) -> Path:
    """Fixture creating a temporary YAML file with the complex configuration."""
    config_path = tmp_path / "test_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(complex_config_dict, f)
    return config_path


def test_create_single_builder_from_dict() -> None:
    """Test creating a builder from configuration dictionaries."""
    # Minimal config
    minimal_config = {
        "n_timesteps": 50,
        "n_samples": 20,
        "classes": [
            {
                "id": 0,
                "signals": [{"function": "random_walk", "params": {"step_size": 0.1}}],
            }
        ],
    }
    builder = _create_single_builder_from_dict(minimal_config)
    assert isinstance(builder, TimeSeriesBuilder)
    assert builder.n_timesteps == 50
    assert builder.n_samples == 20
    assert len(builder.class_definitions) == 1

    # Complex config with multiple classes, dimensions, features
    complex_config = {
        "n_timesteps": 100,
        "n_samples": 50,
        "n_dimensions": 2,
        "random_state": 42,
        "classes": [
            {
                "id": 0,
                "signals": [
                    {
                        "function": "random_walk",
                        "params": {"step_size": 0.1},
                        "dimensions": [0],
                    },
                    {
                        "function": "gaussian_noise",
                        "params": {"sigma": 0.05},
                    },
                ],
            },
            {
                "id": 1,
                "weight": 1.5,
                "signals": [
                    {
                        "function": "random_walk",
                        "params": {"step_size": 0.1},
                    }
                ],
                "features": [
                    {
                        "function": "peak",
                        "params": {"amplitude": 1.5, "width": 3},
                        "length_pct": 0.1,
                        "random_location": True,
                        "dimensions": [0],
                    }
                ],
            },
        ],
    }
    builder = _create_single_builder_from_dict(complex_config)
    assert builder.n_timesteps == 100
    assert builder.n_dimensions == 2
    assert builder.random_state == 42
    assert len(builder.class_definitions) == 2
    assert builder.class_definitions[1]["weight"] == 1.5
    assert len(builder.class_definitions[0]["components"]["background"]) == 2
    assert len(builder.class_definitions[1]["components"]["features"]) == 1


def test_create_single_builder_validation() -> None:
    """Test _create_single_builder_from_dict validation errors."""
    # Missing 'classes' key
    with pytest.raises(ValueError, match="must contain a 'classes' list"):
        _create_single_builder_from_dict({"n_timesteps": 50, "n_samples": 20})

    # Missing class 'id'
    with pytest.raises(ValueError, match="must have an 'id'"):
        _create_single_builder_from_dict(
            {
                "n_timesteps": 50,
                "n_samples": 20,
                "classes": [{"signals": [{"function": "random_walk"}]}],
            }
        )

    # Missing signal 'function'
    with pytest.raises(ValueError, match="must include a 'function' name"):
        _create_single_builder_from_dict(
            {
                "n_timesteps": 50,
                "n_samples": 20,
                "classes": [{"id": 0, "signals": [{"params": {"step_size": 0.1}}]}],
            }
        )

    # Non-existent function
    with pytest.raises(AttributeError, match="Could not find signal function"):
        _create_single_builder_from_dict(
            {
                "n_timesteps": 50,
                "n_samples": 20,
                "classes": [
                    {"id": 0, "signals": [{"function": "nonexistent_function"}]}
                ],
            }
        )


def test_load_builders_from_config_validation(
    minimal_config_dict, complex_config_dict
) -> None:
    """Test load_builders_from_config input validation errors."""
    # No source provided
    with pytest.raises(ValueError, match="Exactly one of"):
        load_builders_from_config()

    # Multiple sources provided
    with pytest.raises(ValueError, match="Exactly one of"):
        load_builders_from_config(
            config_dict=minimal_config_dict, config_str=yaml.dump(minimal_config_dict)
        )

    # Invalid file path
    with pytest.raises(FileNotFoundError, match="Configuration file not found"):
        load_builders_from_config(config_path="/nonexistent/path.yaml")

    # Invalid YAML string
    with pytest.raises(yaml.YAMLError):
        load_builders_from_config(config_str="dataset: {n_timesteps: 50, classes: [{")

    # Non-dictionary config_dict
    with pytest.raises(ValueError, match="config_dict must be a dictionary"):
        load_builders_from_config(config_dict="not a dictionary")

    # Invalid nested path_key
    with pytest.raises(ValueError, match="Path key .* not found in configuration"):
        load_builders_from_config(
            config_dict=complex_config_dict, path_key="nonexistent/path"
        )


def test_load_builders_from_config_filtering(complex_config_dict) -> None:
    """Test load_builders_from_config filtering with path_key and dataset_names."""
    # Loading from nested path_key
    builders = load_builders_from_config(
        config_dict=complex_config_dict, path_key="nested/datasets"
    )
    assert len(builders) == 1
    assert "dataset_c" in builders
    assert builders["dataset_c"].n_timesteps == 80

    # Loading specific datasets by name
    builders = load_builders_from_config(
        config_dict=complex_config_dict, dataset_names=["dataset_b"]
    )
    assert len(builders) == 1
    assert "dataset_b" in builders
    assert "dataset_a" not in builders

    # Nonexistent dataset returns empty dict
    builders = load_builders_from_config(
        config_dict=complex_config_dict, dataset_names=["nonexistent_dataset"]
    )
    assert len(builders) == 0


def test_load_builders_from_config_build_dataset(complex_config_dict) -> None:
    """Test that loaded builders can successfully build datasets."""
    builders = load_builders_from_config(config_dict=complex_config_dict)
    dataset = builders["dataset_a"].build()

    assert "X" in dataset
    assert "y" in dataset
    assert "components" in dataset
    assert dataset["X"].shape == (50, 2, 100)
    assert dataset["y"].shape == (50,)
    assert len(dataset["components"]) == 50
    assert not np.isnan(dataset["X"]).any()


def test_reproducibility_across_sources(complex_config_dict, temp_config_file) -> None:
    """Test that builders loaded from dict, file, and string produce identical datasets."""
    # Load from all three sources
    builders_dict = load_builders_from_config(config_dict=complex_config_dict)
    builders_file = load_builders_from_config(config_path=temp_config_file)
    builders_str = load_builders_from_config(config_str=yaml.dump(complex_config_dict))

    # All should have same datasets
    assert set(builders_dict.keys()) == set(builders_file.keys())
    assert set(builders_dict.keys()) == set(builders_str.keys())

    # Build with same random state and compare
    dataset_dict = builders_dict["dataset_a"].clone(random_state=123).build()
    dataset_file = builders_file["dataset_a"].clone(random_state=123).build()
    dataset_str = builders_str["dataset_a"].clone(random_state=123).build()

    np.testing.assert_array_equal(dataset_dict["X"], dataset_file["X"])
    np.testing.assert_array_equal(dataset_dict["X"], dataset_str["X"])


def test_yaml_anchors_and_merge_keys() -> None:
    """Test that YAML anchors (&) and merge keys (<<:) work correctly."""
    yaml_config = """
common: &common
  n_timesteps: 100
  n_samples: 50
  random_state: 42
  n_dimensions: 1

gaussian_signal: &gaussian_signal
  function: gaussian_noise
  params: { sigma: 0.5 }

level_shift: &level_shift
  function: constant
  params: { value: 1.0 }

short_feature: &short_feature
  length_pct: 0.2
  random_location: true

datasets:
  test_anchors:
    <<: *common
    classes:
      - id: 0
        signals: [*gaussian_signal]
        features:
          - <<: [*level_shift, *short_feature]
      - id: 1
        signals: [*gaussian_signal]
        features:
          - <<: *level_shift
            start_pct: 0.4
            end_pct: 0.6
"""
    builders = load_builders_from_config(config_str=yaml_config, path_key="datasets")

    assert "test_anchors" in builders
    builder = builders["test_anchors"]
    assert builder.n_timesteps == 100
    assert builder.n_samples == 50
    assert builder.random_state == 42

    dataset = builder.build()
    assert dataset["X"].shape == (50, 1, 100)
    assert len(np.unique(dataset["y"])) == 2
    assert len(dataset["feature_masks"]) == 2


def test_to_config() -> None:
    """Test that to_config() exports valid configuration dictionaries."""
    import xaitimesynth as xts

    # Basic config
    builder = (
        xts.TimeSeriesBuilder(n_timesteps=100, n_samples=50, random_state=42)
        .for_class(0)
        .add_signal(xts.gaussian_noise(sigma=0.1))
        .for_class(1)
        .add_signal(xts.gaussian_noise(sigma=0.1))
        .add_feature(xts.constant(value=1.0), start_pct=0.3, end_pct=0.6)
    )
    config = builder.to_config()

    assert config["n_timesteps"] == 100
    assert config["n_samples"] == 50
    assert config["random_state"] == 42
    assert len(config["classes"]) == 2
    assert config["classes"][0]["signals"][0]["function"] == "gaussian_noise"
    assert config["classes"][1]["features"][0]["start_pct"] == 0.3

    # Weights: default (1.0) should be omitted, non-default should be included
    builder_weights = (
        xts.TimeSeriesBuilder(n_timesteps=50, n_samples=20)
        .for_class(0, weight=1.0)
        .add_signal(xts.gaussian_noise(sigma=0.1))
        .for_class(1, weight=2.0)
        .add_signal(xts.gaussian_noise(sigma=0.1))
    )
    config_weights = builder_weights.to_config()
    assert "weight" not in config_weights["classes"][0]
    assert config_weights["classes"][1]["weight"] == 2.0

    # Multivariate with dimension-specific features
    builder_mv = (
        xts.TimeSeriesBuilder(n_timesteps=100, n_samples=40, n_dimensions=3)
        .for_class(0)
        .add_signal(xts.random_walk(step_size=0.1), dim=[0, 1, 2])
        .add_feature(xts.constant(value=1.0), start_pct=0.4, end_pct=0.6, dim=[0])
    )
    config_mv = builder_mv.to_config()
    assert config_mv["n_dimensions"] == 3
    assert "dimensions" in config_mv["classes"][0]["features"][0]


def test_to_config_round_trip() -> None:
    """Test that to_config() output can be loaded back to create equivalent datasets."""
    import xaitimesynth as xts

    original_builder = (
        xts.TimeSeriesBuilder(
            n_timesteps=80, n_samples=30, n_dimensions=2, random_state=123
        )
        .for_class(0)
        .add_signal(xts.random_walk(step_size=0.1), dim=[0, 1])
        .add_signal(xts.gaussian_noise(sigma=0.05))
        .for_class(1)
        .add_signal(xts.random_walk(step_size=0.1), dim=[0, 1])
        .add_signal(xts.gaussian_noise(sigma=0.05))
        .add_feature(
            xts.peak(amplitude=1.5, width=3), length_pct=0.2, random_location=True
        )
    )

    config = original_builder.to_config()
    reloaded_builder = load_builders_from_config(config_dict={"test_dataset": config})[
        "test_dataset"
    ]

    original_dataset = original_builder.clone(random_state=999).build()
    reloaded_dataset = reloaded_builder.clone(random_state=999).build()

    assert original_dataset["X"].shape == reloaded_dataset["X"].shape
    assert original_dataset["y"].shape == reloaded_dataset["y"].shape
    assert set(original_dataset["feature_masks"].keys()) == set(
        reloaded_dataset["feature_masks"].keys()
    )
