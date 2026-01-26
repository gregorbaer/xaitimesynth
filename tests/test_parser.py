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
                            "role": "foundation",
                            "dimensions": [0],
                        },
                        {
                            "function": "gaussian",
                            "params": {"sigma": 0.05},
                            "role": "noise",
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
                            "role": "foundation",
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


def test_create_single_builder_from_dict_minimal() -> None:
    """Test creating a builder from a minimal configuration dictionary."""
    config = {
        "n_timesteps": 50,
        "n_samples": 20,
        "classes": [
            {
                "id": 0,
                "signals": [{"function": "random_walk", "params": {"step_size": 0.1}}],
            }
        ],
    }

    builder = _create_single_builder_from_dict(config)

    # Verify builder properties
    assert isinstance(builder, TimeSeriesBuilder), "Should return a TimeSeriesBuilder"
    assert builder.n_timesteps == 50, "n_timesteps should be set correctly"
    assert builder.n_samples == 20, "n_samples should be set correctly"

    # Verify class definition
    assert len(builder.class_definitions) == 1, "Should have one class definition"
    assert builder.class_definitions[0]["label"] == 0, "Class label should be 0"

    # Verify components
    assert len(builder.class_definitions[0]["components"]["foundation"]) == 1, (
        "Should have one foundation component"
    )
    assert "random_walk" in str(
        builder.class_definitions[0]["components"]["foundation"][0]
    ), "Component should be random_walk"


def test_create_single_builder_from_dict_complex() -> None:
    """Test creating a builder from a more complex configuration dictionary."""
    config = {
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
                        "role": "foundation",
                        "dimensions": [0],
                    },
                    {
                        "function": "gaussian",
                        "params": {"sigma": 0.05},
                        "role": "noise",
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
                        "role": "foundation",
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
    }

    builder = _create_single_builder_from_dict(config)

    # Verify builder properties
    assert builder.n_timesteps == 100
    assert builder.n_samples == 50
    assert builder.n_dimensions == 2
    assert builder.random_state == 42

    # Verify class definitions
    assert len(builder.class_definitions) == 2
    assert builder.class_definitions[0]["label"] == 0
    assert builder.class_definitions[1]["label"] == 1
    assert builder.class_definitions[1]["weight"] == 1.5

    # Verify components for class 0
    class0 = builder.class_definitions[0]
    assert len(class0["components"]["foundation"]) == 1
    assert len(class0["components"]["noise"]) == 1
    assert len(class0["components"]["features"]) == 0

    # Verify components for class 1
    class1 = builder.class_definitions[1]
    # If dimensions are not specified for a signal, shared_randomness defaults to True,
    # resulting in a single component entry covering all dimensions.
    assert len(class1["components"]["foundation"]) == 1, (
        "Expected 1 foundation component for class 1 when dimensions are omitted (shared_randomness=True)"
    )
    assert len(class1["components"]["features"]) == 1


def test_create_single_builder_missing_classes() -> None:
    """Test that an error is raised when the 'classes' key is missing."""
    config = {"n_timesteps": 50, "n_samples": 20}

    with pytest.raises(ValueError, match="must contain a 'classes' list"):
        _create_single_builder_from_dict(config)


def test_create_single_builder_missing_class_id() -> None:
    """Test that an error is raised when a class definition is missing an 'id'."""
    config = {
        "n_timesteps": 50,
        "n_samples": 20,
        "classes": [
            {
                # Missing id
                "signals": [{"function": "random_walk", "params": {"step_size": 0.1}}]
            }
        ],
    }

    with pytest.raises(ValueError, match="must have an 'id'"):
        _create_single_builder_from_dict(config)


def test_create_single_builder_missing_signal_function() -> None:
    """Test that an error is raised when a signal is missing a 'function'."""
    config = {
        "n_timesteps": 50,
        "n_samples": 20,
        "classes": [
            {
                "id": 0,
                "signals": [
                    {
                        # Missing function
                        "params": {"step_size": 0.1}
                    }
                ],
            }
        ],
    }

    with pytest.raises(ValueError, match="must include a 'function' name"):
        _create_single_builder_from_dict(config)


def test_create_single_builder_nonexistent_function() -> None:
    """Test that an error is raised when a non-existent function is specified."""
    config = {
        "n_timesteps": 50,
        "n_samples": 20,
        "classes": [
            {"id": 0, "signals": [{"function": "nonexistent_function", "params": {}}]}
        ],
    }

    with pytest.raises(AttributeError, match="Could not find signal function"):
        _create_single_builder_from_dict(config)


def test_load_builders_from_config_dict(minimal_config_dict) -> None:
    """Test loading builders from a dictionary."""
    builders = load_builders_from_config(config_dict=minimal_config_dict)

    assert len(builders) == 1, "Should return one builder"
    assert "test_dataset" in builders, "Builder should be keyed by dataset name"
    assert isinstance(builders["test_dataset"], TimeSeriesBuilder), (
        "Value should be a TimeSeriesBuilder"
    )
    assert builders["test_dataset"].n_timesteps == 50, (
        "Builder should have correct parameters"
    )


def test_load_builders_from_config_file(temp_config_file) -> None:
    """Test loading builders from a YAML file."""
    builders = load_builders_from_config(config_path=temp_config_file)

    assert len(builders) == 2, "Should return two builders (excluding nested)"
    assert "dataset_a" in builders, "Should include dataset_a"
    assert "dataset_b" in builders, "Should include dataset_b"
    assert "dataset_c" not in builders, "Should not include nested dataset_c by default"

    # Verify properties of one builder
    assert builders["dataset_a"].n_timesteps == 100
    assert builders["dataset_a"].n_samples == 50
    assert builders["dataset_a"].n_dimensions == 2


def test_load_builders_from_config_string(complex_config_dict) -> None:
    """Test loading builders from a YAML string."""
    config_str = yaml.dump(complex_config_dict)
    builders = load_builders_from_config(config_str=config_str)

    assert len(builders) == 2, "Should return two builders (excluding nested)"
    assert "dataset_a" in builders, "Should include dataset_a"
    assert "dataset_b" in builders, "Should include dataset_b"

    # Verify properties of one builder
    assert builders["dataset_b"].n_timesteps == 200
    assert builders["dataset_b"].n_samples == 30
    assert builders["dataset_b"].n_dimensions == 1


def test_load_builders_from_config_no_source() -> None:
    """Test that an error is raised when no configuration source is provided."""
    with pytest.raises(
        ValueError,
        match="Exactly one of config_path, config_dict, or config_str must be provided",
    ):
        load_builders_from_config()


def test_load_builders_from_config_multiple_sources(minimal_config_dict) -> None:
    """Test that an error is raised when multiple configuration sources are provided."""
    with pytest.raises(
        ValueError,
        match="Exactly one of config_path, config_dict, or config_str must be provided",
    ):
        load_builders_from_config(
            config_dict=minimal_config_dict, config_str=yaml.dump(minimal_config_dict)
        )


def test_load_builders_from_config_invalid_path() -> None:
    """Test that an error is raised when an invalid file path is provided."""
    with pytest.raises(FileNotFoundError, match="Configuration file not found"):
        load_builders_from_config(config_path="/nonexistent/path.yaml")


def test_load_builders_from_config_invalid_yaml_string() -> None:
    """Test that an error is raised when invalid YAML is provided as a string."""
    invalid_yaml = "dataset: {n_timesteps: 50, classes: [{id: 0, signals: [{"

    with pytest.raises(yaml.YAMLError):
        load_builders_from_config(config_str=invalid_yaml)


def test_load_builders_from_config_invalid_dict() -> None:
    """Test that an error is raised when config_dict is not a dictionary."""
    with pytest.raises(ValueError, match="config_dict must be a dictionary"):
        load_builders_from_config(config_dict="not a dictionary")


def test_load_builders_from_config_nested_path(complex_config_dict) -> None:
    """Test loading builders from a nested path within the configuration."""
    builders = load_builders_from_config(
        config_dict=complex_config_dict, path_key="nested/datasets"
    )

    assert len(builders) == 1, "Should return one builder"
    assert "dataset_c" in builders, "Should include dataset_c"
    assert builders["dataset_c"].n_timesteps == 80, (
        "Builder should have correct parameters"
    )


def test_load_builders_from_config_specific_datasets(complex_config_dict) -> None:
    """Test loading only specific datasets."""
    builders = load_builders_from_config(
        config_dict=complex_config_dict, dataset_names=["dataset_b"]
    )

    assert len(builders) == 1, "Should return one builder"
    assert "dataset_b" in builders, "Should include dataset_b"
    assert "dataset_a" not in builders, "Should not include dataset_a"


def test_load_builders_from_config_invalid_nested_path(complex_config_dict) -> None:
    """Test that an error is raised when an invalid nested path is provided."""
    with pytest.raises(ValueError, match="Path key .* not found in configuration"):
        load_builders_from_config(
            config_dict=complex_config_dict, path_key="nonexistent/path"
        )


def test_load_builders_from_config_nonexistent_dataset(complex_config_dict) -> None:
    """Test behavior when a requested dataset doesn't exist."""
    # This should not raise an error, just a warning (printed to stdout)
    builders = load_builders_from_config(
        config_dict=complex_config_dict, dataset_names=["nonexistent_dataset"]
    )

    assert len(builders) == 0, (
        "Should return empty dictionary when dataset doesn't exist"
    )


def test_load_builders_from_config_build_dataset(complex_config_dict) -> None:
    """Test that loaded builders can successfully build datasets."""
    builders = load_builders_from_config(config_dict=complex_config_dict)

    # Build dataset from dataset_a
    dataset = builders["dataset_a"].build()

    # Verify the dataset structure
    assert "X" in dataset, "Dataset should contain X"
    assert "y" in dataset, "Dataset should contain y"
    assert "components" in dataset, "Dataset should contain components"

    # Verify shapes
    assert dataset["X"].shape == (50, 2, 100), (
        "X should have shape (n_samples, n_dimensions, n_timesteps)"
    )
    assert dataset["y"].shape == (50,), "y should have shape (n_samples,)"

    # Since we specified random_state=42, we should get deterministic results
    # Test a few samples just to ensure the build process worked
    assert len(dataset["components"]) == 50, "Should have components for each sample"

    # At least ensure there's no NaN in the data (would indicate generation issues)
    assert not np.isnan(dataset["X"]).any(), "Dataset should not contain NaN values"


def test_reproducibility_across_sources(complex_config_dict, temp_config_file) -> None:
    """Test that builders loaded from different sources with the same config build the same dataset."""
    # Load from dictionary
    builders_dict = load_builders_from_config(config_dict=complex_config_dict)

    # Load from file
    builders_file = load_builders_from_config(config_path=temp_config_file)

    # Load from string
    config_str = yaml.dump(complex_config_dict)
    builders_str = load_builders_from_config(config_str=config_str)

    # Build datasets with the same random state
    dataset_dict = builders_dict["dataset_a"].clone(random_state=123).build()
    dataset_file = builders_file["dataset_a"].clone(random_state=123).build()
    dataset_str = builders_str["dataset_a"].clone(random_state=123).build()

    # Compare datasets
    np.testing.assert_array_equal(
        dataset_dict["X"],
        dataset_file["X"],
        err_msg="Datasets from dict and file should be identical",
    )

    np.testing.assert_array_equal(
        dataset_dict["X"],
        dataset_str["X"],
        err_msg="Datasets from dict and string should be identical",
    )
