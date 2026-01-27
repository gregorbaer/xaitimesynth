from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import yaml

# Import the main package namespace to access component functions dynamically
import xaitimesynth


# Renamed original function to be internal, handles a single dataset config dict
def _create_single_builder_from_dict(
    dataset_config: Dict[str, Any],
) -> "xaitimesynth.TimeSeriesBuilder":
    """Creates a single TimeSeriesBuilder from its configuration dictionary.

    Args:
        dataset_config (Dict[str, Any]): A dictionary containing the configuration
            for a single dataset builder.

    Returns:
        TimeSeriesBuilder: A configured TimeSeriesBuilder instance.

    Raises:
        ValueError: If the configuration dictionary is missing required keys or
                    contains invalid values.
        AttributeError: If a specified component function name does not exist in
                        the xaitimesynth package.
    """
    # --- 1. Initialize Builder ---
    builder_args = {
        "n_timesteps": dataset_config.get("n_timesteps", 100),
        "n_samples": dataset_config.get("n_samples", 1000),
        "n_dimensions": dataset_config.get("n_dimensions", 1),
        "normalization": dataset_config.get("normalization", "zscore"),
        "random_state": dataset_config.get("random_state"),
        "normalization_kwargs": dataset_config.get("normalization_kwargs", {}),
        "feature_fill_value": dataset_config.get("feature_fill_value", np.nan),
        "foundation_fill_value": dataset_config.get("foundation_fill_value", 0.0),
        "data_format": dataset_config.get("data_format", "channels_first"),
    }
    builder = xaitimesynth.TimeSeriesBuilder(**builder_args)

    # --- 2. Configure Classes ---
    if "classes" not in dataset_config:
        raise ValueError("Dataset configuration must contain a 'classes' list.")

    for class_config in dataset_config["classes"]:
        class_label = class_config.get("id")
        if class_label is None:
            raise ValueError("Each class definition must have an 'id'.")
        weight = class_config.get("weight", 1.0)
        builder.for_class(class_label, weight=weight)

        # --- 2a. Add Signals ---
        for signal_config in class_config.get("signals", []):
            func_name = signal_config.get("function")
            if not func_name:
                raise ValueError("Signal definition must include a 'function' name.")

            try:
                # Dynamically get the component function from the xaitimesynth package
                func = getattr(xaitimesynth, func_name)
            except AttributeError:
                raise AttributeError(
                    f"Could not find signal function '{func_name}' in xaitimesynth."
                )

            params = signal_config.get("params", {})
            component = func(**params)  # Create the component definition dict

            # Extract common and segment-specific arguments
            dim = signal_config.get("dimensions")  # None if not present or null

            # Default shared_randomness based on whether dimensions are specified.
            # If dim is None, default to True (apply same component across all dims).
            # If dim is specified, default to False (apply potentially different
            # randomness per specified dim, unless overridden in config).
            if dim is None:
                shared_randomness = signal_config.get("shared_randomness", True)
            else:
                shared_randomness = signal_config.get("shared_randomness", False)

            start_pct = signal_config.get("start_pct")
            end_pct = signal_config.get("end_pct")
            length_pct = signal_config.get("length_pct")
            random_location = signal_config.get("random_location", False)
            shared_location = signal_config.get("shared_location", True)

            # add_signal() handles both full-series and segment modes
            builder.add_signal(
                component,
                dim=dim,
                shared_randomness=shared_randomness,
                start_pct=start_pct,
                end_pct=end_pct,
                length_pct=length_pct,
                random_location=random_location,
                shared_location=shared_location,
            )

        # --- 2b. Add Features ---
        for feature_config in class_config.get("features", []):
            func_name = feature_config.get("function")
            if not func_name:
                raise ValueError("Feature definition must include a 'function' name.")

            try:
                # Dynamically get the component function
                func = getattr(xaitimesynth, func_name)
            except AttributeError:
                raise AttributeError(
                    f"Could not find feature function '{func_name}' in xaitimesynth."
                )

            params = feature_config.get("params", {})
            component = func(**params)  # Create the component definition dict

            # Extract feature arguments
            start_pct = feature_config.get("start_pct")
            end_pct = feature_config.get("end_pct")
            length_pct = feature_config.get("length_pct")
            random_location = feature_config.get("random_location", False)
            dim = feature_config.get("dimensions")  # None if not present or null
            shared_location = feature_config.get("shared_location", True)
            shared_randomness = feature_config.get("shared_randomness", False)

            builder.add_feature(
                component,
                start_pct=start_pct,
                end_pct=end_pct,
                length_pct=length_pct,
                random_location=random_location,
                dim=dim,
                shared_location=shared_location,
                shared_randomness=shared_randomness,
            )

    return builder


def load_builders_from_config(
    config_path: Optional[Union[str, Path]] = None,
    config_dict: Optional[Dict[str, Any]] = None,
    config_str: Optional[str] = None,
    path_key: Optional[str] = None,
    dataset_names: Optional[List[str]] = None,
) -> Dict[str, "xaitimesynth.TimeSeriesBuilder"]:
    """Loads and creates TimeSeriesBuilder instances from various configuration sources.

    This function can load configurations from a dictionary, a YAML file path,
    or a string containing YAML content. Exactly one of `config_path`,
    `config_dict`, or `config_str` must be provided.

    Args:
        config_path (Optional[Union[str, Path]]): Path to a YAML configuration file.
        config_dict (Optional[Dict[str, Any]]): A dictionary containing the configuration.
        config_str (Optional[str]): A string containing YAML configuration.
        path_key (Optional[str]): A key (or path using '/' as separator) within the
            configuration dictionary where the dataset definitions are located.
            If None, assumes the top-level dictionary contains the dataset definitions.
            Example: "experiments/datasets". Default is None.
        dataset_names (Optional[List[str]]): A list of specific dataset names to load.
            If None, all datasets found at the specified location are loaded.
            Default is None.

    Returns:
        Dict[str, TimeSeriesBuilder]: A dictionary where keys are the dataset names
        and values are the configured TimeSeriesBuilder instances.

    Raises:
        ValueError: If not exactly one configuration source is provided, if the
                    configuration source is invalid, the path_key does not lead
                    to a dictionary, or required keys are missing.
        FileNotFoundError: If config_path is provided and the file does not exist.
        yaml.YAMLError: If config_str or the file at config_path contains invalid YAML.
        AttributeError: If a specified component function name does not exist in
                        the xaitimesynth package.

    Detailed Configuration Structure:
        The configuration (whether from file, string, or dict) must ultimately resolve
        to a Python dictionary. This dictionary contains dataset definitions, either at
        the top level or nested under the `path_key`.

        Each dataset definition (the value associated with a dataset name key) is a
        dictionary specifying the parameters for a `TimeSeriesBuilder` and its components.
        Key elements include:
        - Builder arguments: `n_timesteps`, `n_samples`, `n_dimensions`, `random_state`, etc.
        - `classes` (list, mandatory): A list of dictionaries, each defining a class.
            - `id` (mandatory): The class label.
            - `weight` (float, optional): Sampling weight for the class.
            - `signals` (list, optional): List of signal component dictionaries.
                - `function` (str, mandatory): Name of a signal generator function (e.g., "random_walk").
                - `params` (dict, optional): Parameters for the generator function.
                - `dimensions` (list, optional): Dimensions to apply to.
                - `shared_randomness` (bool, optional).
                - Location keys (optional): `start_pct`, `end_pct`, `length_pct`, `random_location`, `shared_location`.
            - `features` (list, optional): List of feature component dictionaries.
                - `function` (str, mandatory): Name of a feature generator function (e.g., "peak").
                - `params` (dict, optional): Parameters for the generator function.
                - Location keys (optional): `start_pct`, `end_pct`, `length_pct`, `random_location`, `shared_location`.
                - `dimensions` (list, optional): Dimensions to apply to.
                - `shared_randomness` (bool, optional).


    Example YAML Structure (config.yaml):
        ```yaml
        # Option 1: Top-level dataset definition (path_key=None)
        my_dataset_1:
          n_timesteps: 150
          n_samples: 200
          n_dimensions: 2
          random_state: 42
          classes:
            - id: 0 # Class 0 definition
              weight: 1.0
              signals:
                - function: random_walk
                  params: { step_size: 0.1 }
                  dimensions: [0, 1] # Apply to both dimensions
                - function: gaussian
                  params: { sigma: 0.05 }
                  # dimensions omitted -> applies to all
              features: [] # No specific features for class 0

            - id: 1 # Class 1 definition
              weight: 1.5 # Sample class 1 more often
              signals:
                - { function: random_walk, params: { step_size: 0.1 }, dimensions: [0, 1] }
                - { function: gaussian, params: { sigma: 0.05 } }
              features:
                - function: peak
                  params: { amplitude: 1.5, width: 3 }
                  length_pct: 0.1 # Feature length is 10% of total timesteps
                  random_location: true # Place it randomly
                  dimensions: [0] # Only in dimension 0
                  shared_location: false # If dim had >1 element, location would differ
                - function: constant
                  params: { value: -1.0 }
                  start_pct: 0.7
                  end_pct: 0.9
                  dimensions: [1] # Only in dimension 1

        # Option 2: Nested dataset definitions (path_key="experiments/datasets")
        experiments:
          datasets:
            dataset_nested:
              n_timesteps: 80
              n_samples: 50
              classes:
                - id: 0
                  signals: [ { function: seasonal, params: { period: 10 } } ]
                # ... potentially more classes ...
        ```

    YAML Anchors and Aliases:
        YAML's anchor/alias feature can be used to reuse configuration across multiple datasets.
        This is particularly useful for defining common settings, signals, or features.

        Example:
        ```yaml
        # Define common settings with anchor (&)
        common: &common_settings
          n_timesteps: 100
          n_samples: 1000
          random_state: 42
          normalization: "zscore"

        # Define common signal configuration
        base_random_walk: &base_signal
          function: random_walk
          params:
            step_size: 0.1

        # Use aliases (*) to reference the anchors
        dataset_a:
          <<: *common_settings  # Merges all common settings
          n_dimensions: 1
          classes:
            - id: 0
              signals:
                - <<: *base_signal  # Use the common signal definition

        dataset_b:
          <<: *common_settings
          n_samples: 2000  # Override specific settings
          n_dimensions: 2
          classes:
            - id: 0
              signals:
                - <<: *base_signal
                  dimensions: [0, 1]  # Add dimensions parameter
        ```

        The `<<:` syntax is a YAML merge key that merges all key-value pairs from the
        referenced anchor into the current mapping.

    Example Usage:
        ```python
        from xaitimesynth.parser import load_builders_from_config

        # Load all datasets from top level of a file
        builders_file = load_builders_from_config(config_path="config.yaml")

        # Load only 'dataset_c' from a nested path in a file
        builders_c = load_builders_from_config(
            config_path="config.yaml",
            path_key="experiments/datasets",
            dataset_names=["dataset_c"]
        )

        # Load from a dictionary
        my_config = {
            "my_dataset": {"n_timesteps": 10, "classes": [{"id": 0}]}
        }
        builders_dict = load_builders_from_config(config_dict=my_config)

        # Load from a YAML string
        yaml_str = "my_data:\n  n_timesteps: 5"
        builders_str = load_builders_from_config(config_str=yaml_str)
        ```
    """
    # --- 1. Validate and Load configuration dictionary ---
    provided_configs = sum(
        arg is not None for arg in [config_path, config_dict, config_str]
    )
    if provided_configs != 1:
        raise ValueError(
            "Exactly one of config_path, config_dict, or config_str must be provided."
        )

    loaded_config_dict: Dict[str, Any]

    if config_dict is not None:
        if not isinstance(config_dict, dict):
            raise ValueError("config_dict must be a dictionary.")
        loaded_config_dict = config_dict
    elif config_str is not None:
        try:
            loaded_config_dict = yaml.safe_load(config_str)
            if not isinstance(loaded_config_dict, dict):
                raise ValueError(
                    "config_str is valid YAML but not a dictionary-based config."
                )
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Could not parse config_str as YAML: {e}")
        except Exception as e:
            raise ValueError(f"Could not load config from config_str: {e}")
    elif config_path is not None:
        path = Path(config_path)
        if not path.is_file():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        try:
            with open(path, "r") as f:
                loaded_config_dict = yaml.safe_load(f)
            if not isinstance(loaded_config_dict, dict):
                raise ValueError(
                    f"File at {path} is valid YAML but not a dictionary-based config."
                )
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Could not parse file {path} as YAML: {e}")
        except Exception as e:
            raise ValueError(f"Could not load config from file {path}: {e}")
    # This else should be unreachable due to the initial check, but added for safety
    else:
        raise ValueError("Internal error: No configuration source identified.")

    # --- 2. Locate the dataset definitions within the dictionary ---
    datasets_dict = loaded_config_dict
    if path_key:
        keys = path_key.split("/")
        try:
            for key in keys:
                datasets_dict = datasets_dict[key]
        except KeyError:
            raise ValueError(f"Path key '{path_key}' not found in configuration.")
        except TypeError:
            raise ValueError(f"Element at path '{path_key}' is not a dictionary.")

    if not isinstance(datasets_dict, dict):
        raise ValueError(
            f"Configuration at path '{path_key or 'top-level'}' is not a dictionary of datasets."
        )

    # --- 3. Filter and Create Builders ---
    builders = {}
    datasets_to_load = (
        dataset_names if dataset_names is not None else datasets_dict.keys()
    )

    for name in datasets_to_load:
        if name not in datasets_dict:
            print(
                f"Warning: Dataset '{name}' requested but not found in configuration."
            )
            continue

        single_dataset_config = datasets_dict[name]
        if not isinstance(single_dataset_config, dict):
            print(
                f"Warning: Configuration for dataset '{name}' is not a dictionary. Skipping."
            )
            continue

        # Check if the dictionary looks like a dataset config (must have 'classes')
        if "classes" not in single_dataset_config:
            print(
                f"Warning: Configuration for '{name}' does not contain a 'classes' key. Skipping."
            )
            continue

        try:
            builders[name] = _create_single_builder_from_dict(single_dataset_config)
        except (ValueError, AttributeError) as e:
            print(f"Error creating builder for dataset '{name}': {e}")
            # Re-raise the exception after printing the context
            raise ValueError(f"Error processing dataset '{name}': {e}") from e

    return builders
