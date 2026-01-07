"""
Configuration management for LUCS AlphaFold analysis tools.

Author: Ben Orr
Date: 12.18.25

This module provides a flexible configuration system that supports:
- YAML configuration files
- Environment variable overrides
- Command-line argument integration
- Sensible default values

Usage
-----
Basic usage with yaml file:
    >>> config = Config.from_yaml("my_config.yaml")
    >>> print(config.paths.starting_structure_path)
    /path/to/structure.pdb

With environment variable overrides:
    >>> import os
    >>> os.environ['LUCS_START_STRUCT'] = '/custom/path.pdb'
    >>> config = Config.from_yaml("config.yaml")
    >>> print(config.paths.starting_structure_path)
    /custom/path.pdb

Integrating with argparse:
    >>> import argparse
    >>> parser = argparse.ArgumentParser()
    >>> config = Config()
    >>> config.add_to_argparse(parser, 'starting_structure')
    >>> args = parser.parse_args()
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional, Union
from dataclasses import dataclass, field, asdict


@dataclass
class PathConfig:
    """
    Configuration for common file paths.

    Attributes
    ----------
    starting_structure_path : str
        Path to the LUCS starting structure pdb file.
    starting_structure_insertion_points : str
        Path to the insertion points json file for the starting structure.
    data_dir : str
        Base directory for data files.
    output_dir : str
        Base directory for output files.
    """
    starting_structure_path: str = ""
    starting_structure_insertion_points: str = ""
    data_dir: str = "./data"
    output_dir: str = "./output"


@dataclass
class AnalysisConfig:
    """
    Configuration for analysis parameters.

    Attributes
    ----------
    filter_by_subsequence : str
        Filter designs by required subsequence.
    start_struct_res_start_idx : int
        Starting residue index for the starting structure.
    verbose : bool
        Enable verbose output.
    num_tasks : int
        Number of parallel tasks for job distribution.
    task_id : int
        Current task ID (1-indexed) for parallel execution.
    """
    filter_by_subsequence: str = ""
    start_struct_res_start_idx: int = 1
    verbose: bool = False
    num_tasks: int = 1
    task_id: int = 1


@dataclass
class AlphaFoldConfig:
    """
    Configuration for AlphaFold-related parameters.

    Attributes
    ----------
    model_type : str
        AlphaFold model type ('af2' or 'af3').
    use_best_model_only : bool
        Whether to analyze only the best ranked model.
    confidence_threshold : float
        pLDDT threshold for confidence filtering.
    """
    model_type: str = "af2"
    use_best_model_only: bool = False
    confidence_threshold: float = 70.0


@dataclass
class Config:
    """
    Main configuration class for LUCS AlphaFold analysis.

    This class consolidates all configuration settings and provides methods
    for loading from yaml files, environment variables, and integration with
    command-line argument parsers.

    Attributes
    ----------
    paths : PathConfig
        File path configuration.
    analysis : AnalysisConfig
        Analysis parameter configuration.
    alphafold : AlphaFoldConfig
        AlphaFold-specific configuration.

    Examples
    --------
    >>> # Load from YAML
    >>> config = Config.from_yaml("config.yaml")
    >>>
    >>> # Create with custom values
    >>> config = Config()
    >>> config.paths.data_dir = "/my/data"
    >>> config.analysis.verbose = True
    >>>
    >>> # Save to YAML
    >>> config.to_yaml("my_config.yaml")
    """
    paths: PathConfig = field(default_factory=PathConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    alphafold: AlphaFoldConfig = field(default_factory=AlphaFoldConfig)

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> 'Config':
        """
        Load configuration from a yaml file.

        Parameters
        ----------
        yaml_path : str or Path
            Path to yaml configuration file.

        Returns
        -------
        Config
            Configuration object with values from yaml file.

        Examples
        --------
        >>> config = Config.from_yaml("config.yaml")
        >>> print(config.paths.data_dir)
        /path/to/data
        """
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Config file not found: {yaml_path}")

        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        # Create config with nested dataclasses
        config = cls()
        if 'paths' in config_dict:
            config.paths = PathConfig(**config_dict['paths'])
        if 'analysis' in config_dict:
            config.analysis = AnalysisConfig(**config_dict['analysis'])
        if 'alphafold' in config_dict:
            config.alphafold = AlphaFoldConfig(**config_dict['alphafold'])

        # Apply environment variable overrides
        config._apply_env_overrides()

        return config

    def _apply_env_overrides(self) -> None:
        """
        Override configuration values with environment variables.

        Environment variable format: LUCS_<SECTION>_<KEY>
        Examples:
            LUCS_PATHS_DATA_DIR
            LUCS_ANALYSIS_VERBOSE
            LUCS_ALPHAFOLD_MODEL_TYPE

        Notes
        -----
        This method is called automatically by from_yaml() but can also
        be called manually to update configuration from environment.
        """
        # Path overrides
        env_mappings = {
            'LUCS_START_STRUCT': ('paths', 'starting_structure_path'),
            'LUCS_START_STRUCT_INSERTIONS': ('paths', 'starting_structure_insertion_points'),
            'LUCS_DATA_DIR': ('paths', 'data_dir'),
            'LUCS_OUTPUT_DIR': ('paths', 'output_dir'),
            'LUCS_VERBOSE': ('analysis', 'verbose'),
            'LUCS_NUM_TASKS': ('analysis', 'num_tasks'),
            'LUCS_TASK_ID': ('analysis', 'task_id'),
            'LUCS_AF_MODEL_TYPE': ('alphafold', 'model_type'),
        }

        for env_var, (section, key) in env_mappings.items():
            if env_var in os.environ:
                value = os.environ[env_var]
                section_obj = getattr(self, section)

                # Type conversion based on current value type
                current_value = getattr(section_obj, key)
                if isinstance(current_value, bool):
                    value = value.lower() in ('true', '1', 'yes')
                elif isinstance(current_value, int):
                    value = int(value)
                elif isinstance(current_value, float):
                    value = float(value)

                setattr(section_obj, key, value)

    def to_yaml(self, yaml_path: Union[str, Path]) -> None:
        """
        Save configuration to a yaml file.

        Parameters
        ----------
        yaml_path : str or Path
            Path where yaml file will be saved.

        Examples
        --------
        >>> config = Config()
        >>> config.paths.data_dir = "/my/data"
        >>> config.to_yaml("my_config.yaml")
        """
        yaml_path = Path(yaml_path)
        config_dict = {
            'paths': asdict(self.paths),
            'analysis': asdict(self.analysis),
            'alphafold': asdict(self.alphafold),
        }

        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        with open(yaml_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to a dictionary.

        Returns
        -------
        Dict[str, Any]
            Dictionary representation of the configuration.

        Examples
        --------
        >>> config = Config()
        >>> config_dict = config.to_dict()
        >>> print(config_dict['paths']['data_dir'])
        ./data
        """
        return {
            'paths': asdict(self.paths),
            'analysis': asdict(self.analysis),
            'alphafold': asdict(self.alphafold),
        }

    def update_from_args(self, args: Any) -> None:
        """
        Update configuration from argparse Namespace.

        Parameters
        ----------
        args : argparse.Namespace
            Parsed command-line arguments.

        Examples
        --------
        >>> import argparse
        >>> parser = argparse.ArgumentParser()
        >>> parser.add_argument('--data_dir', type=str)
        >>> parser.add_argument('--verbose', action='store_true')
        >>> args = parser.parse_args(['--data_dir', '/data', '--verbose'])
        >>> config = Config()
        >>> config.update_from_args(args)
        >>> print(config.paths.data_dir)
        /data
        >>> print(config.analysis.verbose)
        True
        """
        # Map argparse attributes to config attributes
        arg_mappings = {
            # Paths
            'starting_structure_path': ('paths', 'starting_structure_path'),
            'starting_structure_insertion_points': ('paths', 'starting_structure_insertion_points'),
            'data_dir': ('paths', 'data_dir'),
            'outfolder': ('paths', 'output_dir'),
            # Analysis
            'filter_by_subsequence': ('analysis', 'filter_by_subsequence'),
            'start_struct_res_start_idx': ('analysis', 'start_struct_res_start_idx'),
            'verbose': ('analysis', 'verbose'),
            'num_tasks': ('analysis', 'num_tasks'),
            'task_id': ('analysis', 'task_id'),
            # AlphaFold
            'model_type': ('alphafold', 'model_type'),
        }

        for arg_name, (section, key) in arg_mappings.items():
            if hasattr(args, arg_name):
                value = getattr(args, arg_name)
                if value is not None:  # Only update if value was provided
                    section_obj = getattr(self, section)
                    setattr(section_obj, key, value)

    @staticmethod
    def get_default_config_path() -> Path:
        """
        Get the default configuration file path.

        Searches for config.yaml in:
        1. Current working directory
        2. User's home directory (~/.lucs_af/config.yaml)
        3. Script's directory

        Returns
        -------
        Path
            Path to config file, or None if not found.

        Examples
        --------
        >>> config_path = Config.get_default_config_path()
        >>> if config_path and config_path.exists():
        ...     config = Config.from_yaml(config_path)
        """
        search_paths = [
            Path.cwd() / "config.yaml",
            Path.home() / ".lucs_af" / "config.yaml",
            Path(__file__).parent / "config.yaml",
        ]

        for path in search_paths:
            if path.exists():
                return path

        return None


def load_config(
    yaml_path: Optional[Union[str, Path]] = None,
    use_defaults: bool = True
) -> Config:
    """
    Convenience function to load configuration.

    Parameters
    ----------
    yaml_path : str or Path, optional
        Path to YAML config file. If None, searches for default config.
    use_defaults : bool, optional
        If True and no config file found, returns Config with defaults.
        If False and no config file found, raises FileNotFoundError.

    Returns
    -------
    Config
        Loaded configuration object.

    Raises
    ------
    FileNotFoundError
        If yaml_path is None, no default config found, and use_defaults=False.

    Examples
    --------
    >>> # Load from specific file
    >>> config = load_config("my_config.yaml")
    >>>
    >>> # Load from default location or use defaults
    >>> config = load_config()
    >>>
    >>> # Require a config file to exist
    >>> config = load_config(use_defaults=False)
    """
    if yaml_path is not None:
        return Config.from_yaml(yaml_path)

    default_path = Config.get_default_config_path()
    if default_path is not None:
        return Config.from_yaml(default_path)

    if use_defaults:
        config = Config()
        config._apply_env_overrides()
        return config

    raise FileNotFoundError(
        "No configuration file found. Searched:\n"
        "  - ./config.yaml\n"
        "  - ~/.lucs_af/config.yaml\n"
        "  - <script_dir>/config.yaml"
    )
