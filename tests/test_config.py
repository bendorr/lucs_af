"""Unit tests for config module."""

import pytest
import os
import tempfile
from pathlib import Path

from degree_of_reshaping.config import Config, PathConfig, AnalysisConfig, AlphaFoldConfig, load_config


class TestPathConfig:
    """Test PathConfig dataclass."""

    def test_default_values(self):
        """Test PathConfig expected defaults."""
        config = PathConfig()

        assert config.starting_structure_path == ""
        assert config.starting_structure_insertion_points == ""
        assert config.data_dir == "./data"
        assert config.output_dir == "./output"

    def test_custom_values(self):
        """Test setting custom values."""
        config = PathConfig(
            starting_structure_path="/path/to/struct.pdb",
            data_dir="/custom/data"
        )

        assert config.starting_structure_path == "/path/to/struct.pdb"
        assert config.data_dir == "/custom/data"


class TestAnalysisConfig:
    """Test AnalysisConfig dataclass."""

    def test_default_values(self):
        """Test AnalysisConfig expected defaults."""
        config = AnalysisConfig()

        assert config.filter_by_subsequence == ""
        assert config.start_struct_res_start_idx == 1
        assert config.verbose is False
        assert config.num_tasks == 1
        assert config.task_id == 1


class TestAlphaFoldConfig:
    """Test AlphaFoldConfig dataclass."""

    def test_default_values(self):
        """Test AlphaFoldConfig expected defaults."""
        config = AlphaFoldConfig()

        assert config.model_type == "af2"
        assert config.use_best_model_only is False
        assert config.confidence_threshold == 70.0


class TestConfig:
    """Test main Config class."""

    def test_default_initialization(self):
        """Test Config initializes with default sub-configs."""
        config = Config()

        assert isinstance(config.paths, PathConfig)
        assert isinstance(config.analysis, AnalysisConfig)
        assert isinstance(config.alphafold, AlphaFoldConfig)

    def test_to_dict(self):
        """Test dict conversion."""
        config = Config()
        config.paths.data_dir = "/test/data"
        config.analysis.verbose = True

        config_dict = config.to_dict()

        assert config_dict['paths']['data_dir'] == "/test/data"
        assert config_dict['analysis']['verbose'] is True

    def test_to_yaml_and_from_yaml(self):
        """Test YAML save and load."""
        config = Config()
        config.paths.starting_structure_path = "/test/struct.pdb"
        config.analysis.verbose = True
        config.alphafold.model_type = "af3"

        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_path = f.name

        try:
            config.to_yaml(temp_path)

            # Load from file
            loaded_config = Config.from_yaml(temp_path)

            assert loaded_config.paths.starting_structure_path == "/test/struct.pdb"
            assert loaded_config.analysis.verbose is True
            assert loaded_config.alphafold.model_type == "af3"
        finally:
            os.unlink(temp_path)

    def test_env_overrides(self):
        """Test env variable overrides."""
        # Set environment variables
        os.environ['LUCS_DATA_DIR'] = '/env/data'
        os.environ['LUCS_VERBOSE'] = 'true'
        os.environ['LUCS_NUM_TASKS'] = '10'

        try:
            config = Config()
            config._apply_env_overrides()

            assert config.paths.data_dir == '/env/data'
            assert config.analysis.verbose is True
            assert config.analysis.num_tasks == 10
        finally:
            # Clean up environment
            del os.environ['LUCS_DATA_DIR']
            del os.environ['LUCS_VERBOSE']
            del os.environ['LUCS_NUM_TASKS']

    def test_update_from_args(self):
        """Test update from argparse Namespace."""
        import argparse

        args = argparse.Namespace(
            data_dir='/args/data',
            verbose=True,
            num_tasks=5
        )

        config = Config()
        config.update_from_args(args)

        assert config.paths.data_dir == '/args/data'
        assert config.analysis.verbose is True
        assert config.analysis.num_tasks == 5


class TestLoadConfig:
    """Test load_config convenience function."""

    def test_load_with_defaults(self):
        """Test load with default values."""
        config = load_config(use_defaults=True)

        assert isinstance(config, Config)
        assert config.paths.data_dir == "./data"

    def test_load_nonexistent_file_raises_error(self):
        """Test that loading a nonexistent file raises FileNotFoundError.

        Note: When an explicit yaml_path is provided, load_config always
        tries to load that file directly - it does not fall back to defaults.
        """
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path.yaml", use_defaults=True)

    def test_load_from_yaml_nonexistent(self):
        """Test Config.from_yaml raises error for nonexistent file."""
        with pytest.raises(FileNotFoundError):
            Config.from_yaml("/nonexistent/config.yaml")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
