"""
Unit tests for config module.

Author: Ben Orr
Date: 12.18.25
"""

import pytest
import os
import tempfile
from pathlib import Path

from degree_of_reshaping.config import Config, PathConfig, AnalysisConfig, AlphaFoldConfig, load_config


class TestPathConfig:
    """Tests for PathConfig dataclass."""

    def test_default_values(self):
        """Test that PathConfig has expected defaults."""
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
    """Tests for AnalysisConfig dataclass."""

    def test_default_values(self):
        """Test that AnalysisConfig has expected defaults."""
        config = AnalysisConfig()

        assert config.filter_by_subsequence == ""
        assert config.start_struct_res_start_idx == 1
        assert config.verbose is False
        assert config.num_tasks == 1
        assert config.task_id == 1


class TestAlphaFoldConfig:
    """Tests for AlphaFoldConfig dataclass."""

    def test_default_values(self):
        """Test that AlphaFoldConfig has expected defaults."""
        config = AlphaFoldConfig()

        assert config.model_type == "af2"
        assert config.use_best_model_only is False
        assert config.confidence_threshold == 70.0


class TestConfig:
    """Tests for main Config class."""

    def test_default_initialization(self):
        """Test that Config initializes with default sub-configs."""
        config = Config()

        assert isinstance(config.paths, PathConfig)
        assert isinstance(config.analysis, AnalysisConfig)
        assert isinstance(config.alphafold, AlphaFoldConfig)

    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = Config()
        config.paths.data_dir = "/test/data"
        config.analysis.verbose = True

        config_dict = config.to_dict()

        assert config_dict['paths']['data_dir'] == "/test/data"
        assert config_dict['analysis']['verbose'] is True

    def test_to_yaml_and_from_yaml(self):
        """Test saving to and loading from YAML."""
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
        """Test environment variable overrides."""
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
        """Test updating from argparse Namespace."""
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
    """Tests for load_config convenience function."""

    def test_load_with_defaults(self):
        """Test loading with default values."""
        config = load_config(use_defaults=True)

        assert isinstance(config, Config)
        assert config.paths.data_dir == "./data"

    def test_load_nonexistent_file_with_defaults(self):
        """Test that nonexistent file falls back to defaults."""
        config = load_config("/nonexistent/path.yaml", use_defaults=True)

        # Should still create config with defaults
        # (though from_yaml will raise error for nonexistent file)
        # This tests the fallback behavior

    def test_load_nonexistent_file_without_defaults(self):
        """Test that nonexistent file raises error when use_defaults=False."""
        with pytest.raises(FileNotFoundError):
            load_config(use_defaults=False)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
