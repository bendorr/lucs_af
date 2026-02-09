# Tests for LUCS AlphaFold Analysis

This directory contains the test suite for the LUCS AlphaFold analysis tools.

## Running Tests

### Run all tests
```bash
pytest
```

### Run specific test file
```bash
pytest tests/test_structural_alignment.py
```

### Run tests with specific markers
```bash
# Run only unit tests (fast, no external dependencies)
pytest -m unit

# Run integration tests (may require PyRosetta)
pytest -m integration

# Skip slow tests
pytest -m "not slow"

# Skip tests requiring PyRosetta
pytest -m "not requires_pyrosetta"
```

### Run with coverage
```bash
pytest --cov=common --cov-report=html
# Open htmlcov/index.html to view coverage report
```

### Verbose output
```bash
pytest -v
```

## Test Organization

### Test Files
- `test_structural_alignment.py` - Tests for Kabsch alignment and coordinate transformations
- `test_rmsd_calculations.py` - Tests for RMSD calculation functions
- `test_config.py` - Tests for configuration management system

### Test Markers
Tests are categorized using pytest markers:

- `@pytest.mark.unit` - Fast unit tests with no external dependencies
- `@pytest.mark.integration` - Integration tests that may use PyRosetta
- `@pytest.mark.slow` - Long-running tests
- `@pytest.mark.requires_pyrosetta` - Tests requiring PyRosetta installation
- `@pytest.mark.requires_gpu` - Tests requiring GPU/CUDA

## Writing Tests

### Example test structure
```python
import pytest
from degree_of_reshaping.common.my_module import my_function

class TestMyFunction:
    """Tests for my_function."""

    def test_basic_functionality(self):
        """Test basic use case."""
        result = my_function(input_data)
        assert result == expected_output

    @pytest.mark.slow
    def test_large_dataset(self):
        """Test with large dataset (slow)."""
        # Long-running test
        pass

    @pytest.mark.requires_pyrosetta
    def test_with_pyrosetta(self):
        """Test that requires PyRosetta."""
        import pyrosetta
        # PyRosetta-specific test
        pass
```

### Test Guidelines
1. **Name tests descriptively** - Use `test_<what_is_being_tested>` pattern
2. **One assertion per test** - Or related assertions for the same behavior
3. **Use fixtures for setup** - Avoid code duplication
4. **Mark dependencies** - Use markers for tests requiring external dependencies
5. **Test edge cases** - Include tests for boundary conditions and error cases

## Test Coverage Goals

- **Common modules**: ~75% coverage
  - `degree_of_reshaping/common/structural_alignment.py`
  - `degree_of_reshaping/common/rmsd_calculations.py`
  - `degree_of_reshaping/common/pyrosetta_utils.py`
  - `degree_of_reshaping/config.py`

- **Critical functions**: 100% coverage
  - Kabsch algorithm
  - RMSD calculations
  - Configuration management

## Continuous Integration

Tests should be run:
- Before committing changes
- In CI/CD pipeline
- Before releasing new versions

## Dependencies

### Required for all tests
- pytest
- numpy
- torch

### Required for specific tests
- PyRosetta (for `@pytest.mark.requires_pyrosetta` tests)
- BioPandas (for PDB file tests)
- CUDA (for `@pytest.mark.requires_gpu` tests)

### Installing test dependencies
```bash
pip install pytest pytest-cov
```

## Troubleshooting

### Import errors
Make sure you're running pytest from the repository root:
```bash
cd /path/to/lucs_af
pytest
```

### PyRosetta tests failing
If PyRosetta is not installed, these tests will be skipped automatically.
To install PyRosetta, follow instructions at: https://www.pyrosetta.org/

### Coverage not working
Install pytest-cov:
```bash
pip install pytest-cov
```

---

### `tests/`

#### > `test_config.py`
Unit tests for configuration management, including .yml loading, environment variables, arg parsing, and default values.

#### > `test_rmsd_calculations.py`
Tests for RMSD calculation functions across different implementations (Biopython, PyRosetta, aligning residue subsets, etc.).

#### > `test_structural_alignment.py`
Tests for structural alignment algorithms.
