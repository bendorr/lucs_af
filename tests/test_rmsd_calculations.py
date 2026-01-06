"""
Unit tests for degree_of_reshaping.common.rmsd_calculations module.

Author: Ben Orr
Date: 12.18.25
"""

import pytest
import numpy as np

from degree_of_reshaping.common.rmsd_calculations import calc_rmsd


class TestCalcRMSD:
    """Tests for calc_rmsd function."""

    def test_identical_coordinates(self):
        """Test RMSD of identical coordinate sets."""
        coords = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]], dtype=np.float64)
        rmsd = calc_rmsd(coords, coords)

        assert np.abs(rmsd) < 1e-10

    def test_known_displacement(self):
        """Test RMSD with known uniform displacement."""
        ref = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
        # Shift all points by (1, 0, 0)
        coords = np.array([[1, 0, 0], [2, 0, 0], [1, 1, 0]], dtype=np.float64)

        rmsd = calc_rmsd(ref, coords)

        # RMSD should be 1.0 (uniform shift of 1 Å)
        assert np.abs(rmsd - 1.0) < 1e-10

    def test_2d_coordinates(self):
        """Test RMSD calculation with 2D coordinates."""
        ref = np.array([[0, 0], [1, 1], [2, 2]], dtype=np.float64)
        coords = np.array([[0.1, 0.1], [1.1, 1.1], [2.1, 2.1]], dtype=np.float64)

        rmsd = calc_rmsd(ref, coords)

        # Expected RMSD: sqrt(mean(0.1^2 + 0.1^2)) = sqrt(0.02) ≈ 0.1414
        expected = np.sqrt(0.02)
        assert np.abs(rmsd - expected) < 1e-10

    def test_random_coordinates(self):
        """Test that RMSD is non-negative and reasonable."""
        np.random.seed(42)
        ref = np.random.randn(100, 3)
        coords = np.random.randn(100, 3)

        rmsd = calc_rmsd(ref, coords)

        # RMSD should be positive
        assert rmsd > 0
        # Should be reasonable for random data
        assert rmsd < 100  # Sanity check

    def test_single_point(self):
        """Test RMSD with single point."""
        ref = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)
        coords = np.array([[1.5, 2.5, 3.5]], dtype=np.float64)

        rmsd = calc_rmsd(ref, coords)

        # Distance is sqrt((0.5^2 + 0.5^2 + 0.5^2)) = sqrt(0.75)
        expected = np.sqrt(0.75)
        assert np.abs(rmsd - expected) < 1e-10

    def test_shape_mismatch_raises_error(self):
        """Test that mismatched shapes raise an error."""
        ref = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float64)
        coords = np.array([[0, 0, 0]], dtype=np.float64)

        with pytest.raises(IndexError):
            calc_rmsd(ref, coords)


# PyRosetta tests would require PyRosetta installation
# Mark them to be skipped if PyRosetta is not available
@pytest.mark.requires_pyrosetta
class TestPyRosettaFunctions:
    """Tests for PyRosetta-based RMSD functions."""

    def test_get_backbone_points_placeholder(self):
        """Placeholder for PyRosetta tests."""
        pytest.skip("PyRosetta tests require PyRosetta installation")

    def test_calc_backbone_rmsd_placeholder(self):
        """Placeholder for PyRosetta tests."""
        pytest.skip("PyRosetta tests require PyRosetta installation")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
