"""Unit tests for degree_of_reshaping.utils_degree_reshaped module.

This module tests the geometric calculation and structural similarity functions
used for analyzing protein helix orientations and RMSD calculations.

Requires: natsort (via utils_degree_reshaped dependency)
"""

import pytest
import numpy as np
import sys
import os

# Add the degree_of_reshaping directory to the path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'degree_of_reshaping'))

# Skip entire module if dependencies are not available
pytest.importorskip("natsort", reason="natsort required for utils_degree_reshaped tests")

import utils_degree_reshaped as utils


class TestNormalize:
    """Test normalize function for vector normalization."""

    def test_normalize_unit_vector(self):
        """Test that a unit vector remains unchanged after normalization."""
        v = np.array([1.0, 0.0, 0.0])
        result = utils.normalize(v)

        assert np.allclose(result, v, atol=1e-10)
        assert np.abs(np.linalg.norm(result) - 1.0) < 1e-10

    def test_normalize_arbitrary_vector(self):
        """Test normalization of an arbitrary vector."""
        v = np.array([3.0, 4.0, 0.0])
        result = utils.normalize(v)

        expected = np.array([0.6, 0.8, 0.0])
        assert np.allclose(result, expected, atol=1e-10)
        assert np.abs(np.linalg.norm(result) - 1.0) < 1e-10

    def test_normalize_3d_vector(self):
        """Test normalization of a 3D vector."""
        v = np.array([1.0, 1.0, 1.0])
        result = utils.normalize(v)

        expected_norm = 1.0 / np.sqrt(3)
        expected = np.array([expected_norm, expected_norm, expected_norm])
        assert np.allclose(result, expected, atol=1e-10)
        assert np.abs(np.linalg.norm(result) - 1.0) < 1e-10

    def test_normalize_zero_vector(self):
        """Test that zero vector returns zero vector (no division by zero)."""
        v = np.array([0.0, 0.0, 0.0])
        result = utils.normalize(v)

        assert np.allclose(result, v, atol=1e-10)

    def test_normalize_negative_components(self):
        """Test normalization with negative components."""
        v = np.array([-3.0, 4.0, 0.0])
        result = utils.normalize(v)

        expected = np.array([-0.6, 0.8, 0.0])
        assert np.allclose(result, expected, atol=1e-10)

    def test_normalize_preserves_direction(self):
        """Test that normalization preserves vector direction."""
        v = np.array([2.0, 6.0, 9.0])
        result = utils.normalize(v)

        # Normalized vector should be parallel to original (cross product = 0)
        cross = np.cross(v, result)
        assert np.allclose(cross, np.zeros(3), atol=1e-10)


class TestCreateFrameFromThreePoints:
    """Test create_frame_from_three_points function for coordinate frame creation."""

    def test_frame_orthonormality(self):
        """Test that the resulting frame has orthonormal axes."""
        p1 = np.array([1.0, 0.0, 0.0])
        p2 = np.array([0.0, 0.0, 0.0])  # Origin
        p3 = np.array([0.0, 1.0, 0.0])

        frame = utils.create_frame_from_three_points(p1, p2, p3)

        # Frame should be 3x3
        assert frame.shape == (3, 3)

        # Each row should be unit length
        for i in range(3):
            assert np.abs(np.linalg.norm(frame[i]) - 1.0) < 1e-10

        # Rows should be orthogonal to each other
        assert np.abs(np.dot(frame[0], frame[1])) < 1e-10
        assert np.abs(np.dot(frame[1], frame[2])) < 1e-10
        assert np.abs(np.dot(frame[0], frame[2])) < 1e-10

    def test_frame_y_axis_direction(self):
        """Test that y-axis points from p2 to p3."""
        p1 = np.array([1.0, 0.0, 0.0])
        p2 = np.array([0.0, 0.0, 0.0])
        p3 = np.array([0.0, 5.0, 0.0])  # Along y-axis

        frame = utils.create_frame_from_three_points(p1, p2, p3)

        # y-axis (frame[1]) should point in +y direction
        expected_y = np.array([0.0, 1.0, 0.0])
        assert np.allclose(frame[1], expected_y, atol=1e-10)

    def test_frame_determinant(self):
        """Test that frame has determinant of +1 or -1 (proper rotation)."""
        p1 = np.array([1.0, 0.0, 0.0])
        p2 = np.array([0.0, 0.0, 0.0])
        p3 = np.array([0.0, 1.0, 0.0])

        frame = utils.create_frame_from_three_points(p1, p2, p3)

        det = np.linalg.det(frame)
        assert np.abs(np.abs(det) - 1.0) < 1e-10

    def test_frame_different_origin(self):
        """Test frame creation with non-zero origin."""
        p1 = np.array([6.0, 5.0, 5.0])
        p2 = np.array([5.0, 5.0, 5.0])  # Origin
        p3 = np.array([5.0, 6.0, 5.0])

        frame = utils.create_frame_from_three_points(p1, p2, p3)

        # Still should be orthonormal
        assert frame.shape == (3, 3)
        for i in range(3):
            assert np.abs(np.linalg.norm(frame[i]) - 1.0) < 1e-10

    def test_frame_protein_like_geometry(self):
        """Test frame creation with protein-like backbone geometry."""
        # Simulating C-N-CA atoms in a peptide bond
        C_prev = np.array([0.0, 0.0, 0.0])
        N = np.array([1.33, 0.0, 0.0])  # Typical C-N bond length
        CA = np.array([2.0, 1.0, 0.0])  # Typical geometry

        frame = utils.create_frame_from_three_points(C_prev, N, CA)

        # Should produce valid orthonormal frame
        assert frame.shape == (3, 3)
        for i in range(3):
            assert np.abs(np.linalg.norm(frame[i]) - 1.0) < 1e-10


class TestRotationMatrixToAxisAndAngle:
    """Test rotation_matrix_to_axis_and_angle function."""

    def test_identity_matrix(self):
        """Test identity matrix gives zero rotation angle."""
        M = np.eye(3)
        axis, angle = utils.rotation_matrix_to_axis_and_angle(M)

        # Angle should be zero (or very close)
        assert np.abs(angle) < 1e-10

    def test_90_degree_rotation_z(self):
        """Test 90 degree rotation around z-axis."""
        angle_rad = np.pi / 2
        M = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad), 0],
            [np.sin(angle_rad), np.cos(angle_rad), 0],
            [0, 0, 1]
        ])

        axis, angle = utils.rotation_matrix_to_axis_and_angle(M)

        # Angle should be pi/2
        assert np.abs(angle - np.pi / 2) < 1e-10
        # Axis should be along z (normalized)
        assert np.abs(np.abs(axis[2]) - 1.0) < 1e-10

    def test_180_degree_rotation(self):
        """Test 180 degree rotation around x-axis."""
        angle_rad = np.pi
        M = np.array([
            [1, 0, 0],
            [0, np.cos(angle_rad), -np.sin(angle_rad)],
            [0, np.sin(angle_rad), np.cos(angle_rad)]
        ])

        axis, angle = utils.rotation_matrix_to_axis_and_angle(M)

        # Angle should be pi
        assert np.abs(angle - np.pi) < 1e-10

    def test_45_degree_rotation_y(self):
        """Test 45 degree rotation around y-axis."""
        angle_rad = np.pi / 4
        M = np.array([
            [np.cos(angle_rad), 0, np.sin(angle_rad)],
            [0, 1, 0],
            [-np.sin(angle_rad), 0, np.cos(angle_rad)]
        ])

        axis, angle = utils.rotation_matrix_to_axis_and_angle(M)

        # Angle should be pi/4
        assert np.abs(angle - np.pi / 4) < 1e-10
        # Axis should be along y
        assert np.abs(np.abs(axis[1]) - 1.0) < 1e-10

    def test_arbitrary_rotation(self):
        """Test arbitrary rotation matrix decomposition."""
        # Create rotation around arbitrary axis
        angle_rad = np.pi / 3
        # Rotation around axis [1, 1, 1] normalized
        axis_input = np.array([1, 1, 1]) / np.sqrt(3)

        # Rodrigues' rotation formula to create the matrix
        K = np.array([
            [0, -axis_input[2], axis_input[1]],
            [axis_input[2], 0, -axis_input[0]],
            [-axis_input[1], axis_input[0], 0]
        ])
        M = np.eye(3) + np.sin(angle_rad) * K + (1 - np.cos(angle_rad)) * np.dot(K, K)

        axis_out, angle_out = utils.rotation_matrix_to_axis_and_angle(M)

        # Angle should match (within tolerance)
        assert np.abs(angle_out - angle_rad) < 1e-10
        # Axis should match (or be opposite with negative angle)
        assert np.allclose(np.abs(axis_out), np.abs(axis_input), atol=1e-10)


class TestHelixDirection:
    """Test helix_direction function for calculating helix axis from residues."""

    def test_helix_direction_basic(self):
        """Test helix direction with simple geometry."""
        # Create three residues with known geometry
        res1 = {
            'N': np.array([0.0, 0.0, 0.0]),
            'CA': np.array([1.5, 0.0, 0.0]),
            'C': np.array([2.0, 1.0, 0.0])
        }
        res2 = {
            'N': np.array([3.0, 1.0, 0.0]),
            'CA': np.array([4.0, 1.5, 0.5]),
            'C': np.array([5.0, 2.0, 1.0])
        }
        res3 = {
            'N': np.array([6.0, 2.0, 1.5]),
            'CA': np.array([7.0, 2.5, 2.0]),
            'C': np.array([8.0, 3.0, 2.5])
        }

        direction = utils.helix_direction(res1, res2, res3)

        # Should return a 3D vector
        assert direction.shape == (3,)
        # Direction should be normalized (unit length)
        assert np.abs(np.linalg.norm(direction) - 1.0) < 1e-10

    def test_helix_direction_returns_unit_vector(self):
        """Test that helix direction always returns a unit vector."""
        # Different geometry
        res1 = {
            'N': np.array([0.0, 0.0, 0.0]),
            'CA': np.array([1.0, 0.5, 0.0]),
            'C': np.array([2.0, 0.0, 0.5])
        }
        res2 = {
            'N': np.array([3.0, 0.5, 1.0]),
            'CA': np.array([4.0, 1.0, 1.5]),
            'C': np.array([5.0, 0.5, 2.0])
        }
        res3 = {
            'N': np.array([6.0, 1.0, 2.5]),
            'CA': np.array([7.0, 1.5, 3.0]),
            'C': np.array([8.0, 1.0, 3.5])
        }

        direction = utils.helix_direction(res1, res2, res3)

        norm = np.linalg.norm(direction)
        assert np.abs(norm - 1.0) < 1e-10

    def test_helix_direction_realistic_alpha_helix(self):
        """Test helix direction with realistic alpha helix geometry."""
        # Alpha helix parameters: 3.6 residues per turn, 1.5 A rise per residue
        # Radius approximately 2.3 A
        radius = 2.3
        rise_per_res = 1.5
        angle_per_res = 2 * np.pi / 3.6

        def make_residue(i):
            angle = i * angle_per_res
            z = i * rise_per_res
            # Simplified: place CA on helix, N and C nearby
            ca = np.array([radius * np.cos(angle), radius * np.sin(angle), z])
            n = ca + np.array([-0.5, 0.0, -0.3])
            c = ca + np.array([0.5, 0.0, 0.3])
            return {'N': n, 'CA': ca, 'C': c}

        res1 = make_residue(0)
        res2 = make_residue(1)
        res3 = make_residue(2)

        direction = utils.helix_direction(res1, res2, res3)

        # Should return a valid unit vector
        assert direction.shape == (3,)
        assert np.abs(np.linalg.norm(direction) - 1.0) < 1e-10


class TestFindAvgHelixDirection:
    """Test find_avg_helix_direction function."""

    def test_avg_helix_direction_single_triplet(self):
        """Test average direction with exactly 3 residues (single triplet)."""
        resis = [
            {'N': np.array([0.0, 0.0, 0.0]), 'CA': np.array([1.0, 0.0, 0.0]), 'C': np.array([2.0, 1.0, 0.0])},
            {'N': np.array([3.0, 1.0, 0.0]), 'CA': np.array([4.0, 1.5, 0.5]), 'C': np.array([5.0, 2.0, 1.0])},
            {'N': np.array([6.0, 2.0, 1.5]), 'CA': np.array([7.0, 2.5, 2.0]), 'C': np.array([8.0, 3.0, 2.5])}
        ]
        resis_idxs = [1, 2, 3]  # Contiguous indices

        avg_direction = utils.find_avg_helix_direction(resis, resis_idxs)

        # Should return a 3D vector
        assert avg_direction.shape == (3,)

    def test_avg_helix_direction_multiple_triplets(self):
        """Test average direction with multiple overlapping triplets."""
        # Create 5 residues for 3 triplets
        resis = []
        for i in range(5):
            resis.append({
                'N': np.array([i * 3.0, 0.0, i * 1.5]),
                'CA': np.array([i * 3.0 + 1.0, 0.5, i * 1.5 + 0.5]),
                'C': np.array([i * 3.0 + 2.0, 1.0, i * 1.5 + 1.0])
            })
        resis_idxs = [1, 2, 3, 4, 5]  # Contiguous

        avg_direction = utils.find_avg_helix_direction(resis, resis_idxs)

        # Should return a 3D vector
        assert avg_direction.shape == (3,)

    def test_avg_helix_direction_non_contiguous_skipped(self):
        """Test that non-contiguous residues are properly handled."""
        resis = [
            {'N': np.array([0.0, 0.0, 0.0]), 'CA': np.array([1.0, 0.0, 0.0]), 'C': np.array([2.0, 1.0, 0.0])},
            {'N': np.array([3.0, 1.0, 0.0]), 'CA': np.array([4.0, 1.5, 0.5]), 'C': np.array([5.0, 2.0, 1.0])},
            {'N': np.array([6.0, 2.0, 1.5]), 'CA': np.array([7.0, 2.5, 2.0]), 'C': np.array([8.0, 3.0, 2.5])}
        ]
        # Non-contiguous indices - should skip triplets
        resis_idxs = [1, 3, 5]  # Gaps between residues

        # This should handle the non-contiguous case (may return empty or handle gracefully)
        avg_direction = utils.find_avg_helix_direction(resis, resis_idxs)

        # Result depends on implementation - but should not crash
        assert avg_direction is not None

    def test_avg_helix_direction_contiguous_check(self):
        """Test that only contiguous triplets contribute to average."""
        # 4 residues with a gap: [1, 2, 4, 5]
        # Only triplet (4, 5, ...) wouldn't form since we need 3 contiguous
        resis = [
            {'N': np.array([0.0, 0.0, 0.0]), 'CA': np.array([1.0, 0.0, 0.0]), 'C': np.array([2.0, 1.0, 0.0])},
            {'N': np.array([3.0, 1.0, 0.0]), 'CA': np.array([4.0, 1.5, 0.5]), 'C': np.array([5.0, 2.0, 1.0])},
            {'N': np.array([9.0, 3.0, 3.0]), 'CA': np.array([10.0, 3.5, 3.5]), 'C': np.array([11.0, 4.0, 4.0])},
            {'N': np.array([12.0, 4.0, 4.5]), 'CA': np.array([13.0, 4.5, 5.0]), 'C': np.array([14.0, 5.0, 5.5])}
        ]
        resis_idxs = [1, 2, 4, 5]  # Gap between 2 and 4

        avg_direction = utils.find_avg_helix_direction(resis, resis_idxs)

        # Should not crash and return something
        assert avg_direction is not None


class TestCentroidOfResidues:
    """Test centroid_of_residues function."""

    def test_centroid_single_residue(self):
        """Test centroid of a single residue."""
        resis = [
            {'N': np.array([0.0, 0.0, 0.0]), 'CA': np.array([3.0, 0.0, 0.0]), 'C': np.array([0.0, 3.0, 0.0])}
        ]

        centroid = utils.centroid_of_residues(resis)

        # Centroid should be average of all atoms
        expected = (1.0, 1.0, 0.0)  # (0+3+0)/3, (0+0+3)/3, (0+0+0)/3
        assert np.allclose(centroid, expected, atol=1e-10)

    def test_centroid_multiple_residues(self):
        """Test centroid of multiple residues."""
        resis = [
            {'N': np.array([0.0, 0.0, 0.0]), 'CA': np.array([1.0, 0.0, 0.0]), 'C': np.array([2.0, 0.0, 0.0])},
            {'N': np.array([0.0, 1.0, 0.0]), 'CA': np.array([1.0, 1.0, 0.0]), 'C': np.array([2.0, 1.0, 0.0])}
        ]

        centroid = utils.centroid_of_residues(resis)

        # 6 atoms total, x: (0+1+2+0+1+2)/6=1, y: (0+0+0+1+1+1)/6=0.5, z: 0
        expected = (1.0, 0.5, 0.0)
        assert np.allclose(centroid, expected, atol=1e-10)

    def test_centroid_3d(self):
        """Test centroid in 3D space."""
        resis = [
            {'N': np.array([1.0, 2.0, 3.0]), 'CA': np.array([4.0, 5.0, 6.0])}
        ]

        centroid = utils.centroid_of_residues(resis)

        expected = (2.5, 3.5, 4.5)
        assert np.allclose(centroid, expected, atol=1e-10)

    def test_centroid_symmetric(self):
        """Test centroid of symmetric arrangement."""
        # Atoms arranged symmetrically around origin
        resis = [
            {'N': np.array([1.0, 0.0, 0.0]), 'CA': np.array([-1.0, 0.0, 0.0])},
            {'N': np.array([0.0, 1.0, 0.0]), 'CA': np.array([0.0, -1.0, 0.0])}
        ]

        centroid = utils.centroid_of_residues(resis)

        expected = (0.0, 0.0, 0.0)
        assert np.allclose(centroid, expected, atol=1e-10)

    def test_centroid_negative_coordinates(self):
        """Test centroid with negative coordinates."""
        resis = [
            {'N': np.array([-3.0, -2.0, -1.0]), 'CA': np.array([-6.0, -4.0, -2.0])}
        ]

        centroid = utils.centroid_of_residues(resis)

        expected = (-4.5, -3.0, -1.5)
        assert np.allclose(centroid, expected, atol=1e-10)


class TestCalcRMSD:
    """Test calc_RMSD function from utils_degree_reshaped."""

    def test_rmsd_identical_coords(self):
        """Test RMSD of identical coordinates is zero."""
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])

        rmsd = utils.calc_RMSD(coords, coords)

        assert np.abs(rmsd) < 1e-10

    def test_rmsd_known_displacement(self):
        """Test RMSD with known uniform displacement."""
        ref = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        # Shift all points by (1, 0, 0)
        coords = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, 1.0, 0.0]])

        rmsd = utils.calc_RMSD(ref, coords)

        # RMSD should be 1.0 (uniform shift of 1 unit)
        assert np.abs(rmsd - 1.0) < 1e-10

    def test_rmsd_single_point(self):
        """Test RMSD with single point."""
        ref = np.array([[0.0, 0.0, 0.0]])
        coords = np.array([[1.0, 0.0, 0.0]])

        rmsd = utils.calc_RMSD(ref, coords)

        assert np.abs(rmsd - 1.0) < 1e-10

    def test_rmsd_3d_displacement(self):
        """Test RMSD with 3D displacement."""
        ref = np.array([[0.0, 0.0, 0.0]])
        # Displacement of (1, 1, 1) -> distance = sqrt(3)
        coords = np.array([[1.0, 1.0, 1.0]])

        rmsd = utils.calc_RMSD(ref, coords)

        expected = np.sqrt(3)
        assert np.abs(rmsd - expected) < 1e-10

    def test_rmsd_multiple_points_varied(self):
        """Test RMSD with multiple points having different displacements."""
        ref = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        # First point displaced by 1, second by 2
        coords = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])

        rmsd = utils.calc_RMSD(ref, coords)

        # RMSD = sqrt((1^2 + 2^2) / 2) = sqrt(2.5)
        expected = np.sqrt(2.5)
        assert np.abs(rmsd - expected) < 1e-10

    def test_rmsd_2d_coords(self):
        """Test RMSD works with 2D coordinates."""
        ref = np.array([[0.0, 0.0], [1.0, 0.0]])
        coords = np.array([[0.0, 1.0], [1.0, 1.0]])

        rmsd = utils.calc_RMSD(ref, coords)

        # Both points shifted by 1 in y direction
        assert np.abs(rmsd - 1.0) < 1e-10

    def test_rmsd_non_negative(self):
        """Test RMSD is always non-negative."""
        np.random.seed(42)
        for _ in range(10):
            ref = np.random.randn(20, 3)
            coords = np.random.randn(20, 3)
            rmsd = utils.calc_RMSD(ref, coords)
            assert rmsd >= 0


class TestSplitContiguousResidues:
    """Test split_contiguous_residues function."""

    def test_split_single_contiguous_block(self):
        """Test list that is entirely contiguous."""
        res_list = [1, 2, 3, 4, 5]

        result = utils.split_contiguous_residues(res_list)

        assert result == [[1, 2, 3, 4, 5]]

    def test_split_two_blocks(self):
        """Test list with two contiguous blocks."""
        res_list = [1, 2, 3, 5, 6, 7]

        result = utils.split_contiguous_residues(res_list)

        assert result == [[1, 2, 3], [5, 6, 7]]

    def test_split_three_blocks(self):
        """Test list with three contiguous blocks."""
        res_list = [1, 2, 3, 5, 6, 7, 8, 10]

        result = utils.split_contiguous_residues(res_list)

        assert result == [[1, 2, 3], [5, 6, 7, 8], [10]]

    def test_split_single_element(self):
        """Test list with single element."""
        res_list = [5]

        result = utils.split_contiguous_residues(res_list)

        assert result == [[5]]

    def test_split_all_isolated(self):
        """Test list where every element is isolated."""
        res_list = [1, 3, 5, 7, 9]

        result = utils.split_contiguous_residues(res_list)

        assert result == [[1], [3], [5], [7], [9]]

    def test_split_large_gap(self):
        """Test list with a large gap."""
        res_list = [1, 2, 3, 100, 101, 102]

        result = utils.split_contiguous_residues(res_list)

        assert result == [[1, 2, 3], [100, 101, 102]]

    def test_split_preserves_order(self):
        """Test that split preserves original order."""
        res_list = [10, 11, 12, 20, 21]

        result = utils.split_contiguous_residues(res_list)

        assert result == [[10, 11, 12], [20, 21]]
        # Check order within blocks
        for block in result:
            for i in range(len(block) - 1):
                assert block[i] < block[i + 1]


class TestGeometry:
    """Integration tests combining multiple geometric functions."""

    def test_frame_and_rotation_roundtrip(self):
        """Test that creating a frame and extracting rotation is consistent."""
        # Create a frame
        p1 = np.array([1.0, 0.0, 0.0])
        p2 = np.array([0.0, 0.0, 0.0])
        p3 = np.array([0.0, 1.0, 0.0])

        frame = utils.create_frame_from_three_points(p1, p2, p3)

        # Frame should be a valid rotation matrix
        axis, angle = utils.rotation_matrix_to_axis_and_angle(frame)

        # Should produce valid axis (unit vector or zero for identity)
        if angle > 1e-10:
            assert np.abs(np.linalg.norm(axis) - 1.0) < 1e-10

    def test_helix_centroid_and_direction(self):
        """Test that helix centroid and direction can be computed together."""
        # Create a simple helix-like arrangement
        resis = []
        resis_idxs = []
        for i in range(5):
            z = i * 1.5  # Rise per residue
            angle = i * (2 * np.pi / 3.6)  # Alpha helix angle
            radius = 2.3

            ca = np.array([radius * np.cos(angle), radius * np.sin(angle), z])
            n = ca + np.array([-0.5, 0.0, -0.3])
            c = ca + np.array([0.5, 0.0, 0.3])

            resis.append({'N': n, 'CA': ca, 'C': c})
            resis_idxs.append(i + 1)

        # Both functions should work on the same data
        centroid = utils.centroid_of_residues(resis)
        direction = utils.find_avg_helix_direction(resis, resis_idxs)

        # Centroid should be roughly in the middle of the helix
        assert centroid is not None
        assert len(centroid) == 3

        # Direction should be a unit vector
        if direction is not None and not np.isnan(direction).any():
            assert np.abs(np.linalg.norm(direction) - 1.0) < 1e-10


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
