"""Unit tests for degree_of_reshaping.common.structural_alignment module."""

import pytest
import numpy as np

from degree_of_reshaping.common.structural_alignment import (
    apply_transform,
    kabsch,
    rmsd_,
)


class TestApplyTransform:
    """Test apply_transform function."""

    def test_identity_transformation(self):
        """Test identity transformation leaves coords unchanged."""
        # Create test coordinates
        A = np.random.randn(1, 10, 3)
        # Identity rotation
        R = np.eye(3)[np.newaxis, :, :]
        # Zero translation
        t = np.zeros((1, 1, 3))

        A_transformed = apply_transform(A, R, t)

        assert np.allclose(A, A_transformed, atol=1e-6)

    def test_translation_only(self):
        """Test pure translation without rotation."""
        A = np.random.randn(1, 10, 3)
        R = np.eye(3)[np.newaxis, :, :]
        t = np.array([[[1.0, 2.0, 3.0]]])

        A_transformed = apply_transform(A, R, t)

        # Check that translation was applied
        expected = A + t
        assert np.allclose(A_transformed, expected, atol=1e-6)

    def test_rotation_only(self):
        """Test pure rotation without translation."""
        A = np.random.randn(1, 10, 3)
        # 90 degree rotation around z-axis
        angle = np.pi / 2
        R = np.array([[[np.cos(angle), -np.sin(angle), 0],
                       [np.sin(angle), np.cos(angle), 0],
                       [0, 0, 1]]], dtype=np.float32)
        t = np.zeros((1, 1, 3))

        A_transformed = apply_transform(A, R, t)

        # Verify rotation was applied
        assert A_transformed.shape == A.shape
        # After 90° rotation around z, x→-y and y→x (due to transpose in implementation)
        assert np.allclose(A_transformed[0, :, 0], -A[0, :, 1], atol=1e-5)
        assert np.allclose(A_transformed[0, :, 1], A[0, :, 0], atol=1e-5)


class TestKabsch:
    """Test Kabsch algorithm."""

    def test_identical_structures(self):
        """Test identical structures have RMSD=0."""
        A = np.random.randn(1, 10, 3).astype(np.float64)
        B = A.copy()

        A_aligned, R, t = kabsch(A, B)

        # RMSD should be zero
        rmsd = np.mean(np.sqrt(np.sum((A_aligned - B)**2, -1)))
        assert rmsd < 1e-10

        # Rotation should be identity
        assert np.allclose(R, np.eye(3, dtype=np.float64)[np.newaxis, :, :], atol=1e-6)

    def test_translated_structure(self):
        """Test alignment of translated structure."""
        A = np.random.randn(1, 10, 3).astype(np.float64)
        # Translate B
        translation = np.array([[[5.0, 3.0, -2.0]]], dtype=np.float64)
        B = A + translation

        A_aligned, R, t = kabsch(A, B)

        # Should align perfectly
        rmsd = np.mean(np.sqrt(np.sum((A_aligned - B)**2, -1)))
        assert rmsd < 1e-10

        # Rotation should be identity (pure translation)
        assert np.allclose(R, np.eye(3, dtype=np.float64)[np.newaxis, :, :], atol=1e-6)

    def test_rotated_structure(self):
        """Test alignment of rotated structure."""
        A = np.random.randn(1, 10, 3).astype(np.float64)

        # Apply known rotation to create B
        angle = np.pi / 4  # 45 degrees
        R_true = np.array([[[np.cos(angle), -np.sin(angle), 0],
                            [np.sin(angle), np.cos(angle), 0],
                            [0, 0, 1]]], dtype=np.float64)
        B = np.matmul(R_true, A.transpose(0, 2, 1)).transpose(0, 2, 1)

        A_aligned, R, t = kabsch(A, B)

        # Should align perfectly
        rmsd = np.mean(np.sqrt(np.sum((A_aligned - B)**2, -1)))
        assert rmsd < 1e-10

    def test_batch_processing(self):
        """Test Kabsch works with batch size > 1."""
        batch_size = 5
        A = np.random.randn(batch_size, 10, 3).astype(np.float64)
        B = np.random.randn(batch_size, 10, 3).astype(np.float64)

        A_aligned, R, t = kabsch(A, B)

        # Check shapes
        assert A_aligned.shape == (batch_size, 10, 3)
        assert R.shape == (batch_size, 3, 3)
        assert t.shape == (batch_size, 1, 3)


class TestRMSD:
    """Test RMSD calculation."""

    def test_rmsd_identical(self):
        """Test RMSD of identical structures."""
        pred = np.random.randn(1, 20, 3).astype(np.float64)
        true = pred.copy()

        pred_aligned, rmsd, R, t = rmsd_(pred, true)

        assert rmsd < 1e-10

    def test_rmsd_known_distance(self):
        """Test RMSD with known displacement."""
        # Create structures with known RMSD
        pred = np.zeros((1, 10, 3), dtype=np.float64)
        # Displace all points by 1.0 Å in x-direction
        true = np.zeros((1, 10, 3), dtype=np.float64)
        true[:, :, 0] = 1.0

        pred_aligned, rmsd, R, t = rmsd_(pred, true)

        # After alignment, RMSD should be close to 0
        # (since it's a pure translation)
        assert rmsd < 1e-6

    def test_rmsd_output_types(self):
        """Test RMSD returns correct types."""
        pred = np.random.randn(1, 15, 3).astype(np.float64)
        true = np.random.randn(1, 15, 3).astype(np.float64)

        pred_aligned, rmsd, R, t = rmsd_(pred, true)

        assert isinstance(pred_aligned, np.ndarray)
        assert isinstance(rmsd, (float, np.floating))
        assert isinstance(R, np.ndarray)
        assert isinstance(t, np.ndarray)

        assert pred_aligned.shape == (1, 15, 3)
        assert R.shape == (1, 3, 3)
        assert t.shape == (1, 1, 3)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
