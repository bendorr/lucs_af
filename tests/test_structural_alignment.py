"""
Unit tests for degree_of_reshaping.common.structural_alignment module.

Author: Ben Orr
Date: 12.18.25
"""

import pytest
import numpy as np
import torch

from degree_of_reshaping.common.structural_alignment import (
    apply_transform,
    kabsch,
    rmsd_,
)


class TestApplyTransform:
    """Tests for apply_transform function."""

    def test_identity_transformation(self):
        """Test that identity transformation leaves coordinates unchanged."""
        # Create test coordinates
        A = torch.randn(1, 10, 3)
        # Identity rotation
        R = torch.eye(3).unsqueeze(0)
        # Zero translation
        t = torch.zeros(1, 1, 3)

        A_transformed = apply_transform(A, R, t)

        assert torch.allclose(A, A_transformed, atol=1e-6)

    def test_translation_only(self):
        """Test pure translation without rotation."""
        A = torch.randn(1, 10, 3)
        R = torch.eye(3).unsqueeze(0)
        t = torch.tensor([[[1.0, 2.0, 3.0]]])

        A_transformed = apply_transform(A, R, t)

        # Check that translation was applied
        expected = A + t
        assert torch.allclose(A_transformed, expected, atol=1e-6)

    def test_rotation_only(self):
        """Test pure rotation without translation."""
        A = torch.randn(1, 10, 3)
        # 90 degree rotation around z-axis
        angle = np.pi / 2
        R = torch.tensor([[[np.cos(angle), -np.sin(angle), 0],
                           [np.sin(angle), np.cos(angle), 0],
                           [0, 0, 1]]], dtype=torch.float32)
        t = torch.zeros(1, 1, 3)

        A_transformed = apply_transform(A, R, t)

        # Verify rotation was applied
        assert A_transformed.shape == A.shape
        # After 90° rotation around z, x→y and y→-x
        assert torch.allclose(A_transformed[0, :, 0], A[0, :, 1], atol=1e-5)
        assert torch.allclose(A_transformed[0, :, 1], -A[0, :, 0], atol=1e-5)


class TestKabsch:
    """Tests for Kabsch algorithm."""

    def test_identical_structures(self):
        """Test that identical structures have RMSD=0."""
        A = torch.randn(1, 10, 3, dtype=torch.float64)
        B = A.clone()

        A_aligned, R, t = kabsch(A, B)

        # RMSD should be zero
        rmsd = torch.mean(torch.sqrt(torch.sum((A_aligned - B).pow(2), -1)))
        assert rmsd < 1e-10

        # Rotation should be identity
        assert torch.allclose(R, torch.eye(3, dtype=torch.float64).unsqueeze(0), atol=1e-6)

    def test_translated_structure(self):
        """Test alignment of translated structure."""
        A = torch.randn(1, 10, 3, dtype=torch.float64)
        # Translate B
        translation = torch.tensor([[[5.0, 3.0, -2.0]]], dtype=torch.float64)
        B = A + translation

        A_aligned, R, t = kabsch(A, B)

        # Should align perfectly
        rmsd = torch.mean(torch.sqrt(torch.sum((A_aligned - B).pow(2), -1)))
        assert rmsd < 1e-10

        # Rotation should be identity (pure translation)
        assert torch.allclose(R, torch.eye(3, dtype=torch.float64).unsqueeze(0), atol=1e-6)

    def test_rotated_structure(self):
        """Test alignment of rotated structure."""
        A = torch.randn(1, 10, 3, dtype=torch.float64)

        # Apply known rotation to create B
        angle = np.pi / 4  # 45 degrees
        R_true = torch.tensor([[[np.cos(angle), -np.sin(angle), 0],
                                [np.sin(angle), np.cos(angle), 0],
                                [0, 0, 1]]], dtype=torch.float64)
        B = torch.bmm(R_true, A.transpose(1, 2)).transpose(1, 2)

        A_aligned, R, t = kabsch(A, B)

        # Should align perfectly
        rmsd = torch.mean(torch.sqrt(torch.sum((A_aligned - B).pow(2), -1)))
        assert rmsd < 1e-10

    def test_batch_processing(self):
        """Test that Kabsch works with batch size > 1."""
        batch_size = 5
        A = torch.randn(batch_size, 10, 3, dtype=torch.float64)
        B = torch.randn(batch_size, 10, 3, dtype=torch.float64)

        A_aligned, R, t = kabsch(A, B)

        # Check shapes
        assert A_aligned.shape == (batch_size, 10, 3)
        assert R.shape == (batch_size, 3, 3)
        assert t.shape == (batch_size, 1, 3)


class TestRMSD:
    """Tests for RMSD calculation."""

    def test_rmsd_identical(self):
        """Test RMSD of identical structures."""
        pred = torch.randn(1, 20, 3, dtype=torch.float64)
        true = pred.clone()

        pred_aligned, rmsd, R, t = rmsd_(pred, true)

        assert rmsd < 1e-10

    def test_rmsd_known_distance(self):
        """Test RMSD with known displacement."""
        # Create structures with known RMSD
        pred = torch.zeros(1, 10, 3, dtype=torch.float64)
        # Displace all points by 1.0 Å in x-direction
        true = torch.zeros(1, 10, 3, dtype=torch.float64)
        true[:, :, 0] = 1.0

        pred_aligned, rmsd, R, t = rmsd_(pred, true)

        # After alignment, RMSD should be close to 0
        # (since it's a pure translation)
        assert rmsd < 1e-6

    def test_rmsd_output_types(self):
        """Test that RMSD returns correct types."""
        pred = torch.randn(1, 15, 3, dtype=torch.float64)
        true = torch.randn(1, 15, 3, dtype=torch.float64)

        pred_aligned, rmsd, R, t = rmsd_(pred, true)

        assert isinstance(pred_aligned, torch.Tensor)
        assert isinstance(rmsd, torch.Tensor)
        assert isinstance(R, torch.Tensor)
        assert isinstance(t, torch.Tensor)

        assert pred_aligned.shape == (1, 15, 3)
        assert R.shape == (1, 3, 3)
        assert t.shape == (1, 1, 3)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
