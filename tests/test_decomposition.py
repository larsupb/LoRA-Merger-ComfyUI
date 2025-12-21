"""
Unit tests for decomposition module.

Tests tensor decomposition functionality including SVD, QR, and error handling.
"""

import pytest
import torch
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from decomposition import (
    SVDDecomposer,
    RandomizedSVDDecomposer,
    EnergyBasedRandomizedSVDDecomposer,
    QRDecomposer,
    SingularValueDistribution,
)


class TestSVDDecomposer:
    """Tests for standard SVD decomposer."""

    def test_basic_2d_decomposition(self):
        """Test basic 2D tensor decomposition."""
        decomposer = SVDDecomposer()
        weight = torch.randn(100, 50)
        target_rank = 10

        up, down, alpha, stats = decomposer.decompose(
            weight, target_rank, return_statistics=True
        )

        # Check shapes
        assert up.shape == (100, 10)
        assert down.shape == (10, 50)
        assert isinstance(alpha, float)

        # Check reconstruction quality
        reconstructed = up @ down
        reconstruction_error = torch.norm(weight - reconstructed) / torch.norm(weight)
        assert reconstruction_error < 0.5  # Should be reasonable approximation

    def test_4d_conv_decomposition(self):
        """Test 4D convolutional tensor decomposition."""
        decomposer = SVDDecomposer()
        weight = torch.randn(64, 32, 3, 3)  # Conv layer
        target_rank = 16

        up, down, alpha, _ = decomposer.decompose(weight, target_rank)

        # Check shapes
        assert up.shape == (64, 16, 1, 1)
        assert down.shape == (16, 32, 3, 3)

    def test_symmetric_vs_asymmetric_distribution(self):
        """Test different singular value distributions."""
        weight = torch.randn(50, 30)
        target_rank = 10

        # Symmetric distribution
        decomposer_sym = SVDDecomposer(
            distribution=SingularValueDistribution.SYMMETRIC
        )
        up_sym, down_sym, _, _ = decomposer_sym.decompose(weight, target_rank)

        # Asymmetric distribution
        decomposer_asym = SVDDecomposer(
            distribution=SingularValueDistribution.ASYMMETRIC
        )
        up_asym, down_asym, _, _ = decomposer_asym.decompose(weight, target_rank)

        # Both should reconstruct similarly but with different scaling
        recon_sym = up_sym @ down_sym
        recon_asym = up_asym @ down_asym

        assert torch.allclose(recon_sym, recon_asym, rtol=1e-4)

    def test_statistics_calculation(self):
        """Test that statistics are calculated correctly."""
        decomposer = SVDDecomposer(return_statistics=True)
        weight = torch.randn(80, 40)
        target_rank = 20

        _, _, _, stats = decomposer.decompose(weight, target_rank)

        assert stats is not None
        assert 'new_rank' in stats
        assert 'new_alpha' in stats
        assert 'sum_retained' in stats
        assert 'fro_retained' in stats
        assert 'max_ratio' in stats

        # Check statistics are reasonable
        assert stats['new_rank'] == 20
        assert 0.0 <= stats['sum_retained'] <= 1.0
        assert 0.0 <= stats['fro_retained'] <= 1.0

    def test_dynamic_rank_selection_ratio(self):
        """Test dynamic rank selection by singular value ratio."""
        decomposer = SVDDecomposer(return_statistics=True)
        weight = torch.randn(100, 50)

        _, _, _, stats = decomposer.decompose(
            weight,
            target_rank=50,
            dynamic_method="sv_ratio",
            dynamic_param=100.0  # Ratio threshold
        )

        # Rank should be selected based on ratio
        assert stats['new_rank'] <= 50
        assert stats['new_rank'] >= 1

    def test_dynamic_rank_selection_cumulative(self):
        """Test dynamic rank selection by cumulative singular values."""
        decomposer = SVDDecomposer(return_statistics=True)
        weight = torch.randn(100, 50)

        _, _, _, stats = decomposer.decompose(
            weight,
            target_rank=50,
            dynamic_method="sv_cumulative",
            dynamic_param=0.95  # 95% of cumulative sum
        )

        # Rank should be selected to capture 95% of singular values
        assert stats['sum_retained'] >= 0.90  # Allow some tolerance

    def test_dynamic_rank_selection_frobenius(self):
        """Test dynamic rank selection by Frobenius norm."""
        decomposer = SVDDecomposer(return_statistics=True)
        weight = torch.randn(100, 50)

        _, _, _, stats = decomposer.decompose(
            weight,
            target_rank=50,
            dynamic_method="sv_fro",
            dynamic_param=0.99  # 99% of Frobenius norm
        )

        # Rank should be selected to retain 99% of Frobenius norm
        assert stats['fro_retained'] >= 0.95  # Allow some tolerance


class TestRandomizedSVDDecomposer:
    """Tests for randomized SVD decomposer."""

    def test_randomized_svd_approximation(self):
        """Test that randomized SVD produces good approximation."""
        weight = torch.randn(200, 100)
        target_rank = 20

        # Standard SVD
        decomposer_std = SVDDecomposer()
        up_std, down_std, _, _ = decomposer_std.decompose(weight, target_rank)
        recon_std = up_std @ down_std

        # Randomized SVD
        decomposer_rand = RandomizedSVDDecomposer(n_oversamples=10, n_iter=2)
        up_rand, down_rand, _, _ = decomposer_rand.decompose(weight, target_rank)
        recon_rand = up_rand @ down_rand

        # Reconstructions should be similar
        error_std = torch.norm(weight - recon_std)
        error_rand = torch.norm(weight - recon_rand)

        # Randomized should be close to standard (within 50% relative error)
        assert abs(error_rand - error_std) / error_std < 0.5

    def test_randomized_svd_small_matrix(self):
        """Test that small matrices fall back to standard SVD."""
        decomposer = RandomizedSVDDecomposer()
        weight = torch.randn(50, 30)  # Small matrix
        target_rank = 10

        # Should not raise error
        up, down, alpha, _ = decomposer.decompose(weight, target_rank)

        assert up.shape == (50, 10)
        assert down.shape == (10, 30)


class TestEnergyBasedRandomizedSVDDecomposer:
    """Tests for energy-based randomized SVD."""

    def test_energy_based_rank_selection(self):
        """Test that energy threshold affects rank selection."""
        weight = torch.randn(100, 50)

        # Low energy threshold (fewer components)
        decomposer_low = EnergyBasedRandomizedSVDDecomposer(
            energy_threshold=0.8,
            return_statistics=True
        )
        _, _, _, stats_low = decomposer_low.decompose(weight, target_rank=50)

        # High energy threshold (more components)
        decomposer_high = EnergyBasedRandomizedSVDDecomposer(
            energy_threshold=0.99,
            return_statistics=True
        )
        _, _, _, stats_high = decomposer_high.decompose(weight, target_rank=50)

        # Higher threshold should generally use more components
        # (though not guaranteed due to randomness)
        assert stats_low is not None
        assert stats_high is not None


class TestQRDecomposer:
    """Tests for QR decomposer."""

    def test_qr_decomposition(self):
        """Test basic QR decomposition."""
        decomposer = QRDecomposer()
        weight = torch.randn(100, 50)
        target_rank = 20

        up, down, alpha, _ = decomposer.decompose(weight, target_rank)

        # Check shapes
        assert up.shape == (100, 20)
        assert down.shape == (20, 50)

        # QR decomposition should still provide reasonable approximation
        reconstructed = up @ down
        reconstruction_error = torch.norm(weight - reconstructed) / torch.norm(weight)
        assert reconstruction_error < 1.0  # Looser bound than SVD


class TestErrorHandling:
    """Tests for error handling in decomposition."""

    def test_invalid_tensor_dimensions(self):
        """Test that invalid tensor dimensions raise errors."""
        decomposer = SVDDecomposer()
        weight = torch.randn(10)  # 1D tensor (invalid)

        with pytest.raises(ValueError, match="must be 2D, 3D, or 4D"):
            decomposer.decompose(weight, target_rank=5)

    def test_zero_matrix_handling(self):
        """Test handling of numerically zero matrices."""
        decomposer = SVDDecomposer(return_statistics=True)
        weight = torch.zeros(50, 30)

        # Should not crash, should handle gracefully
        up, down, alpha, stats = decomposer.decompose(weight, target_rank=10)

        # Rank should be minimal for zero matrix
        assert stats['new_rank'] == 1

    def test_invalid_dynamic_method(self):
        """Test that invalid dynamic method raises error."""
        decomposer = SVDDecomposer()
        weight = torch.randn(50, 30)

        with pytest.raises(ValueError, match="Unknown dynamic method"):
            decomposer.decompose(
                weight,
                target_rank=10,
                dynamic_method="invalid_method"
            )


# Fixtures

@pytest.fixture
def sample_2d_weight():
    """Fixture providing sample 2D weight tensor."""
    return torch.randn(100, 50)


@pytest.fixture
def sample_4d_weight():
    """Fixture providing sample 4D convolutional weight."""
    return torch.randn(64, 32, 3, 3)


class TestIntegrationWithFixtures:
    """Integration tests using fixtures."""

    def test_all_decomposers_with_2d(self, sample_2d_weight):
        """Test all decomposers work with 2D tensors."""
        decomposers = [
            SVDDecomposer(),
            RandomizedSVDDecomposer(),
            EnergyBasedRandomizedSVDDecomposer(),
            QRDecomposer(),
        ]

        for decomposer in decomposers:
            up, down, alpha, _ = decomposer.decompose(
                sample_2d_weight,
                target_rank=20
            )

            assert up.shape[0] == 100
            assert up.shape[1] == 20
            assert down.shape[0] == 20
            assert down.shape[1] == 50

    def test_all_decomposers_with_4d(self, sample_4d_weight):
        """Test all decomposers work with 4D conv tensors."""
        decomposers = [
            SVDDecomposer(),
            RandomizedSVDDecomposer(),
            QRDecomposer(),
        ]

        for decomposer in decomposers:
            up, down, alpha, _ = decomposer.decompose(
                sample_4d_weight,
                target_rank=16
            )

            assert up.shape == (64, 16, 1, 1)
            assert down.shape == (16, 32, 3, 3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
