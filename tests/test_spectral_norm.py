"""
Tests for spectral norm regularization.

This module tests the spectral norm computation and per-layer clipping functionality.
"""

import torch
import pytest
import sys
import os

# Add parent directory to path for direct imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import directly from spectral_norm module to avoid ComfyUI dependencies
import importlib.util
spec = importlib.util.spec_from_file_location(
    "spectral_norm",
    os.path.join(os.path.dirname(__file__), "..", "src", "utils", "spectral_norm.py")
)
spectral_norm_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(spectral_norm_module)

spectral_norm = spectral_norm_module.spectral_norm
apply_spectral_norm = spectral_norm_module.apply_spectral_norm


class TestSpectralNorm:
    """Test spectral norm computation."""

    def test_spectral_norm_identity(self):
        """Test that identity matrix has spectral norm of 1."""
        I = torch.eye(100)
        sn, _ = spectral_norm(I, num_iter=20)
        assert abs(sn.item() - 1.0) < 0.01, f"Expected ~1.0, got {sn.item()}"

    def test_spectral_norm_scaled_identity(self):
        """Test that scaled identity has spectral norm equal to scale."""
        scale = 5.0
        I = torch.eye(100) * scale
        sn, _ = spectral_norm(I, num_iter=20)
        assert abs(sn.item() - scale) < 0.1, f"Expected ~{scale}, got {sn.item()}"

    def test_spectral_norm_positive(self):
        """Test that spectral norm is always positive."""
        W = torch.randn(64, 32)
        sn, _ = spectral_norm(W, num_iter=10)
        assert sn.item() > 0, "Spectral norm must be positive"

    def test_spectral_norm_convergence(self):
        """Test that more iterations lead to more stable results."""
        W = torch.randn(100, 50)
        sn_10, _ = spectral_norm(W, num_iter=10)
        sn_50, _ = spectral_norm(W, num_iter=50)
        # Results should be very close with more iterations
        assert abs(sn_10.item() - sn_50.item()) / sn_50.item() < 0.1


class TestApplySpectralNorm:
    """Test per-layer spectral norm clipping."""

    @pytest.fixture
    def mixed_lora(self):
        """Create LoRA with mixed spectral norms for testing."""
        torch.manual_seed(42)
        return {
            "small.weight": torch.randn(64, 32) * 0.02,   # SN ~0.3 (below target)
            "medium.weight": torch.randn(64, 32) * 0.05,  # SN ~0.7 (below target)
            "large.weight": torch.randn(64, 32) * 0.2,    # SN ~3 (above target)
            "huge.weight": torch.randn(64, 32) * 2.0,     # SN ~25 (way above)
            "alpha": torch.tensor(32.0),
        }

    def test_per_layer_clipping(self, mixed_lora):
        """Test that per-layer clipping preserves layers below target."""
        target = 1.0
        result = apply_spectral_norm(mixed_lora, scale=target, num_iter=10)

        # Check that layers below target are preserved (minimal change)
        for key in ["small.weight", "medium.weight"]:
            original_sn, _ = spectral_norm(mixed_lora[key], num_iter=10)
            result_sn, _ = spectral_norm(result[key], num_iter=10)

            # Allow for numerical precision issues
            if original_sn.item() < target:
                # Layer should be mostly unchanged
                change_ratio = abs(result_sn.item() - original_sn.item()) / original_sn.item()
                assert change_ratio < 0.05, (
                    f"{key}: Expected minimal change for layer below target, "
                    f"but got {change_ratio*100:.1f}% change"
                )

    def test_clipping_exceeding_layers(self, mixed_lora):
        """Test that layers exceeding target are clipped."""
        target = 1.0
        result = apply_spectral_norm(mixed_lora, scale=target, num_iter=10)

        # Check that all layers are at or below target
        for key, weight in result.items():
            if "alpha" in key or weight.ndim < 2:
                continue

            result_sn, _ = spectral_norm(weight, num_iter=10)
            assert result_sn.item() <= target * 1.1, (  # Allow 10% tolerance
                f"{key}: Expected SN <= {target}, got {result_sn.item()}"
            )

    def test_alpha_preservation(self, mixed_lora):
        """Test that alpha values are preserved unchanged."""
        target = 1.0
        result = apply_spectral_norm(mixed_lora, scale=target, num_iter=10)

        assert torch.equal(result["alpha"], mixed_lora["alpha"]), (
            "Alpha values must be preserved unchanged"
        )

    def test_better_than_global_scaling(self, mixed_lora):
        """Test that per-layer clipping preserves more strength than global scaling."""
        target = 1.0

        # Compute global scaling approach (OLD behavior)
        layer_sns = {}
        for key, weight in mixed_lora.items():
            if "alpha" in key or weight.ndim < 2:
                continue
            sn, _ = spectral_norm(weight, num_iter=10)
            layer_sns[key] = sn.item()

        max_sn = max(layer_sns.values())
        global_scale_factor = target / max_sn

        # Apply global scaling to all layers
        global_scaled = {}
        for key, weight in mixed_lora.items():
            if "alpha" in key:
                global_scaled[key] = weight
            else:
                global_scaled[key] = weight * global_scale_factor

        # Apply per-layer clipping (NEW behavior)
        per_layer_clipped = apply_spectral_norm(mixed_lora, scale=target, num_iter=10)

        # Compare average spectral norms
        global_avg = sum(
            spectral_norm(global_scaled[k], num_iter=10)[0].item()
            for k in layer_sns.keys()
        ) / len(layer_sns)

        per_layer_avg = sum(
            spectral_norm(per_layer_clipped[k], num_iter=10)[0].item()
            for k in layer_sns.keys()
        ) / len(layer_sns)

        # Per-layer clipping should preserve more strength on average
        assert per_layer_avg > global_avg, (
            f"Per-layer clipping should preserve more strength: "
            f"per_layer={per_layer_avg:.4f}, global={global_avg:.4f}"
        )

    def test_empty_dict(self):
        """Test handling of empty dictionary."""
        result = apply_spectral_norm({}, scale=1.0)
        assert result == {}, "Empty dict should return empty dict"

    def test_only_alpha(self):
        """Test handling of dict with only alpha values."""
        lora = {"alpha": torch.tensor(32.0)}
        result = apply_spectral_norm(lora, scale=1.0)
        assert torch.equal(result["alpha"], lora["alpha"])

    def test_1d_tensors_skipped(self):
        """Test that 1D tensors (biases) are skipped."""
        lora = {
            "weight": torch.randn(64, 32),
            "bias": torch.randn(64),  # 1D tensor
        }
        result = apply_spectral_norm(lora, scale=1.0)

        # Bias should be unchanged
        assert torch.equal(result["bias"], lora["bias"])

    def test_different_scales(self):
        """Test behavior with different target scales."""
        torch.manual_seed(42)
        lora = {"weight": torch.randn(64, 32) * 5.0}  # SN ~60

        # Test conservative scaling
        result_conservative = apply_spectral_norm(lora, scale=0.5)
        sn_conservative, _ = spectral_norm(result_conservative["weight"], num_iter=10)
        assert sn_conservative.item() <= 0.6  # Allow tolerance

        # Test aggressive scaling
        result_aggressive = apply_spectral_norm(lora, scale=10.0)
        sn_aggressive, _ = spectral_norm(result_aggressive["weight"], num_iter=10)
        assert sn_aggressive.item() <= 11.0  # Should be close to or below target


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
