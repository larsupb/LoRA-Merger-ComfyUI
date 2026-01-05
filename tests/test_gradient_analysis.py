"""
Tests for gradient-based semantic analysis.

This test file validates the GradientSemanticAnalyzer implementation.
"""

import pytest
import torch
from unittest.mock import Mock, MagicMock, patch

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from analysis.gradient_analyzer import GradientSemanticAnalyzer
from analysis.feature_prompts import get_feature_prompts, get_all_features


class TestGradientSemanticAnalyzer:
    """Test GradientSemanticAnalyzer class."""

    @pytest.fixture
    def mock_model(self):
        """Create mock ComfyUI MODEL object."""
        model = Mock()
        model.clone = Mock(return_value=model)
        model.add_patches = Mock()

        # Mock apply_model to return a tensor
        def apply_model_fn(latent, timestep, **kwargs):
            # Return a tensor that requires grad
            return torch.randn(1, 4, 64, 64, requires_grad=True)

        model.apply_model = Mock(side_effect=apply_model_fn)

        return model

    @pytest.fixture
    def mock_clip(self):
        """Create mock ComfyUI CLIP object."""
        clip = Mock()

        # Mock tokenize
        clip.tokenize = Mock(return_value={"input_ids": torch.randint(0, 1000, (1, 77))})

        # Mock encode_from_tokens
        def encode_fn(tokens, return_pooled=False):
            cond = torch.randn(1, 77, 768)
            pooled = torch.randn(1, 768) if return_pooled else None
            if return_pooled:
                return cond, pooled
            return cond

        clip.encode_from_tokens = Mock(side_effect=encode_fn)

        return clip

    @pytest.fixture
    def mock_lora_patches(self):
        """Create mock LoRA patches."""
        from comfy.weight_adapter import LoRAAdapter

        patches = {}

        # Create a few mock layers
        for layer_idx in range(5):
            layer_key = f"diffusion_model.layers.{layer_idx}.attention.qkv"

            # Create LoRAAdapter with up/down weights
            up = torch.randn(64, 16, requires_grad=True)
            down = torch.randn(16, 64, requires_grad=True)
            alpha = torch.tensor(16.0)

            weights = (up, down, alpha, None, None, None)
            adapter = LoRAAdapter(weights=weights, loaded_keys=set())

            patches[layer_key] = adapter

        return patches

    def test_initialization(self, mock_model, mock_clip):
        """Test analyzer initialization."""
        analyzer = GradientSemanticAnalyzer(mock_model, mock_clip)

        assert analyzer.model is mock_model
        assert analyzer.clip is mock_clip
        assert analyzer.device is not None
        assert analyzer.dtype is not None

    def test_create_lora_model(self, mock_model, mock_clip, mock_lora_patches):
        """Test temporary model creation with LoRA."""
        analyzer = GradientSemanticAnalyzer(mock_model, mock_clip)

        temp_model = analyzer._create_lora_model(mock_lora_patches)

        # Verify model was cloned
        mock_model.clone.assert_called_once()

        # Verify patches were added
        temp_model.add_patches.assert_called_once_with(mock_lora_patches, strength_patch=1.0)

    def test_collect_lora_params(self, mock_model, mock_clip, mock_lora_patches):
        """Test LoRA parameter collection."""
        analyzer = GradientSemanticAnalyzer(mock_model, mock_clip)

        params = analyzer._collect_lora_params(mock_lora_patches)

        # Should collect both up and down for each layer
        assert len(params) == 2 * len(mock_lora_patches)  # 2 params per layer (up, down)

        # Check parameter names
        for layer_key in mock_lora_patches.keys():
            assert f"{layer_key}.up" in params
            assert f"{layer_key}.down" in params

        # Check tensors require gradients
        for param in params.values():
            assert param.requires_grad

    def test_create_zero_attributions(self, mock_model, mock_clip, mock_lora_patches):
        """Test zero attribution creation."""
        analyzer = GradientSemanticAnalyzer(mock_model, mock_clip)

        zero_attrs = analyzer._create_zero_attributions(mock_lora_patches)

        # Should have same keys as params
        assert len(zero_attrs) == 2 * len(mock_lora_patches)

        # All should be zeros
        for attr in zero_attrs.values():
            assert torch.all(attr == 0)

    def test_average_attributions(self, mock_model, mock_clip):
        """Test attribution averaging."""
        analyzer = GradientSemanticAnalyzer(mock_model, mock_clip)

        # Create multiple attribution dicts
        attr1 = {"layer1.up": torch.ones(10, 10)}
        attr2 = {"layer1.up": torch.ones(10, 10) * 2}
        attr3 = {"layer1.up": torch.ones(10, 10) * 3}

        averaged = analyzer._average_attributions([attr1, attr2, attr3])

        # Should average to 2.0
        assert torch.allclose(averaged["layer1.up"], torch.ones(10, 10) * 2.0)

    def test_normalize_semantic_map(self, mock_model, mock_clip):
        """Test semantic map normalization."""
        analyzer = GradientSemanticAnalyzer(mock_model, mock_clip)

        # Create semantic map with unnormalized values
        semantic_map = {
            "hair": {"layer1": torch.ones(10, 10) * 2.0},
            "eyes": {"layer1": torch.ones(10, 10) * 1.0},
            "clothing": {"layer1": torch.ones(10, 10) * 1.0},
        }

        normalized = analyzer._normalize_semantic_map(semantic_map)

        # Sum should be 1.0 for each element
        total = (
            normalized["hair"]["layer1"]
            + normalized["eyes"]["layer1"]
            + normalized["clothing"]["layer1"]
        )

        assert torch.allclose(total, torch.ones(10, 10), atol=1e-6)

        # Hair should be 0.5 (2/4), eyes and clothing 0.25 each (1/4)
        assert torch.allclose(normalized["hair"]["layer1"], torch.ones(10, 10) * 0.5, atol=1e-6)
        assert torch.allclose(normalized["eyes"]["layer1"], torch.ones(10, 10) * 0.25, atol=1e-6)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires GPU")
    def test_compute_feature_attribution_integration(self, mock_model, mock_clip, mock_lora_patches):
        """
        Integration test for attribution computation.

        This test requires a working PyTorch autograd setup.
        """
        analyzer = GradientSemanticAnalyzer(mock_model, mock_clip)

        # This is a smoke test - just ensure it doesn't crash
        try:
            attribution = analyzer._compute_feature_attribution(
                mock_model,
                mock_lora_patches,
                prompt="1girl, focus on hair, detailed hair",
                feature="hair"
            )

            # Should return a dict if successful
            assert attribution is None or isinstance(attribution, dict)

        except Exception as e:
            # Expected to fail with mock objects, but shouldn't crash before reaching model call
            assert "apply_model" in str(e) or "autograd" in str(e).lower()

    def test_feature_prompts_integration(self):
        """Test feature prompt generation."""
        # Get all available features
        features = get_all_features()
        assert len(features) > 0

        # Get prompts for each feature
        for feature in features:
            prompts = get_feature_prompts(feature, architecture="sdxl", num_prompts=3)
            assert len(prompts) == 3
            assert all(isinstance(p, str) for p in prompts)
            assert all(len(p) > 0 for p in prompts)

    def test_convert_to_semantic_map(self, mock_model, mock_clip, mock_lora_patches):
        """Test conversion from raw attributions to semantic map."""
        analyzer = GradientSemanticAnalyzer(mock_model, mock_clip)

        # Create mock feature attributions
        feature_attributions = {
            "hair": {
                "diffusion_model.layers.0.attention.qkv.up": torch.ones(64, 16),
                "diffusion_model.layers.0.attention.qkv.down": torch.ones(16, 64),
            },
            "eyes": {
                "diffusion_model.layers.0.attention.qkv.up": torch.ones(64, 16) * 0.5,
                "diffusion_model.layers.0.attention.qkv.down": torch.ones(16, 64) * 0.5,
            },
        }

        features = ["hair", "eyes"]

        semantic_map = analyzer._convert_to_semantic_map(
            feature_attributions,
            mock_lora_patches,
            features
        )

        # Should have entries for each feature
        assert "hair" in semantic_map
        assert "eyes" in semantic_map

        # Should have layer keys (without .up/.down suffixes)
        for feature in features:
            assert "diffusion_model.layers.0.attention.qkv" in semantic_map[feature]


class TestFeaturePrompts:
    """Test feature prompt utilities."""

    def test_get_all_features(self):
        """Test getting all available features."""
        features = get_all_features()

        assert isinstance(features, list)
        assert len(features) > 0
        assert "hair" in features
        assert "eyes" in features
        assert "clothing" in features

    def test_get_feature_prompts_default(self):
        """Test getting feature prompts with defaults."""
        prompts = get_feature_prompts("hair")

        assert isinstance(prompts, list)
        assert len(prompts) == 3  # default num_prompts
        assert all(isinstance(p, str) for p in prompts)

    def test_get_feature_prompts_custom_count(self):
        """Test getting custom number of prompts."""
        prompts = get_feature_prompts("hair", num_prompts=2)

        assert len(prompts) == 2

    def test_get_feature_prompts_architecture_adjustment(self):
        """Test architecture-specific prompt adjustments."""
        sd15_prompts = get_feature_prompts("hair", architecture="sd15")
        sdxl_prompts = get_feature_prompts("hair", architecture="sdxl")

        # SD1.5 should have quality prefix
        assert any("masterpiece" in p or "best quality" in p for p in sd15_prompts)

        # Prompts should be different
        assert sd15_prompts != sdxl_prompts

    def test_custom_feature(self):
        """Test custom feature name support."""
        # Should not raise error for custom features
        prompts = get_feature_prompts("uniform")

        assert len(prompts) == 3
        assert all(isinstance(p, str) for p in prompts)
        assert all("uniform" in p.lower() for p in prompts)

    def test_custom_feature_variations(self):
        """Test various custom feature names."""
        custom_features = ["weapon", "background", "tattoo", "wings", "armor"]

        for feature in custom_features:
            prompts = get_feature_prompts(feature, num_prompts=2)

            assert len(prompts) == 2
            assert all(feature in p.lower() for p in prompts)
            assert all(isinstance(p, str) for p in prompts)

    def test_predefined_vs_custom_features(self):
        """Test that predefined features use optimized prompts, custom use generic."""
        # Predefined feature
        hair_prompts = get_feature_prompts("hair", num_prompts=1)

        # Custom feature
        sword_prompts = get_feature_prompts("sword", num_prompts=1)

        # Both should work
        assert len(hair_prompts) == 1
        assert len(sword_prompts) == 1

        # Custom should contain the feature name
        assert "sword" in sword_prompts[0].lower()


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
