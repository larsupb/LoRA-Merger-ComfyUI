"""
Test semantic key projection in gradient analyzer.

Verifies that diffusers parameter paths are correctly projected to
semantic group keys for adapter training compatibility.
"""

import pytest
import torch

from src.semantic.gradient_analyzer import GradientSemanticAnalyzer
from src.semantic.training.semantic_key_projection import project_layer_key_to_semantic_key


@pytest.fixture
def mock_clip():
    """Mock CLIP object for testing."""
    class MockClip:
        pass
    return MockClip()


def test_semantic_key_projection():
    """Test that layer keys are projected to semantic groups."""
    # Test projection function
    assert project_layer_key_to_semantic_key("down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q") == "self_attention.query"
    assert project_layer_key_to_semantic_key("mid_block.attentions.0.transformer_blocks.0.attn2.to_v") == "cross_attention.value"
    assert project_layer_key_to_semantic_key("up_blocks.1.attentions.1.transformer_blocks.0.attn1.to_out") == "self_attention.output"
    assert project_layer_key_to_semantic_key("down_blocks.0.attentions.0.transformer_blocks.0.ff.net.0") == "mlp"


def test_normalize_semantic_map_with_diffusers_paths(mock_clip):
    """Test that _normalize_semantic_map projects diffusers paths to semantic keys."""
    analyzer = GradientSemanticAnalyzer(
        clip=mock_clip,
        device=torch.device("cpu"),
        dtype=torch.float32
    )

    # Create semantic map with diffusers parameter paths (what gradient analyzer produces)
    semantic_map = {
        "hair": {
            "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.lora_A.weight": torch.tensor(0.8),
            "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_k.lora_A.weight": torch.tensor(0.7),
            "mid_block.attentions.0.transformer_blocks.0.attn2.to_v.lora_A.weight": torch.tensor(0.3),
        },
        "eyes": {
            "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.lora_A.weight": torch.tensor(0.2),
            "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_k.lora_A.weight": torch.tensor(0.3),
            "mid_block.attentions.0.transformer_blocks.0.attn2.to_v.lora_A.weight": torch.tensor(0.7),
        },
    }

    # Normalize (which should also project keys)
    normalized = analyzer._normalize_semantic_map(semantic_map)

    # Verify the output has semantic group keys, not diffusers paths
    assert "hair" in normalized
    assert "eyes" in normalized

    hair_keys = list(normalized["hair"].keys())
    eyes_keys = list(normalized["eyes"].keys())

    # Should have semantic group keys
    assert "self_attention.query" in hair_keys or "self_attention.key" in hair_keys or "cross_attention.value" in hair_keys
    assert "self_attention.query" in eyes_keys or "self_attention.key" in eyes_keys or "cross_attention.value" in eyes_keys

    # Should NOT have diffusers parameter paths
    for key in hair_keys:
        assert not key.startswith("down_blocks")
        assert not key.startswith("mid_block")
        assert "lora_A" not in key
        assert "weight" not in key

    for key in eyes_keys:
        assert not key.startswith("down_blocks")
        assert not key.startswith("mid_block")
        assert "lora_A" not in key
        assert "weight" not in key

    # Verify normalization (values should sum to 1.0 per semantic group)
    # For each semantic group, sum across features should be close to 1.0
    all_keys = set(hair_keys) | set(eyes_keys)
    for semantic_key in all_keys:
        total = 0.0
        if semantic_key in normalized["hair"]:
            total += normalized["hair"][semantic_key].item()
        if semantic_key in normalized["eyes"]:
            total += normalized["eyes"][semantic_key].item()

        # Should be normalized (sum ≈ 1.0)
        assert abs(total - 1.0) < 0.01, f"Semantic key {semantic_key} not normalized: sum={total}"

    print("\n✓ Semantic map successfully projected to semantic group keys")
    print(f"  Hair keys: {hair_keys}")
    print(f"  Eyes keys: {eyes_keys}")


def test_aggregation_of_same_semantic_group(mock_clip):
    """Test that multiple diffusers paths mapping to the same semantic group are aggregated."""
    analyzer = GradientSemanticAnalyzer(
        clip=mock_clip,
        device=torch.device("cpu"),
        dtype=torch.float32
    )

    # Create semantic map with multiple paths that map to the same semantic group
    # Both should map to "self_attention.query"
    semantic_map = {
        "hair": {
            "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.lora_A.weight": torch.tensor(0.6),
            "up_blocks.1.attentions.0.transformer_blocks.0.attn1.to_q.lora_B.weight": torch.tensor(0.4),
            # Both map to self_attention.query, should be averaged to 0.5
        },
    }

    normalized = analyzer._normalize_semantic_map(semantic_map)

    # Verify aggregation happened
    assert "self_attention.query" in normalized["hair"]

    # Should have averaged the two values: (0.6 + 0.4) / 2 = 0.5
    # Then normalized to 1.0 (since it's the only semantic group for 'hair')
    assert normalized["hair"]["self_attention.query"].item() == pytest.approx(1.0, abs=0.01)

    print("\n✓ Multiple diffusers paths correctly aggregated to single semantic group")


if __name__ == "__main__":
    # Simple test runner
    import sys

    class MockClipLocal:
        pass

    clip = MockClipLocal()

    print("=" * 60)
    print("Testing Semantic Key Projection")
    print("=" * 60)

    try:
        test_semantic_key_projection()
        test_normalize_semantic_map_with_diffusers_paths(clip)
        test_aggregation_of_same_semantic_group(clip)
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED ✓")
        print("=" * 60)
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
