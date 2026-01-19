"""
Test adapter initialization and numerical stability.

Verifies that adapters don't produce NaN values with proper initialization.
"""

import torch
from src.semantic.adapter.output_space import OutputSpaceAdapter
from src.semantic.adapter.rank_space import RankSpaceAdapter


def test_output_space_adapter_initialization():
    """Test that OutputSpaceAdapter initializes without NaN and produces valid outputs."""
    adapter = OutputSpaceAdapter(
        layer_key="test.layer",
        semantic_dim=8,
        feature_names=["hair", "eyes", "clothing"],
        output_dim=768,
        num_loras=2,
    )

    # Check that all weights are finite (not NaN/Inf)
    for name, param in adapter.named_parameters():
        assert torch.isfinite(param).all(), f"Parameter {name} contains NaN/Inf after initialization"

    # Test forward pass with synthetic inputs
    batch_size = 4
    seq_len = 77
    channels = 768

    lora_deltas = {
        "lora1": torch.randn(batch_size, seq_len, channels) * 0.1,
        "lora2": torch.randn(batch_size, seq_len, channels) * 0.1,
    }
    semantic_vector = torch.randn(batch_size, 8)

    # Forward pass
    output = adapter(lora_deltas, semantic_vector)

    # Check output is finite
    assert torch.isfinite(output).all(), "Output contains NaN/Inf"
    assert output.shape == (batch_size, seq_len, channels), f"Wrong output shape: {output.shape}"

    print("✓ OutputSpaceAdapter: Initialization and forward pass successful")
    print(f"  Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")


def test_rank_space_adapter_initialization():
    """Test that RankSpaceAdapter initializes without NaN and produces valid outputs."""
    adapter = RankSpaceAdapter(
        layer_key="test.layer",
        semantic_dim=8,
        feature_names=["hair", "eyes", "clothing"],
        rank=32,
        num_loras=2,
        use_residual_mlp=True,
    )

    # Check that all weights are finite (not NaN/Inf)
    for name, param in adapter.named_parameters():
        assert torch.isfinite(param).all(), f"Parameter {name} contains NaN/Inf after initialization"

    # Test forward pass with synthetic inputs
    batch_size = 4
    seq_len = 77
    rank = 32
    out_dim = 768

    lora_down_outputs = {
        "lora1": torch.randn(batch_size, seq_len, rank) * 0.1,
        "lora2": torch.randn(batch_size, seq_len, rank) * 0.1,
    }
    b_shared = torch.randn(out_dim, rank) * 0.1
    semantic_vector = torch.randn(batch_size, 8)

    # Forward pass
    output = adapter(lora_down_outputs, b_shared, semantic_vector)

    # Check output is finite
    assert torch.isfinite(output).all(), "Output contains NaN/Inf"
    assert output.shape == (batch_size, seq_len, out_dim), f"Wrong output shape: {output.shape}"

    print("✓ RankSpaceAdapter: Initialization and forward pass successful")
    print(f"  Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")


def test_adapter_with_extreme_inputs():
    """Test that adapters handle extreme (but valid) input values."""
    adapter = OutputSpaceAdapter(
        layer_key="test.layer",
        semantic_dim=8,
        feature_names=["hair", "eyes", "clothing"],
        output_dim=768,
        num_loras=2,
    )

    batch_size = 4
    seq_len = 77
    channels = 768

    # Test with larger input values (but still reasonable for normalized deltas)
    lora_deltas = {
        "lora1": torch.randn(batch_size, seq_len, channels) * 1.0,  # Larger scale
        "lora2": torch.randn(batch_size, seq_len, channels) * 1.0,
    }
    semantic_vector = torch.randn(batch_size, 8)

    # Forward pass
    output = adapter(lora_deltas, semantic_vector)

    # Check output is still finite
    assert torch.isfinite(output).all(), "Output contains NaN/Inf with larger inputs"

    print("✓ Adapter handles extreme inputs without NaN/Inf")
    print(f"  Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")


def test_adapter_input_validation():
    """Test that adapters correctly detect NaN/Inf in inputs."""
    adapter = OutputSpaceAdapter(
        layer_key="test.layer",
        semantic_dim=8,
        feature_names=["hair", "eyes", "clothing"],
        output_dim=768,
        num_loras=2,
    )

    batch_size = 4
    seq_len = 77
    channels = 768

    # Test with NaN in delta
    lora_deltas_nan = {
        "lora1": torch.randn(batch_size, seq_len, channels),
        "lora2": torch.full((batch_size, seq_len, channels), float('nan')),
    }
    semantic_vector = torch.randn(batch_size, 8)

    try:
        output = adapter(lora_deltas_nan, semantic_vector)
        assert False, "Should have raised ValueError for NaN input"
    except ValueError as e:
        assert "NaN detected" in str(e)
        print("✓ Adapter correctly detects NaN in input deltas")

    # Test with Inf in semantic vector
    lora_deltas = {
        "lora1": torch.randn(batch_size, seq_len, channels),
        "lora2": torch.randn(batch_size, seq_len, channels),
    }
    semantic_vector_inf = torch.full((batch_size, 8), float('inf'))

    try:
        output = adapter(lora_deltas, semantic_vector_inf)
        assert False, "Should have raised ValueError for Inf input"
    except ValueError as e:
        assert "Inf detected" in str(e)
        print("✓ Adapter correctly detects Inf in semantic_vector")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Adapter Initialization and Numerical Stability")
    print("=" * 60)

    try:
        test_output_space_adapter_initialization()
        test_rank_space_adapter_initialization()
        test_adapter_with_extreme_inputs()
        test_adapter_input_validation()

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED ✓")
        print("=" * 60)
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        import sys
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        import sys
        sys.exit(1)
