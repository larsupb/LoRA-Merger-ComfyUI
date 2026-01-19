"""
Test adapter dtype safety and float8 blocking.

Verifies that adapters block conversion to float8 dtypes to prevent NaN.
"""

import torch
from src.semantic.adapter.output_space import OutputSpaceAdapter
from src.semantic.adapter.rank_space import RankSpaceAdapter


def test_output_space_adapter_blocks_float8():
    """Test that OutputSpaceAdapter blocks float8 conversion."""
    adapter = OutputSpaceAdapter(
        layer_key="test.layer",
        semantic_dim=8,
        feature_names=["hair", "eyes", "clothing"],
        output_dim=768,
        num_loras=2,
    )

    # Initial dtype should be float32 (default)
    initial_dtype = next(adapter.gate_network.parameters()).dtype
    print(f"Initial dtype: {initial_dtype}")
    assert initial_dtype in [torch.float32, torch.float16, torch.bfloat16]

    # Try to convert to float8_e4m3fn (should be blocked)
    adapter_float8 = adapter.to(dtype=torch.float8_e4m3fn)

    # Check that it was converted to float16 instead
    final_dtype = next(adapter_float8.gate_network.parameters()).dtype
    print(f"After float8 conversion attempt: {final_dtype}")
    assert final_dtype == torch.float16, f"Expected float16, got {final_dtype}"

    # Verify all parameters are float16
    for name, param in adapter_float8.named_parameters():
        assert param.dtype == torch.float16, f"Parameter {name} has dtype {param.dtype}"

    # Verify parameters are still valid (not NaN)
    for name, param in adapter_float8.named_parameters():
        assert torch.isfinite(param).all(), f"Parameter {name} contains NaN/Inf"

    print("✓ OutputSpaceAdapter correctly blocks float8 conversion")


def test_rank_space_adapter_blocks_float8():
    """Test that RankSpaceAdapter blocks float8 conversion."""
    adapter = RankSpaceAdapter(
        layer_key="test.layer",
        semantic_dim=8,
        feature_names=["hair", "eyes", "clothing"],
        rank=32,
        num_loras=2,
        use_residual_mlp=True,
    )

    # Try to convert to float8_e5m2 (should be blocked)
    adapter_float8 = adapter.to(dtype=torch.float8_e5m2)

    # Check that it was converted to float16 instead
    final_dtype = next(adapter_float8.gate_network.parameters()).dtype
    print(f"RankSpaceAdapter after float8 conversion attempt: {final_dtype}")
    assert final_dtype == torch.float16, f"Expected float16, got {final_dtype}"

    # Verify parameters are still valid (not NaN)
    for name, param in adapter_float8.named_parameters():
        assert torch.isfinite(param).all(), f"Parameter {name} contains NaN/Inf"

    print("✓ RankSpaceAdapter correctly blocks float8 conversion")


def test_adapter_forward_with_mixed_dtypes():
    """Test adapter forward pass with inputs in different dtypes."""
    adapter = OutputSpaceAdapter(
        layer_key="test.layer",
        semantic_dim=8,
        feature_names=["hair", "eyes", "clothing"],
        output_dim=768,
        num_loras=2,
    )

    # Convert adapter to float16
    adapter = adapter.to(dtype=torch.float16)

    batch_size = 4
    seq_len = 77
    channels = 768

    # Input deltas in float16 (simulating UNet outputs)
    lora_deltas = {
        "lora1": torch.randn(batch_size, seq_len, channels, dtype=torch.float16) * 0.1,
        "lora2": torch.randn(batch_size, seq_len, channels, dtype=torch.float16) * 0.1,
    }
    semantic_vector = torch.randn(batch_size, 8, dtype=torch.float16)

    # Forward pass
    output = adapter(lora_deltas, semantic_vector)

    # Verify output is valid
    assert torch.isfinite(output).all(), "Output contains NaN/Inf"
    assert output.dtype == torch.float16, f"Output dtype mismatch: {output.dtype}"
    assert output.shape == (batch_size, seq_len, channels)

    print("✓ Adapter handles float16 inputs correctly")
    print(f"  Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")


def test_adapter_accepts_valid_dtypes():
    """Test that adapters accept float32/float16/bfloat16."""
    adapter = OutputSpaceAdapter(
        layer_key="test.layer",
        semantic_dim=8,
        feature_names=["hair", "eyes", "clothing"],
        output_dim=768,
        num_loras=2,
    )

    # Test float32
    adapter_f32 = adapter.to(dtype=torch.float32)
    assert next(adapter_f32.parameters()).dtype == torch.float32
    print("✓ Adapter accepts float32")

    # Test float16
    adapter_f16 = adapter.to(dtype=torch.float16)
    assert next(adapter_f16.parameters()).dtype == torch.float16
    print("✓ Adapter accepts float16")

    # Test bfloat16
    adapter_bf16 = adapter.to(dtype=torch.bfloat16)
    assert next(adapter_bf16.parameters()).dtype == torch.bfloat16
    print("✓ Adapter accepts bfloat16")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Adapter Dtype Safety")
    print("=" * 60)

    try:
        test_output_space_adapter_blocks_float8()
        test_rank_space_adapter_blocks_float8()
        test_adapter_forward_with_mixed_dtypes()
        test_adapter_accepts_valid_dtypes()

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
