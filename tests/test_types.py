"""
Unit tests for type system and validators.

Tests type guards, validators, and type definitions from src/types.py.
"""

import pytest
import torch
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from types import (
    is_lora_tensors,
    is_lora_stack,
    validate_lora_tensors,
    validate_lora_stack,
    LORA_TENSORS,
    LORA_STACK,
    MIN_SINGULAR_VALUE,
)


class TestLoRATensorsTypeGuard:
    """Tests for is_lora_tensors type guard."""

    def test_valid_lora_tensors(self):
        """Test that valid LORA_TENSORS tuple is recognized."""
        up = torch.randn(10, 5)
        down = torch.randn(5, 10)
        alpha = torch.tensor(1.0)
        tensors = (up, down, alpha)

        assert is_lora_tensors(tensors)

    def test_valid_lora_tensors_with_float_alpha(self):
        """Test that LORA_TENSORS with float alpha is recognized."""
        up = torch.randn(10, 5)
        down = torch.randn(5, 10)
        alpha = 1.0  # Float instead of tensor
        tensors = (up, down, alpha)

        assert is_lora_tensors(tensors)

    def test_invalid_not_tuple(self):
        """Test that non-tuple is rejected."""
        assert not is_lora_tensors([torch.randn(10, 5), torch.randn(5, 10), 1.0])

    def test_invalid_wrong_length(self):
        """Test that tuple with wrong length is rejected."""
        assert not is_lora_tensors((torch.randn(10, 5), torch.randn(5, 10)))

    def test_invalid_non_tensor_elements(self):
        """Test that tuple with non-tensor up/down is rejected."""
        assert not is_lora_tensors(("not a tensor", torch.randn(5, 10), 1.0))
        assert not is_lora_tensors((torch.randn(10, 5), "not a tensor", 1.0))

    def test_invalid_alpha_type(self):
        """Test that invalid alpha type is rejected."""
        up = torch.randn(10, 5)
        down = torch.randn(5, 10)
        assert not is_lora_tensors((up, down, "invalid"))


class TestLoRAStackTypeGuard:
    """Tests for is_lora_stack type guard."""

    def test_valid_lora_stack(self):
        """Test that valid LORA_STACK is recognized."""
        from comfy.weight_adapter import LoRAAdapter

        # Create mock LoRAAdapter objects
        adapter1 = LoRAAdapter("lora", (torch.randn(10, 5), torch.randn(5, 10), 1.0, None, None, None))
        adapter2 = LoRAAdapter("lora", (torch.randn(10, 5), torch.randn(5, 10), 1.0, None, None, None))

        stack = {
            "lora1": {"layer1": adapter1},
            "lora2": {"layer2": adapter2},
        }

        assert is_lora_stack(stack)

    def test_invalid_not_dict(self):
        """Test that non-dict is rejected."""
        assert not is_lora_stack([])
        assert not is_lora_stack("not a dict")

    def test_invalid_non_string_keys(self):
        """Test that dict with non-string keys is rejected."""
        assert not is_lora_stack({1: {}})

    def test_invalid_non_dict_values(self):
        """Test that dict with non-dict values is rejected."""
        assert not is_lora_stack({"lora1": "not a dict"})


class TestValidateLoRATensors:
    """Tests for validate_lora_tensors validator."""

    def test_valid_2d_tensors(self):
        """Test validation of valid 2D LoRA tensors."""
        up = torch.randn(10, 5)
        down = torch.randn(5, 10)
        alpha = torch.tensor(1.0)
        tensors = (up, down, alpha)

        # Should not raise
        validate_lora_tensors(tensors)

    def test_valid_4d_conv_tensors(self):
        """Test validation of valid 4D convolutional LoRA tensors."""
        up = torch.randn(10, 5, 1, 1)
        down = torch.randn(5, 10, 1, 1)
        alpha = torch.tensor(1.0)
        tensors = (up, down, alpha)

        # Should not raise
        validate_lora_tensors(tensors)

    def test_invalid_structure(self):
        """Test that invalid structure raises ValueError."""
        with pytest.raises(ValueError, match="Invalid LORA_TENSORS structure"):
            validate_lora_tensors("not a tuple")

    def test_invalid_up_dimensions(self):
        """Test that invalid up tensor dimensions raise ValueError."""
        up = torch.randn(10)  # 1D tensor
        down = torch.randn(5, 10)
        alpha = torch.tensor(1.0)
        tensors = (up, down, alpha)

        with pytest.raises(ValueError, match="Invalid up tensor dimensions"):
            validate_lora_tensors(tensors)

    def test_invalid_down_dimensions(self):
        """Test that invalid down tensor dimensions raise ValueError."""
        up = torch.randn(10, 5)
        down = torch.randn(5)  # 1D tensor
        alpha = torch.tensor(1.0)
        tensors = (up, down, alpha)

        with pytest.raises(ValueError, match="Invalid down tensor dimensions"):
            validate_lora_tensors(tensors)


class TestValidateLoRAStack:
    """Tests for validate_lora_stack validator."""

    def test_valid_stack(self):
        """Test validation of valid LoRA stack."""
        from comfy.weight_adapter import LoRAAdapter

        adapter = LoRAAdapter("lora", (torch.randn(10, 5), torch.randn(5, 10), 1.0, None, None, None))
        stack = {"lora1": {"layer1": adapter}}

        # Should not raise
        validate_lora_stack(stack)

    def test_invalid_structure(self):
        """Test that invalid structure raises ValueError."""
        with pytest.raises(ValueError, match="Invalid LORA_STACK structure"):
            validate_lora_stack("not a dict")

    def test_empty_stack(self):
        """Test that empty stack raises ValueError."""
        with pytest.raises(ValueError, match="LORA_STACK cannot be empty"):
            validate_lora_stack({})


class TestConstants:
    """Tests for constants defined in types module."""

    def test_min_singular_value(self):
        """Test that MIN_SINGULAR_VALUE is defined correctly."""
        assert MIN_SINGULAR_VALUE == 1e-6
        assert isinstance(MIN_SINGULAR_VALUE, float)


# Fixtures for reusable test data

@pytest.fixture
def sample_lora_tensors():
    """Fixture providing sample LORA_TENSORS."""
    up = torch.randn(10, 5)
    down = torch.randn(5, 10)
    alpha = torch.tensor(1.0)
    return (up, down, alpha)


@pytest.fixture
def sample_lora_stack():
    """Fixture providing sample LORA_STACK."""
    from comfy.weight_adapter import LoRAAdapter

    adapter1 = LoRAAdapter("lora", (torch.randn(10, 5), torch.randn(5, 10), 1.0, None, None, None))
    adapter2 = LoRAAdapter("lora", (torch.randn(8, 4), torch.randn(4, 8), 1.0, None, None, None))

    return {
        "lora_1": {
            "layer.0.attn1": adapter1,
            "layer.0.attn2": adapter1,
        },
        "lora_2": {
            "layer.0.attn1": adapter2,
            "layer.1.ff": adapter2,
        },
    }


class TestIntegrationWithFixtures:
    """Integration tests using fixtures."""

    def test_sample_tensors_are_valid(self, sample_lora_tensors):
        """Test that sample tensors pass validation."""
        assert is_lora_tensors(sample_lora_tensors)
        validate_lora_tensors(sample_lora_tensors)

    def test_sample_stack_is_valid(self, sample_lora_stack):
        """Test that sample stack passes validation."""
        assert is_lora_stack(sample_lora_stack)
        validate_lora_stack(sample_lora_stack)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
