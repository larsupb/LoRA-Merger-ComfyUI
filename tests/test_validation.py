"""
Unit tests for validation module.

Tests validators for LoRA stacks, tensor shapes, and merge parameters.
"""

import pytest
import torch
import sys
from pathlib import Path
from unittest.mock import Mock

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from validation import (
    LoRAStackValidator,
    TensorShapeValidator,
    MergeParameterValidator,
    validate_lora_stack_for_merge,
    validate_tensor_shapes_compatible,
)
from comfy.weight_adapter import LoRAAdapter


class TestLoRAStackValidator:
    """Tests for LoRA stack validation."""

    def test_valid_stack(self):
        """Test validation of valid LoRA stack."""
        stack = {
            "lora1": {"layer1": Mock(), "layer2": Mock()},
            "lora2": {"layer1": Mock(), "layer2": Mock()},
        }

        result = LoRAStackValidator.validate(stack)

        assert result["valid"] is True
        assert len(result["errors"]) == 0

    def test_empty_stack(self):
        """Test that empty stack is invalid."""
        stack = {}

        result = LoRAStackValidator.validate(stack)

        assert result["valid"] is False
        assert any(e["code"] == "EMPTY_STACK" for e in result["errors"])

    def test_insufficient_loras(self):
        """Test that stack with too few LoRAs is invalid."""
        stack = {
            "lora1": {"layer1": Mock()},
        }

        result = LoRAStackValidator.validate(stack, min_loras=2)

        assert result["valid"] is False
        assert any(e["code"] == "INSUFFICIENT_LORAS" for e in result["errors"])

    def test_empty_lora_in_stack(self):
        """Test detection of LoRA with no layers."""
        stack = {
            "lora1": {"layer1": Mock()},
            "lora2": {},  # Empty LoRA
        }

        result = LoRAStackValidator.validate(stack)

        assert result["valid"] is False
        assert any(e["code"] == "EMPTY_LORA" for e in result["errors"])

    def test_no_common_keys_warning(self):
        """Test warning when LoRAs have no common keys."""
        stack = {
            "lora1": {"layer1": Mock()},
            "lora2": {"layer2": Mock()},  # Different keys
        }

        result = LoRAStackValidator.validate(stack)

        # Should be valid but have warning
        assert result["valid"] is True
        assert len(result["warnings"]) > 0
        assert any("no common layer keys" in w for w in result["warnings"])

    def test_low_key_overlap_warning(self):
        """Test warning for low key overlap."""
        stack = {
            "lora1": {"layer1": Mock(), "layer2": Mock(), "layer3": Mock()},
            "lora2": {"layer1": Mock(), "layer4": Mock(), "layer5": Mock()},
        }

        result = LoRAStackValidator.validate(stack)

        # Only 1 common key out of 5 total (20% overlap)
        assert result["valid"] is True
        assert any("Low key overlap" in w for w in result["warnings"])


class TestTensorShapeValidator:
    """Tests for tensor shape validation."""

    def test_compatible_shapes(self):
        """Test validation of compatible tensor shapes."""
        tensors = {
            "lora1": (torch.randn(10, 5), torch.randn(5, 10), 1.0),
            "lora2": (torch.randn(10, 5), torch.randn(5, 10), 1.0),
        }

        result = TensorShapeValidator.validate_shapes_compatible(tensors)

        assert result["valid"] is True
        assert len(result["errors"]) == 0

    def test_no_tensors(self):
        """Test that empty tensors dict is invalid."""
        tensors = {}

        result = TensorShapeValidator.validate_shapes_compatible(tensors)

        assert result["valid"] is False
        assert any(e["code"] == "NO_TENSORS" for e in result["errors"])

    def test_rank_mismatch(self):
        """Test detection of rank mismatch."""
        tensors = {
            "lora1": (torch.randn(10, 5), torch.randn(5, 10), 1.0),
            "lora2": (torch.randn(10, 8), torch.randn(8, 10), 1.0),  # Different rank
        }

        result = TensorShapeValidator.validate_shapes_compatible(tensors)

        assert result["valid"] is False
        assert any(e["code"] == "RANK_MISMATCH" for e in result["errors"])

    def test_incompatible_up_down_dimensions(self):
        """Test detection of incompatible up/down dimensions."""
        tensors = {
            "lora1": (torch.randn(10, 5), torch.randn(8, 10), 1.0),  # 5 != 8
        }

        result = TensorShapeValidator.validate_shapes_compatible(tensors)

        assert result["valid"] is False
        assert any(e["code"] == "INCOMPATIBLE_DIMENSIONS" for e in result["errors"])

    def test_validate_individual_lora_structure_valid(self):
        """Test validation of valid individual LoRA tensor structure."""
        up = torch.randn(10, 5)
        down = torch.randn(5, 10)
        alpha = 1.0

        result = TensorShapeValidator.validate_lora_tensor_structure(
            up, down, alpha, "test_lora"
        )

        assert result["valid"] is True

    def test_validate_individual_lora_invalid_up_shape(self):
        """Test detection of invalid up tensor shape."""
        up = torch.randn(10)  # 1D tensor (invalid)
        down = torch.randn(5, 10)
        alpha = 1.0

        result = TensorShapeValidator.validate_lora_tensor_structure(
            up, down, alpha, "test_lora"
        )

        assert result["valid"] is False
        assert any(e["code"] == "INVALID_UP_SHAPE" for e in result["errors"])

    def test_validate_individual_lora_invalid_down_shape(self):
        """Test detection of invalid down tensor shape."""
        up = torch.randn(10, 5)
        down = torch.randn(10)  # 1D tensor (invalid)
        alpha = 1.0

        result = TensorShapeValidator.validate_lora_tensor_structure(
            up, down, alpha, "test_lora"
        )

        assert result["valid"] is False
        assert any(e["code"] == "INVALID_DOWN_SHAPE" for e in result["errors"])

    def test_validate_individual_lora_invalid_alpha_type(self):
        """Test detection of invalid alpha type."""
        up = torch.randn(10, 5)
        down = torch.randn(5, 10)
        alpha = "not a number"  # Invalid type

        result = TensorShapeValidator.validate_lora_tensor_structure(
            up, down, alpha, "test_lora"
        )

        assert result["valid"] is False
        assert any(e["code"] == "INVALID_ALPHA_TYPE" for e in result["errors"])

    def test_unusual_dimensions_warning(self):
        """Test warning for unusual tensor dimensions."""
        up = torch.randn(5, 10)  # Wide instead of tall
        down = torch.randn(10, 5)  # Tall instead of wide
        alpha = 1.0

        result = TensorShapeValidator.validate_lora_tensor_structure(
            up, down, alpha
        )

        # Should be valid but have warning
        assert result["valid"] is True
        assert any("Unusual LoRA dimensions" in w for w in result["warnings"])

    def test_negative_alpha_warning(self):
        """Test warning for negative alpha."""
        up = torch.randn(10, 5)
        down = torch.randn(5, 10)
        alpha = -1.0  # Negative

        result = TensorShapeValidator.validate_lora_tensor_structure(
            up, down, alpha
        )

        # Should be valid but have warning
        assert result["valid"] is True
        assert any("negative" in w.lower() for w in result["warnings"])


class TestMergeParameterValidator:
    """Tests for merge parameter validation."""

    def test_validate_weights_valid(self):
        """Test validation of valid weights."""
        weights = {
            "lora1": {"strength_model": 0.5, "strength_clip": 0.5},
            "lora2": {"strength_model": 0.8, "strength_clip": 0.8},
        }
        lora_names = ["lora1", "lora2"]

        result = MergeParameterValidator.validate_weights(weights, lora_names)

        assert result["valid"] is True
        assert len(result["errors"]) == 0

    def test_missing_weight(self):
        """Test detection of missing weight."""
        weights = {
            "lora1": {"strength_model": 0.5},
            # lora2 missing
        }
        lora_names = ["lora1", "lora2"]

        result = MergeParameterValidator.validate_weights(weights, lora_names)

        assert result["valid"] is False
        assert any(e["code"] == "MISSING_WEIGHT" for e in result["errors"])

    def test_missing_strength_model(self):
        """Test detection of missing strength_model."""
        weights = {
            "lora1": {"strength_clip": 0.5},  # Missing strength_model
        }
        lora_names = ["lora1"]

        result = MergeParameterValidator.validate_weights(weights, lora_names)

        assert result["valid"] is False
        assert any(e["code"] == "MISSING_STRENGTH_MODEL" for e in result["errors"])

    def test_unusual_strength_warning(self):
        """Test warning for unusual strength value."""
        weights = {
            "lora1": {"strength_model": 5.0},  # Very high
        }
        lora_names = ["lora1"]

        result = MergeParameterValidator.validate_weights(weights, lora_names)

        # Should be valid but have warning
        assert result["valid"] is True
        assert any("Unusual strength_model" in w for w in result["warnings"])

    def test_validate_method_args_valid(self):
        """Test validation of valid method arguments."""
        method_args = {
            "normalize": True,
            "lambda_": 0.8,
        }

        result = MergeParameterValidator.validate_method_args("linear", method_args)

        assert result["valid"] is True

    def test_slerp_invalid_t_parameter(self):
        """Test detection of invalid t parameter for SLERP."""
        method_args = {
            "t": 1.5,  # Out of range [0, 1]
        }

        result = MergeParameterValidator.validate_method_args("slerp", method_args)

        assert result["valid"] is False
        assert any(e["code"] == "INVALID_PARAMETER" for e in result["errors"])
        assert any(e["location"] == "t" for e in result["errors"])

    def test_sce_invalid_topk_parameter(self):
        """Test detection of invalid select_topk for SCE."""
        method_args = {
            "select_topk": 1.5,  # Out of range (0, 1]
        }

        result = MergeParameterValidator.validate_method_args("sce", method_args)

        assert result["valid"] is False
        assert any(e["code"] == "INVALID_PARAMETER" for e in result["errors"])

    def test_unusual_lambda_warning(self):
        """Test warning for unusual lambda value."""
        method_args = {
            "lambda_": 3.0,  # Unusually high
        }

        result = MergeParameterValidator.validate_method_args("linear", method_args)

        # Should be valid but have warning
        assert result["valid"] is True
        assert any("lambda" in w.lower() for w in result["warnings"])


class TestConvenienceFunctions:
    """Tests for convenience validation functions."""

    def test_validate_lora_stack_for_merge_valid(self):
        """Test comprehensive validation with valid inputs."""
        stack = {
            "lora1": {"layer1": Mock(), "layer2": Mock()},
            "lora2": {"layer1": Mock(), "layer2": Mock()},
        }
        weights = {
            "lora1": {"strength_model": 0.5},
            "lora2": {"strength_model": 0.8},
        }
        method_name = "linear"
        method_args = {"normalize": True}

        result = validate_lora_stack_for_merge(
            stack, weights, method_name, method_args
        )

        assert result["valid"] is True

    def test_validate_lora_stack_for_merge_multiple_errors(self):
        """Test that multiple validation errors are accumulated."""
        stack = {
            "lora1": {},  # Empty LoRA
        }
        weights = {
            # Missing weight for lora1
        }
        method_name = "slerp"
        method_args = {
            "t": 2.0,  # Invalid t
        }

        result = validate_lora_stack_for_merge(
            stack, weights, method_name, method_args
        )

        assert result["valid"] is False
        # Should have multiple errors from different validators
        assert len(result["errors"]) >= 3  # Empty LoRA, missing weight, invalid t

    def test_validate_tensor_shapes_compatible_valid(self):
        """Test validation of compatible tensors across layers."""
        tensors_by_layer = {
            "layer1": {
                "lora1": (torch.randn(10, 5), torch.randn(5, 10), 1.0),
                "lora2": (torch.randn(10, 5), torch.randn(5, 10), 1.0),
            },
            "layer2": {
                "lora1": (torch.randn(20, 8), torch.randn(8, 20), 1.0),
                "lora2": (torch.randn(20, 8), torch.randn(8, 20), 1.0),
            },
        }

        result = validate_tensor_shapes_compatible(tensors_by_layer)

        assert result["valid"] is True

    def test_validate_tensor_shapes_compatible_with_errors(self):
        """Test detection of incompatible shapes across layers."""
        tensors_by_layer = {
            "layer1": {
                "lora1": (torch.randn(10, 5), torch.randn(5, 10), 1.0),
                "lora2": (torch.randn(10, 8), torch.randn(8, 10), 1.0),  # Rank mismatch
            },
        }

        result = validate_tensor_shapes_compatible(tensors_by_layer)

        assert result["valid"] is False
        # Error should include layer context
        assert any("layer1" in e.get("location", "") for e in result["errors"])


# Fixtures

@pytest.fixture
def valid_lora_stack():
    """Fixture providing valid LoRA stack."""
    return {
        "lora_1": {
            "layer.0.attn1": Mock(),
            "layer.0.attn2": Mock(),
        },
        "lora_2": {
            "layer.0.attn1": Mock(),
            "layer.0.attn2": Mock(),
        },
    }


@pytest.fixture
def valid_weights():
    """Fixture providing valid weights."""
    return {
        "lora_1": {"strength_model": 0.6, "strength_clip": 0.6},
        "lora_2": {"strength_model": 0.4, "strength_clip": 0.4},
    }


@pytest.fixture
def compatible_tensors():
    """Fixture providing compatible tensors."""
    return {
        "lora_1": (torch.randn(100, 50), torch.randn(50, 100), 1.0),
        "lora_2": (torch.randn(100, 50), torch.randn(50, 100), 1.0),
    }


class TestIntegrationWithFixtures:
    """Integration tests using fixtures."""

    def test_full_validation_with_valid_inputs(
        self, valid_lora_stack, valid_weights
    ):
        """Test full validation pipeline with valid inputs."""
        result = validate_lora_stack_for_merge(
            valid_lora_stack,
            valid_weights,
            "linear",
            {"normalize": True}
        )

        assert result["valid"] is True
        assert len(result["errors"]) == 0

    def test_shape_validation_with_compatible_tensors(self, compatible_tensors):
        """Test shape validation with compatible tensors."""
        result = TensorShapeValidator.validate_shapes_compatible(
            compatible_tensors
        )

        assert result["valid"] is True
        assert len(result["errors"]) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
