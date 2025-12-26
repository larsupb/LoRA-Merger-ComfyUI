"""
Validators for LoRA merge operations.

Provides validation classes and functions for ensuring input data is correct
before attempting merge operations.
"""

import logging
from typing import List, Dict, Optional, Tuple
import torch

from ..types import (
    LORA_STACK,
    LORA_WEIGHTS,
    LORA_TENSORS,
    LORA_TENSORS_BY_LAYER,
    ValidationResult,
    ValidationError,
)


class LoRAStackValidator:
    """
    Validator for LoRA stacks.

    Validates LoRA stacks have compatible structures for merging.
    """

    @staticmethod
    def validate(
        lora_stack: LORA_STACK,
        min_loras: int = 2
    ) -> ValidationResult:
        """
        Validate a LoRA stack for merge operations.

        Args:
            lora_stack: Stack to validate
            min_loras: Minimum number of LoRAs required

        Returns:
            ValidationResult with errors and warnings
        """
        errors = []
        warnings = []

        # Check not empty
        if not lora_stack:
            errors.append({
                "code": "EMPTY_STACK",
                "message": "LoRA stack is empty",
                "location": None
            })
            return {"valid": False, "errors": errors, "warnings": warnings}

        # Check minimum count
        if len(lora_stack) < min_loras:
            errors.append({
                "code": "INSUFFICIENT_LORAS",
                "message": f"At least {min_loras} LoRAs required, got {len(lora_stack)}",
                "location": None
            })

        # Check each LoRA has layers
        for lora_name, lora_dict in lora_stack.items():
            if not lora_dict:
                errors.append({
                    "code": "EMPTY_LORA",
                    "message": f"LoRA '{lora_name}' has no layers",
                    "location": lora_name
                })

        # Check for common keys across LoRAs
        all_keys = [set(lora_dict.keys()) for lora_dict in lora_stack.values()]
        common_keys = set.intersection(*all_keys) if all_keys else set()

        if not common_keys:
            warnings.append(
                "LoRAs have no common layer keys. "
                "Merge may produce unexpected results."
            )

        # Check key overlap
        total_keys = set.union(*all_keys) if all_keys else set()
        overlap_ratio = len(common_keys) / len(total_keys) if total_keys else 0

        if overlap_ratio < 0.5:
            warnings.append(
                f"Low key overlap ({overlap_ratio*100:.1f}%). "
                f"LoRAs may be from different architectures."
            )

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }


class TensorShapeValidator:
    """
    Validator for tensor shape compatibility.

    Ensures tensors can be merged together (compatible dimensions).
    """

    @staticmethod
    def validate_shapes_compatible(
        tensors_dict: Dict[str, LORA_TENSORS]
    ) -> ValidationResult:
        """
        Validate that tensor shapes are compatible for merging.

        Args:
            tensors_dict: Dictionary mapping LoRA names to (up, down, alpha) tuples

        Returns:
            ValidationResult with errors and warnings
        """
        errors = []
        warnings = []

        if not tensors_dict:
            errors.append({
                "code": "NO_TENSORS",
                "message": "No tensors provided",
                "location": None
            })
            return {"valid": False, "errors": errors, "warnings": warnings}

        # Get first tensor as reference
        first_name = next(iter(tensors_dict.keys()))
        first_up, first_down, first_alpha = tensors_dict[first_name]

        ref_up_shape = first_up.shape
        ref_down_shape = first_down.shape
        ref_rank = first_up.shape[1] if len(first_up.shape) >= 2 else None

        # Check all tensors against reference
        for lora_name, (up, down, alpha) in tensors_dict.items():
            if lora_name == first_name:
                continue

            # Check up tensor shape
            if up.shape != ref_up_shape:
                if len(up.shape) >= 2 and up.shape[1] != ref_rank:
                    errors.append({
                        "code": "RANK_MISMATCH",
                        "message": (
                            f"LoRA '{lora_name}' has different rank: "
                            f"{up.shape[1]} vs {ref_rank}"
                        ),
                        "location": lora_name
                    })
                elif up.shape != ref_up_shape:
                    warnings.append(
                        f"LoRA '{lora_name}' up tensor shape {up.shape} "
                        f"differs from reference {ref_up_shape}"
                    )

            # Check down tensor shape
            if down.shape != ref_down_shape:
                warnings.append(
                    f"LoRA '{lora_name}' down tensor shape {down.shape} "
                    f"differs from reference {ref_down_shape}"
                )

            # Validate up/down dimensions are compatible
            if len(up.shape) >= 2 and len(down.shape) >= 2:
                up_rank = up.shape[1] if len(up.shape) == 2 else up.shape[1]
                down_rank = down.shape[0]

                if up_rank != down_rank:
                    errors.append({
                        "code": "INCOMPATIBLE_DIMENSIONS",
                        "message": (
                            f"LoRA '{lora_name}' has incompatible up/down dimensions: "
                            f"up rank {up_rank} != down rank {down_rank}"
                        ),
                        "location": lora_name
                    })

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }

    @staticmethod
    def validate_lora_tensor_structure(
        up: torch.Tensor,
        down: torch.Tensor,
        alpha: float,
        lora_name: str = "unknown"
    ) -> ValidationResult:
        """
        Validate individual LoRA tensor structure.

        Args:
            up: Up-projection tensor
            down: Down-projection tensor
            alpha: Alpha scaling value
            lora_name: Name for error messages

        Returns:
            ValidationResult
        """
        errors = []
        warnings = []

        # Check dimensions
        if len(up.shape) not in (2, 4):
            errors.append({
                "code": "INVALID_UP_SHAPE",
                "message": f"Up tensor must be 2D or 4D, got shape {up.shape}",
                "location": lora_name
            })

        if len(down.shape) not in (2, 4):
            errors.append({
                "code": "INVALID_DOWN_SHAPE",
                "message": f"Down tensor must be 2D or 4D, got shape {down.shape}",
                "location": lora_name
            })

        # Check alpha is valid
        if not isinstance(alpha, (int, float, torch.Tensor)):
            errors.append({
                "code": "INVALID_ALPHA_TYPE",
                "message": f"Alpha must be numeric, got {type(alpha)}",
                "location": lora_name
            })
        elif alpha < 0:
            warnings.append(f"Alpha is negative ({alpha}), which is unusual")

        # Check for proper up/down dimensions
        if len(up.shape) == 2 and len(down.shape) == 2:
            # Linear layer
            up_is_tall = up.shape[0] > up.shape[1]
            down_is_wide = down.shape[1] > down.shape[0]

            if not (up_is_tall and down_is_wide):
                warnings.append(
                    f"Unusual LoRA dimensions: up {up.shape}, down {down.shape}. "
                    "Expected up to be tall and down to be wide."
                )

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }


class MergeParameterValidator:
    """
    Validator for merge method parameters.

    Validates method-specific parameters are within valid ranges.
    """

    @staticmethod
    def validate_weights(
        weights: LORA_WEIGHTS,
        lora_names: List[str]
    ) -> ValidationResult:
        """
        Validate merge weights.

        Args:
            weights: Weight dictionary
            lora_names: Expected LoRA names

        Returns:
            ValidationResult
        """
        errors = []
        warnings = []

        # Check all LoRAs have weights
        for name in lora_names:
            if name not in weights:
                errors.append({
                    "code": "MISSING_WEIGHT",
                    "message": f"No weight specified for LoRA '{name}'",
                    "location": name
                })
                continue

            weight_info = weights[name]

            # Check strength_model exists
            if "strength_model" not in weight_info:
                errors.append({
                    "code": "MISSING_STRENGTH_MODEL",
                    "message": f"strength_model not specified for '{name}'",
                    "location": name
                })

            # Check strength values are reasonable
            strength_model = weight_info.get("strength_model", 0)
            if strength_model < 0 or strength_model > 2:
                warnings.append(
                    f"Unusual strength_model for '{name}': {strength_model}. "
                    "Typical range is 0.0 to 1.0"
                )

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }

    @staticmethod
    def validate_method_args(
        method_name: str,
        method_args: Dict
    ) -> ValidationResult:
        """
        Validate method-specific arguments.

        Args:
            method_name: Name of merge method
            method_args: Method arguments

        Returns:
            ValidationResult
        """
        errors = []
        warnings = []

        # Validate common parameters
        if "lambda_" in method_args:
            lambda_val = method_args["lambda_"]
            if lambda_val < 0 or lambda_val > 2:
                warnings.append(
                    f"Unusual lambda value: {lambda_val}. "
                    "Typical range is 0.0 to 1.0"
                )

        # Method-specific validation
        if method_name in ["slerp", "nuslerp"]:
            if "t" in method_args:
                t = method_args["t"]
                if t < 0 or t > 1:
                    errors.append({
                        "code": "INVALID_PARAMETER",
                        "message": f"Parameter 't' must be in [0, 1], got {t}",
                        "location": "t"
                    })

        if method_name == "sce":
            if "select_topk" in method_args:
                topk = method_args["select_topk"]
                if topk <= 0 or topk > 1:
                    errors.append({
                        "code": "INVALID_PARAMETER",
                        "message": f"select_topk must be in (0, 1], got {topk}",
                        "location": "select_topk"
                    })

        # DELLA-specific validation
        if method_name == "della":
            density = method_args.get("density", 1.0)
            epsilon = method_args.get("epsilon", 0.1)

            # Validate epsilon constraint: density +/- epsilon must be in (0, 1)
            if density - epsilon <= 0:
                errors.append({
                    "code": "INVALID_EPSILON",
                    "message": (
                        f"Epsilon too large: density - epsilon = {density} - {epsilon} = {density - epsilon} "
                        f"must be > 0. Try reducing epsilon or increasing density."
                    ),
                    "location": "epsilon"
                })

            if density + epsilon >= 1:
                errors.append({
                    "code": "INVALID_EPSILON",
                    "message": (
                        f"Epsilon too large: density + epsilon = {density} + {epsilon} = {density + epsilon} "
                        f"must be < 1. Try reducing epsilon or reducing density."
                    ),
                    "location": "epsilon"
                })

        # Density validation for methods that use it
        if method_name in ["ties", "dare", "della", "breadcrumbs"]:
            if "density" in method_args:
                density = method_args["density"]
                if density <= 0 or density > 1:
                    errors.append({
                        "code": "INVALID_PARAMETER",
                        "message": f"density must be in (0, 1], got {density}",
                        "location": "density"
                    })

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }


# Convenience functions

def validate_lora_stack_for_merge(
    lora_stack: LORA_STACK,
    weights: LORA_WEIGHTS,
    method_name: str,
    method_args: Dict
) -> ValidationResult:
    """
    Comprehensive validation for merge operation.

    Validates all aspects: stack structure, weights, and method parameters.

    Args:
        lora_stack: LoRA stack to merge
        weights: Merge weights
        method_name: Merge method name
        method_args: Method arguments

    Returns:
        Combined ValidationResult
    """
    all_errors = []
    all_warnings = []

    # Validate stack
    stack_result = LoRAStackValidator.validate(lora_stack)
    all_errors.extend(stack_result["errors"])
    all_warnings.extend(stack_result["warnings"])

    # Validate weights
    lora_names = list(lora_stack.keys())
    weight_result = MergeParameterValidator.validate_weights(weights, lora_names)
    all_errors.extend(weight_result["errors"])
    all_warnings.extend(weight_result["warnings"])

    # Validate method args
    method_result = MergeParameterValidator.validate_method_args(method_name, method_args)
    all_errors.extend(method_result["errors"])
    all_warnings.extend(method_result["warnings"])

    # Log warnings
    for warning in all_warnings:
        logging.warning(f"Validation warning: {warning}")

    return {
        "valid": len(all_errors) == 0,
        "errors": all_errors,
        "warnings": all_warnings
    }


def validate_tensor_shapes_compatible(
    tensors_by_layer: LORA_TENSORS_BY_LAYER
) -> ValidationResult:
    """
    Validate tensor shapes are compatible across all layers.

    Args:
        tensors_by_layer: Dictionary of layer -> (LoRA name -> tensors)

    Returns:
        Combined ValidationResult
    """
    all_errors = []
    all_warnings = []

    for layer_key, tensors_dict in tensors_by_layer.items():
        result = TensorShapeValidator.validate_shapes_compatible(tensors_dict)

        # Add layer context to errors
        for error in result["errors"]:
            error["location"] = f"{layer_key}.{error.get('location', '')}"
            all_errors.append(error)

        all_warnings.extend(result["warnings"])

    return {
        "valid": len(all_errors) == 0,
        "errors": all_errors,
        "warnings": all_warnings
    }
