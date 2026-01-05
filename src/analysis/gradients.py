"""
Gradient-based attribution for semantic LoRA analysis.

Implements Integrated Gradients and related methods for attributing
model outputs to specific LoRA weights.
"""

import logging
from typing import Dict, Tuple, Optional, Callable, Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class IntegratedGradientsAttributor:
    """
    Integrated Gradients attribution for LoRA weights.

    Computes importance scores for LoRA weights by integrating gradients
    along the path from a baseline (zero) to the actual LoRA weights.

    References:
        Sundararajan et al. "Axiomatic Attribution for Deep Networks" (2017)
        https://arxiv.org/abs/1703.01365
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        num_steps: int = 50,
    ):
        """
        Initialize attributor.

        Args:
            model: The model to analyze (with LoRA applied)
            device: Device for computation
            num_steps: Number of integration steps (default: 50)
        """
        self.model = model
        self.device = device
        self.num_steps = num_steps

    def attribute_lora_weights(
        self,
        lora_patches: Dict[str, Any],
        forward_fn: Callable,
        target_fn: Callable,
        baseline: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute Integrated Gradients attribution for LoRA weights.

        Args:
            lora_patches: LoRA patch dictionary {layer_key: {sublayer: tensor}}
            forward_fn: Function that runs forward pass and returns output
            target_fn: Function that converts output to scalar target value
            baseline: Baseline LoRA weights (default: zeros)

        Returns:
            Attribution scores for each LoRA weight tensor
            Format: {layer_key: attribution_tensor}

        Example:
            >>> def forward_fn():
            ...     return model(input_latents, timesteps, prompt_embeds)
            >>>
            >>> def target_fn(output):
            ...     # Target: final latent norm (as proxy for feature strength)
            ...     return output.norm()
            >>>
            >>> attributions = attributor.attribute_lora_weights(
            ...     lora_patches, forward_fn, target_fn
            ... )
        """
        # Create baseline (zeros) if not provided
        if baseline is None:
            baseline = self._create_zero_baseline(lora_patches)

        # Collect all LoRA parameters
        lora_params = self._collect_lora_params(lora_patches)

        # Storage for accumulated gradients
        accumulated_grads = {key: torch.zeros_like(param) for key, param in lora_params.items()}

        # Integration loop
        for step in range(self.num_steps + 1):
            alpha = step / self.num_steps

            # Interpolate between baseline and actual weights
            interpolated_weights = self._interpolate_weights(
                baseline, lora_patches, alpha
            )

            # Apply interpolated weights to model (temporarily)
            self._apply_weights_to_model(interpolated_weights)

            # Forward pass
            with torch.enable_grad():
                output = forward_fn()
                target = target_fn(output)

                # Backward pass to compute gradients
                gradients = torch.autograd.grad(
                    outputs=target,
                    inputs=lora_params.values(),
                    retain_graph=False,
                    create_graph=False,
                )

            # Accumulate gradients
            for (key, param), grad in zip(lora_params.items(), gradients):
                if grad is not None:
                    accumulated_grads[key] += grad

        # Average gradients across steps
        for key in accumulated_grads:
            accumulated_grads[key] /= self.num_steps

        # Multiply by (weight - baseline) to get attributions
        attributions = {}
        for key, param in lora_params.items():
            baseline_value = baseline.get(key, torch.zeros_like(param))
            delta = param - baseline_value
            attributions[key] = accumulated_grads[key] * delta

        return attributions

    def _create_zero_baseline(
        self, lora_patches: Dict[str, Any]
    ) -> Dict[str, torch.Tensor]:
        """Create baseline of zeros with same structure as lora_patches."""
        baseline = {}
        for layer_key, sublayers in lora_patches.items():
            if isinstance(sublayers, dict):
                for sublayer_key, tensor in sublayers.items():
                    if isinstance(tensor, torch.Tensor):
                        full_key = f"{layer_key}.{sublayer_key}"
                        baseline[full_key] = torch.zeros_like(tensor)
        return baseline

    def _collect_lora_params(
        self, lora_patches: Dict[str, Any]
    ) -> Dict[str, torch.Tensor]:
        """Collect all LoRA parameter tensors."""
        params = {}
        for layer_key, sublayers in lora_patches.items():
            if isinstance(sublayers, dict):
                for sublayer_key, tensor in sublayers.items():
                    if isinstance(tensor, torch.Tensor):
                        full_key = f"{layer_key}.{sublayer_key}"
                        # Make tensor require gradients
                        tensor_grad = tensor.detach().requires_grad_(True)
                        params[full_key] = tensor_grad
        return params

    def _interpolate_weights(
        self,
        baseline: Dict[str, torch.Tensor],
        lora_patches: Dict[str, Any],
        alpha: float,
    ) -> Dict[str, Any]:
        """Interpolate between baseline and actual weights."""
        interpolated = {}
        for layer_key, sublayers in lora_patches.items():
            if isinstance(sublayers, dict):
                interpolated[layer_key] = {}
                for sublayer_key, tensor in sublayers.items():
                    if isinstance(tensor, torch.Tensor):
                        full_key = f"{layer_key}.{sublayer_key}"
                        baseline_value = baseline.get(
                            full_key, torch.zeros_like(tensor)
                        )
                        interpolated[layer_key][sublayer_key] = (
                            baseline_value + alpha * (tensor - baseline_value)
                        )
                    else:
                        interpolated[layer_key][sublayer_key] = tensor
            else:
                interpolated[layer_key] = sublayers
        return interpolated

    def _apply_weights_to_model(self, weights: Dict[str, Any]) -> None:
        """
        Apply interpolated LoRA weights to model.

        Note: This is a placeholder. Actual implementation needs to
        temporarily patch the model's LoRA weights.
        """
        # TODO: Implement model patching
        # This requires ComfyUI-specific logic to temporarily apply LoRA weights
        pass


class SimpleGradientAttributor:
    """
    Simple gradient-based attribution (faster but less accurate than IG).

    Computes gradients of output w.r.t. LoRA weights at a single point.
    """

    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device

    def attribute_lora_weights(
        self,
        lora_patches: Dict[str, Any],
        forward_fn: Callable,
        target_fn: Callable,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute simple gradient attribution.

        Args:
            lora_patches: LoRA patch dictionary
            forward_fn: Function that runs forward pass
            target_fn: Function that converts output to scalar

        Returns:
            Attribution scores (gradient * weight)
        """
        # Collect parameters
        lora_params = {}
        for layer_key, sublayers in lora_patches.items():
            if isinstance(sublayers, dict):
                for sublayer_key, tensor in sublayers.items():
                    if isinstance(tensor, torch.Tensor):
                        full_key = f"{layer_key}.{sublayer_key}"
                        lora_params[full_key] = tensor.detach().requires_grad_(True)

        # Forward pass
        with torch.enable_grad():
            output = forward_fn()
            target = target_fn(output)

            # Compute gradients
            gradients = torch.autograd.grad(
                outputs=target,
                inputs=lora_params.values(),
                retain_graph=False,
                create_graph=False,
            )

        # Attribution = gradient * weight
        attributions = {}
        for (key, param), grad in zip(lora_params.items(), gradients):
            if grad is not None:
                attributions[key] = grad * param
            else:
                attributions[key] = torch.zeros_like(param)

        return attributions


def compute_weight_importance(
    attributions: Dict[str, torch.Tensor],
    aggregation: str = "abs_sum",
) -> Dict[str, float]:
    """
    Aggregate attribution tensors to scalar importance scores.

    Args:
        attributions: Attribution tensors per layer
        aggregation: Aggregation method:
            - 'abs_sum': Sum of absolute values
            - 'l2': L2 norm
            - 'max': Maximum absolute value

    Returns:
        Scalar importance score per layer
    """
    importance = {}

    for key, attr_tensor in attributions.items():
        if aggregation == "abs_sum":
            score = attr_tensor.abs().sum().item()
        elif aggregation == "l2":
            score = attr_tensor.norm(p=2).item()
        elif aggregation == "max":
            score = attr_tensor.abs().max().item()
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation}")

        importance[key] = score

    return importance


def normalize_importance_scores(
    importance_map: Dict[str, Dict[str, float]]
) -> Dict[str, Dict[str, float]]:
    """
    Normalize importance scores across features for each layer.

    Args:
        importance_map: {feature: {layer_key: importance_score}}

    Returns:
        Normalized importance map (scores sum to 1.0 per layer)
    """
    # Collect all layer keys
    all_layers = set()
    for feature_scores in importance_map.values():
        all_layers.update(feature_scores.keys())

    # Normalize per layer
    normalized = {feature: {} for feature in importance_map}

    for layer_key in all_layers:
        # Get scores for this layer across all features
        layer_scores = {
            feature: importance_map[feature].get(layer_key, 0.0)
            for feature in importance_map
        }

        # Compute total
        total = sum(layer_scores.values())

        # Normalize
        if total > 0:
            for feature, score in layer_scores.items():
                normalized[feature][layer_key] = score / total
        else:
            # If all zeros, distribute equally
            for feature in layer_scores:
                normalized[feature][layer_key] = 1.0 / len(layer_scores)

    return normalized
