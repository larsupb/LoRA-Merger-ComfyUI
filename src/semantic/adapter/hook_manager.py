"""
Forward hook manager for injecting adapters into UNet.

This module manages the injection of semantic adapters into the UNet forward pass,
allowing runtime composition of multiple LoRAs based on semantic importance.
"""

import logging
from typing import Dict, List, Callable, Optional, Any
import torch
import torch.nn as nn

from .registry import AdapterRegistry
from ...types import LORA_STACK

logger = logging.getLogger(__name__)


class ForwardHookManager:
    """
    Manages forward hooks for injecting adapters into UNet.

    Strategy:
    1. Register hooks on all LoRA-compatible layers
    2. During forward pass, intercept layer computation
    3. Compute LoRA outputs separately for each source
    4. Route through appropriate adapter with semantic guidance
    5. Inject adapted output back into computation

    Note: This is a foundational implementation. Full integration with
    ComfyUI's ModelPatcher requires additional work to separate LoRA
    computations that are normally applied together.
    """

    def __init__(
        self,
        model: nn.Module,
        adapter_registry: AdapterRegistry,
        lora_stack: LORA_STACK,
        device: torch.device,
        dtype: torch.dtype,
    ):
        """
        Initialize hook manager.

        Args:
            model: Base model (UNet or transformer)
            adapter_registry: Registry with trained adapters
            lora_stack: LoRA stack with patches
            device: Compute device
            dtype: Compute dtype
        """
        self.model = model
        self.adapter_registry = adapter_registry
        self.lora_stack = lora_stack
        self.device = device
        self.dtype = dtype

        self.hooks: List[torch.utils.hooks.RemovableHandle] = []
        self.semantic_vector: Optional[torch.Tensor] = None
        self.adapter_outputs: Dict[str, torch.Tensor] = {}  # For debugging

        logger.info(
            f"Initialized ForwardHookManager with {len(adapter_registry)} adapters"
        )

    def register_hooks(self):
        """
        Register forward hooks on all relevant model layers.

        This finds layers that have corresponding LoRAs in the stack
        and registers hooks to intercept their forward pass.
        """
        if self.hooks:
            logger.warning("Hooks already registered, removing old hooks first")
            self.remove_hooks()

        # Get all layer keys that have adapters
        adapter_keys = set(self.adapter_registry.get_all_adapters().keys())

        # Try to find corresponding modules in the model
        registered_count = 0
        for name, module in self.model.named_modules():
            # Convert module name to layer key format
            # This is model-specific and may need adjustment
            layer_key = self._module_name_to_layer_key(name)

            if layer_key in adapter_keys:
                # Register hook on this module
                hook = module.register_forward_hook(
                    self._create_adapter_hook(layer_key)
                )
                self.hooks.append(hook)
                registered_count += 1

        logger.info(f"Registered {registered_count} forward hooks")

        if registered_count == 0:
            logger.warning(
                "No hooks were registered! This may indicate a mismatch between "
                "layer key format and model module names. Adapter injection will not work."
            )

    def _module_name_to_layer_key(self, module_name: str) -> str:
        """
        Convert PyTorch module name to LoRA layer key format.

        This is architecture-specific. Different models use different naming:
        - SD: "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q"
        - DiT: "transformer_blocks.0.attn.to_q"
        - ComfyUI: "diffusion_model.input_blocks.1.1.transformer_blocks.0.attn1.to_q"

        Args:
            module_name: PyTorch module name (e.g., "down_blocks.0.attentions.0")

        Returns:
            LoRA layer key (e.g., "lora_unet_down_blocks_0_attentions_0")
        """
        # For now, return as-is and log for debugging
        # This will need to be customized based on actual model structure
        return module_name

    def _is_lora_layer(self, layer_key: str) -> bool:
        """
        Check if layer has LoRA adapters in the stack.

        Args:
            layer_key: Layer identifier

        Returns:
            True if any LoRA in stack has this layer
        """
        for lora_entry in self.lora_stack.values():
            if layer_key in lora_entry["patches"]:
                return True
        return False

    def _create_adapter_hook(self, layer_key: str) -> Callable:
        """
        Create forward hook function for a specific layer.

        The hook:
        1. Lets the original forward pass complete
        2. Computes LoRA outputs separately for each source
        3. Routes through the adapter
        4. Adds adapted output to the result

        Args:
            layer_key: Layer identifier

        Returns:
            Hook function compatible with register_forward_hook
        """
        def hook_fn(
            module: nn.Module,
            input: tuple,
            output: torch.Tensor
        ) -> torch.Tensor:
            """
            Forward hook that injects adapter output.

            Args:
                module: The module being hooked
                input: Tuple of input tensors to the module
                output: Original output from the module

            Returns:
                Modified output with adapter applied
            """
            # Skip if semantic vector not set
            if self.semantic_vector is None:
                logger.warning(
                    f"Semantic vector not set for layer {layer_key}, skipping adapter"
                )
                return output

            # Get adapter for this layer
            try:
                adapter = self.adapter_registry.get_adapter(layer_key)
            except KeyError:
                logger.warning(f"No adapter found for layer {layer_key}")
                return output

            # Get adapter type
            adapter_type = self.adapter_registry.adapter_types[layer_key]

            try:
                if adapter_type == "rank_space":
                    # Rank-space adapter: need down-projection outputs
                    adapted_output = self._apply_rank_space_adapter(
                        layer_key, adapter, input[0], output
                    )
                else:
                    # Output-space adapter: need full LoRA outputs
                    adapted_output = self._apply_output_space_adapter(
                        layer_key, adapter, input[0], output
                    )

                # Store for debugging
                self.adapter_outputs[layer_key] = adapted_output

                # Add to original output
                return output + adapted_output

            except Exception as e:
                logger.error(f"Error applying adapter to {layer_key}: {e}")
                # Return original output on error
                return output

        return hook_fn

    def _apply_rank_space_adapter(
        self,
        layer_key: str,
        adapter,
        input_tensor: torch.Tensor,
        original_output: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply rank-space adapter to a layer.

        Computes down-projections (A @ x) for all LoRAs,
        routes through adapter, applies shared up-projection.

        Args:
            layer_key: Layer identifier
            adapter: RankSpaceAdapter instance
            input_tensor: Input to the layer
            original_output: Original layer output (unused)

        Returns:
            Adapted output tensor
        """
        # Compute down-projection outputs for each LoRA
        lora_down_outputs = {}
        b_shared = None

        for lora_name, lora_entry in self.lora_stack.items():
            if layer_key not in lora_entry["patches"]:
                continue

            up, down, alpha = lora_entry["patches"][layer_key]

            # Compute down projection: A @ x
            # Handle both 2D and 3D inputs
            if input_tensor.ndim == 2:
                # [batch, in_dim]
                down_output = torch.matmul(input_tensor, down.t())
            elif input_tensor.ndim == 3:
                # [batch, seq_len, in_dim]
                down_output = torch.matmul(input_tensor, down.t())
            else:
                logger.warning(
                    f"Unexpected input shape for rank-space adapter: {input_tensor.shape}"
                )
                continue

            lora_down_outputs[lora_name] = down_output

            # Use first LoRA's up matrix as shared (can be customized)
            if b_shared is None:
                b_shared = up

        if not lora_down_outputs:
            logger.warning(f"No LoRA outputs computed for {layer_key}")
            return torch.zeros_like(original_output)

        if b_shared is None:
            logger.warning(f"No shared up-projection found for {layer_key}")
            return torch.zeros_like(original_output)

        # Route through adapter
        adapted_output = adapter(lora_down_outputs, b_shared, self.semantic_vector)

        return adapted_output

    def _apply_output_space_adapter(
        self,
        layer_key: str,
        adapter,
        input_tensor: torch.Tensor,
        original_output: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply output-space adapter to a layer.

        Computes full LoRA outputs (B @ A @ x) for all sources,
        routes through adapter.

        Args:
            layer_key: Layer identifier
            adapter: OutputSpaceAdapter instance
            input_tensor: Input to the layer
            original_output: Original layer output (for shape reference)

        Returns:
            Adapted output tensor
        """
        # Compute full LoRA outputs for each source
        lora_deltas = {}

        for lora_name, lora_entry in self.lora_stack.items():
            if layer_key not in lora_entry["patches"]:
                continue

            up, down, alpha = lora_entry["patches"][layer_key]

            # Compute full LoRA: B @ (A @ x)
            # down: [rank, in_dim]
            # up: [out_dim, rank]
            # x: [batch, ..., in_dim]

            # Down projection
            if input_tensor.ndim == 2:
                # [batch, in_dim] -> [batch, rank]
                down_output = torch.matmul(input_tensor, down.t())
            elif input_tensor.ndim == 3:
                # [batch, seq_len, in_dim] -> [batch, seq_len, rank]
                down_output = torch.matmul(input_tensor, down.t())
            elif input_tensor.ndim == 4:
                # Conv2d: [batch, in_channels, H, W]
                # down: [rank, in_channels]
                # Need to permute to channels-last, matmul, permute back
                x_perm = input_tensor.permute(0, 2, 3, 1)  # [batch, H, W, in_channels]
                down_output = torch.matmul(x_perm, down.t())  # [batch, H, W, rank]
                down_output = down_output.permute(0, 3, 1, 2)  # [batch, rank, H, W]
            else:
                logger.warning(
                    f"Unexpected input shape for output-space adapter: {input_tensor.shape}"
                )
                continue

            # Up projection
            if down_output.ndim == 2:
                # [batch, rank] -> [batch, out_dim]
                full_output = torch.matmul(down_output, up.t())
            elif down_output.ndim == 3:
                # [batch, seq_len, rank] -> [batch, seq_len, out_dim]
                full_output = torch.matmul(down_output, up.t())
            elif down_output.ndim == 4:
                # Conv2d: [batch, rank, H, W]
                # up: [out_channels, rank]
                down_perm = down_output.permute(0, 2, 3, 1)  # [batch, H, W, rank]
                full_output = torch.matmul(down_perm, up.t())  # [batch, H, W, out_channels]
                full_output = full_output.permute(0, 3, 1, 2)  # [batch, out_channels, H, W]
            else:
                logger.warning(
                    f"Unexpected down_output shape: {down_output.shape}"
                )
                continue

            lora_deltas[lora_name] = full_output

        if not lora_deltas:
            logger.warning(f"No LoRA deltas computed for {layer_key}")
            return torch.zeros_like(original_output)

        # Route through adapter
        adapted_output = adapter(lora_deltas, self.semantic_vector)

        return adapted_output

    def set_semantic_vector(self, semantic_vector: torch.Tensor):
        """
        Set semantic vector for next forward pass.

        Args:
            semantic_vector: Feature importance vector [batch, semantic_dim]
        """
        self.semantic_vector = semantic_vector
        logger.debug(f"Set semantic vector with shape {semantic_vector.shape}")

    def remove_hooks(self):
        """Remove all registered hooks and clean up."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.adapter_outputs = {}
        logger.info("Removed all forward hooks")

    def __enter__(self):
        """Context manager entry - register hooks."""
        self.register_hooks()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - remove hooks."""
        self.remove_hooks()
        return False

    def __repr__(self) -> str:
        return (
            f"ForwardHookManager("
            f"hooks={len(self.hooks)}, "
            f"adapters={len(self.adapter_registry)}, "
            f"semantic_vector={'set' if self.semantic_vector is not None else 'unset'})"
        )
