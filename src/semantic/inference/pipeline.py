"""
Inference pipeline for applying trained adapters.

Loads trained adapters and applies them during inference without
requiring additional training.
"""

import logging
from typing import Dict, Optional
import torch
import torch.nn as nn

from ..adapter.registry import AdapterRegistry
from ..adapter.hook_manager import ForwardHookManager

logger = logging.getLogger(__name__)


class AdapterInferencePipeline:
    """
    Inference pipeline for applying trained adapters.

    Takes a trained adapter registry and applies it to a base model
    during inference using forward hooks. No training occurs - adapters
    are frozen.
    """

    def __init__(
        self,
        base_model: nn.Module,
        adapter_registry: AdapterRegistry,
        lora_stack: Dict,
        device: torch.device,
        dtype: torch.dtype,
    ):
        """
        Initialize inference pipeline.

        Args:
            base_model: Base diffusion model (UNet)
            adapter_registry: Registry with trained adapters
            lora_stack: LoRA stack
            device: Compute device
            dtype: Compute dtype
        """
        self.base_model = base_model
        self.adapter_registry = adapter_registry
        self.lora_stack = lora_stack
        self.device = device
        self.dtype = dtype

        # Freeze adapters for inference
        for adapter in self.adapter_registry.get_all_adapters().values():
            adapter.eval()
            for param in adapter.parameters():
                param.requires_grad = False

        # Create hook manager (but don't register yet)
        self.hook_manager: Optional[ForwardHookManager] = None

        logger.info(
            f"Initialized inference pipeline with {len(adapter_registry)} adapters"
        )

    def setup_hooks(self):
        """
        Setup forward hooks for adapter injection.

        Creates and registers a ForwardHookManager that will intercept
        layer computations and route through adapters.
        """
        if self.hook_manager is not None:
            logger.warning("Hooks already set up, removing old hooks first")
            self.remove_hooks()

        # Create hook manager
        self.hook_manager = ForwardHookManager(
            model=self.base_model,
            adapter_registry=self.adapter_registry,
            lora_stack=self.lora_stack,
            device=self.device,
            dtype=self.dtype,
        )

        # Register hooks
        self.hook_manager.register_hooks()

        logger.info("Forward hooks registered for adapter injection")

    def remove_hooks(self):
        """Remove forward hooks."""
        if self.hook_manager is not None:
            self.hook_manager.remove_hooks()
            self.hook_manager = None
            logger.info("Forward hooks removed")

    def set_semantic_vector(self, semantic_vector: torch.Tensor):
        """
        Set semantic vector for next forward pass.

        Args:
            semantic_vector: Feature importance [batch, semantic_dim]
        """
        if self.hook_manager is None:
            raise RuntimeError(
                "Hooks not set up. Call setup_hooks() first."
            )

        self.hook_manager.set_semantic_vector(semantic_vector)

    def generate(
        self,
        prompt: str,
        semantic_vector: torch.Tensor,
        **generation_kwargs
    ) -> torch.Tensor:
        """
        Generate image using adapters.

        Args:
            prompt: Text prompt
            semantic_vector: Semantic importance vector [batch, semantic_dim]
            **generation_kwargs: Additional generation arguments

        Returns:
            Generated image tensor

        Note: This is a placeholder. Full diffusion loop integration
        requires connecting to ComfyUI's sampling infrastructure.
        """
        # Set semantic vector for this generation
        self.set_semantic_vector(semantic_vector)

        # TODO: Implement full diffusion sampling loop
        # For now, this just validates the interface
        logger.warning(
            "Full generation loop not yet implemented - "
            "would integrate with ComfyUI sampling here"
        )

        # Placeholder: return dummy tensor
        return torch.randn(1, 3, 512, 512, device=self.device, dtype=self.dtype)

    def __enter__(self):
        """Context manager entry - setup hooks."""
        self.setup_hooks()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - remove hooks."""
        self.remove_hooks()
        return False

    def __repr__(self) -> str:
        return (
            f"AdapterInferencePipeline("
            f"adapters={len(self.adapter_registry)}, "
            f"hooks={'registered' if self.hook_manager else 'not registered'}, "
            f"device={self.device})"
        )
