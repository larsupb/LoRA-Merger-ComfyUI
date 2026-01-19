"""
Delta providers for semantic adapter training.

Provides abstraction for generating per-layer LoRA deltas, allowing
seamless switching between synthetic training and real UNet forward passes.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional
import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class DeltaProvider(ABC):
    """
    Abstract base class for delta providers.

    Delta providers generate per-layer LoRA deltas for training.
    This abstraction allows switching between:
    - Synthetic deltas (fast, no UNet forward)
    - Real UNet deltas (full forward passes)

    Invariant 1: DeltaProvider abstraction
    ----------------------------------------
    All delta providers must implement get_deltas() with this signature:

        get_deltas(batch) -> Dict[layer_key, Dict[lora_name, Tensor]]

    This ensures synthetic → UNet is a one-line swap.
    """

    @abstractmethod
    def get_deltas(
        self,
        batch: Dict,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Generate per-layer LoRA deltas for a batch.

        Args:
            batch: Training batch from data loader
                   Must contain: prompts, features, timesteps, latents, meta

        Returns:
            Dict[layer_key -> Dict[lora_name -> delta_tensor]]

            Example:
            {
                "down_blocks.0.attentions.0.attn1.to_q": {
                    "lora_A": Tensor[batch_size, output_dim],
                    "lora_B": Tensor[batch_size, output_dim],
                },
                ...
            }
        """
        pass


class SyntheticDeltaProvider(DeltaProvider):
    """
    Synthetic delta provider for fast training without UNet forward passes.

    Generates random deltas with correct shapes, allowing rapid iteration
    on adapter architecture and loss functions.

    Use for:
    - Testing adapter architecture
    - Debugging loss functions
    - Rapid prototyping
    - Unit tests

    Switch to UNetDeltaProvider for real training.
    """

    def __init__(
        self,
        adapter_registry,
        device: torch.device,
        dtype: torch.dtype,
        delta_magnitude: float = 0.1,
    ):
        """
        Initialize synthetic delta provider.

        Args:
            adapter_registry: AdapterRegistry with loaded LoRAs
            device: Compute device
            dtype: Compute dtype
            delta_magnitude: Standard deviation of random deltas
        """
        self.adapter_registry = adapter_registry
        self.device = device
        self.dtype = dtype
        self.delta_magnitude = delta_magnitude

        self.loaded_loras = adapter_registry.loaded_loras
        self.lora_names = list(self.loaded_loras.keys())

        logger.info(
            f"SyntheticDeltaProvider initialized: "
            f"{len(self.lora_names)} LoRAs, "
            f"magnitude={delta_magnitude}"
        )

    def get_deltas(
        self,
        batch: Dict,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Generate synthetic random deltas for all layers.

        Args:
            batch: Training batch (uses 'prompts' for batch size)

        Returns:
            Dict[layer_key -> Dict[lora_name -> delta_tensor]]
        """
        batch_size = len(batch.get("prompts", [1]))

        layer_deltas = {}

        # For each layer in the registry
        for layer_key, adapter in self.adapter_registry.adapters.items():
            layer_deltas[layer_key] = {}

            # Generate a delta for each LoRA
            for lora_name in self.lora_names:
                lora_data = self.loaded_loras[lora_name]

                # Find this layer's tensors in the LoRA
                norm_key = self._find_normalized_key(lora_data, layer_key)

                if norm_key is None:
                    # Layer not present in this LoRA
                    continue

                # Get up tensor to determine output shape
                up_tensor = lora_data["up_tensors"].get(norm_key)
                if up_tensor is None:
                    continue

                # Generate synthetic delta with correct shape
                output_dim = up_tensor.shape[0]

                # For attention layers, generate 3D tensors [batch, seq_len, channels]
                # For linear layers, generate 2D tensors [batch, channels]
                # Assume attention layers have "to_q", "to_k", "to_v", "to_out" in name
                if any(substr in layer_key for substr in ['.to_q', '.to_k', '.to_v', '.to_out', '.attn']):
                    # Attention layer: [batch_size, seq_len, output_dim]
                    seq_len = 77  # Standard CLIP sequence length
                    delta = torch.randn(
                        batch_size, seq_len, output_dim,
                        device=self.device,
                        dtype=self.dtype
                    ) * self.delta_magnitude
                else:
                    # Linear/MLP layer: [batch_size, output_dim]
                    delta = torch.randn(
                        batch_size, output_dim,
                        device=self.device,
                        dtype=self.dtype
                    ) * self.delta_magnitude

                layer_deltas[layer_key][lora_name] = delta

        return layer_deltas

    def _find_normalized_key(self, lora_data: Dict, layer_key: str):
        """
        Find the NormalizedKey that matches a layer_key string.

        Args:
            lora_data: Loaded LoRA data
            layer_key: Layer identifier string

        Returns:
            NormalizedKey or None
        """
        for norm_key in lora_data["up_tensors"].keys():
            norm_key_str = self.adapter_registry._normalized_key_to_string(norm_key)
            if norm_key_str == layer_key:
                return norm_key
        return None


class UNetDeltaProvider(DeltaProvider):
    """
    Real UNet delta provider (Phase 3 implementation).

    Generates deltas by running actual UNet forward passes with LoRAs applied.

    Maintains delta-isolation invariant:
        Δ_lora^ℓ = h_lora^ℓ - h_base^ℓ

    Uses diffusers pipeline with load_lora_weights/set_adapters for LoRA management.
    """

    def __init__(
        self,
        adapter_registry,
        checkpoint_path: str,
        device: torch.device = torch.device("cuda"),
        dtype: torch.dtype = torch.float16,
    ):
        """
        Initialize UNet delta provider.

        Args:
            adapter_registry: AdapterRegistry with loaded LoRAs
            checkpoint_path: Path to diffusion model checkpoint
            device: Compute device
            dtype: Compute dtype
        """
        self.adapter_registry = adapter_registry
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.dtype = dtype

        # Load pipeline with UNet
        from ..sd_loader import load_sd_for_gradients
        logger.info(f"Loading diffusion pipeline from {checkpoint_path}")
        self.pipe = load_sd_for_gradients(checkpoint_path, device, dtype)

        # Load all LoRAs from adapter registry
        self.lora_names = []
        self.lora_adapter_names = {}  # Map lora_name -> adapter_name in diffusers

        for lora_name, lora_entry in adapter_registry.lora_stack.items():
            lora_path = lora_entry["file_path"]

            # Create adapter name (diffusers requirement: no spaces/dashes)
            import os
            adapter_name = os.path.splitext(os.path.basename(lora_path))[0]
            adapter_name = adapter_name.replace(" ", "_").replace("-", "_")

            logger.info(f"  Loading LoRA: {lora_name} -> {adapter_name}")
            self.pipe.load_lora_weights(lora_path, adapter_name=adapter_name)

            self.lora_names.append(lora_name)
            self.lora_adapter_names[lora_name] = adapter_name

        # Disable all LoRAs initially (base state)
        # Use disable_lora() instead of set_adapters([]) to avoid KeyError
        if hasattr(self.pipe, 'disable_lora'):
            self.pipe.disable_lora()
        else:
            # Fallback: set all adapter scales to 0.0
            for adapter_name in self.lora_adapter_names.values():
                self.pipe.set_adapters(adapter_name, adapter_weights=0.0)

        # Hook management
        self.hooks = []
        self.activations = {}

        # Register forward hooks on all target layers
        self._register_hooks()

        logger.info(
            f"UNetDeltaProvider initialized: "
            f"{len(self.lora_names)} LoRAs, "
            f"{len(self.hooks)} hooks registered"
        )

    def _register_hooks(self):
        """Register forward hooks on selective target layers (attention + MLP only)."""
        unet = self.pipe.unet

        for name, module in unet.named_modules():
            if self._should_hook(module, name):
                # Normalize layer key to match adapter registry format
                layer_key = self._normalize_layer_key(name)

                # Register hook with closure to capture layer_key
                hook = module.register_forward_hook(
                    self._make_hook_fn(layer_key)
                )
                self.hooks.append(hook)

        logger.info(f"Registered {len(self.hooks)} forward hooks (selective: attention + MLP only)")

    def _should_hook(self, module, name: str = ""):
        """
        Determine if module should be hooked.

        Optimized selective hooking (Tier 1 VRAM optimization):
        - Hook: to_q, to_v, to_out (attention semantic core)
        - Skip: to_k, MLP, Conv2d, residuals

        Rationale:
        - to_q: Query intent / semantic focus
        - to_v: Semantic content
        - to_out: LoRA effect consolidation (stabilizes deltas)
        - MLP: High VRAM cost, low routing specificity (skip)

        Expected: ~280-340 hooks (~70% reduction from 1050)
        Expected VRAM savings: ~2-2.5GB
        """
        # Only hook Linear layers (skip Conv2d entirely)
        if not isinstance(module, nn.Linear):
            return False

        # Hook attention Q, V, and Out projections
        # This captures LoRA semantic signal with minimal VRAM
        if '.to_q' in name or '.to_v' in name or '.to_out' in name:
            return True

        # Skip everything else (to_k, MLP ff.net, residuals, etc.)
        return False

    def _normalize_layer_key(self, name: str) -> str:
        """
        Normalize diffusers layer name to adapter registry format.

        Args:
            name: Diffusers module name (e.g., "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.base_layer")

        Returns:
            Normalized key matching adapter registry format (without .base_layer suffix)
        """
        # Strip .base_layer suffix added by diffusers LoRA implementation
        if name.endswith('.base_layer'):
            name = name[:-11]  # Remove '.base_layer'

        # Strip any trailing index (e.g., ".0" from "to_out.0")
        # Adapter registry uses "to_out" not "to_out.0"
        if '.to_out.' in name:
            name = name.replace('.to_out.0', '.to_out')

        return name

    def _make_hook_fn(self, layer_key: str):
        """Create hook function with layer_key bound."""
        def hook_fn(module, input, output):
            self._store_activation(layer_key, output)
        return hook_fn

    def _store_activation(self, layer_key: str, output: torch.Tensor):
        """
        Store activation (called by forward hook).

        Tier 1 VRAM optimization: Move to CPU and cast to float16 immediately.
        Expected savings: ~1GB → ~3GB (by offloading to CPU)
        """
        # Detach to prevent gradient accumulation, move to CPU, cast to float16 for VRAM savings
        self.activations[layer_key] = output.detach().cpu().to(torch.float16)

    def _enable_lora(self, lora_name: Optional[str]):
        """
        Enable specific LoRA or disable all.

        Args:
            lora_name: LoRA to enable, or None to disable all
        """
        if lora_name is None:
            # Disable all LoRAs (base state)
            if hasattr(self.pipe, 'disable_lora'):
                self.pipe.disable_lora()
            else:
                # Fallback: set all adapter scales to 0.0
                for adapter_name in self.lora_adapter_names.values():
                    self.pipe.set_adapters(adapter_name, adapter_weights=0.0)
        else:
            # Enable specific LoRA with scale 1.0
            adapter_name = self.lora_adapter_names[lora_name]
            self.pipe.set_adapters(adapter_name, adapter_weights=1.0)

    def _run_and_capture(
        self,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        prompt_embeds: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Run UNet forward and capture activations.

        Args:
            latents: Latent tensors [batch_size, 4, H, W]
            timesteps: Timestep tensor [batch_size]
            prompt_embeds: CLIP embeddings [batch_size, seq_len, embed_dim]

        Returns:
            Dict of captured activations
        """
        self.activations = {}  # Reset storage

        # Run forward pass (hooks will populate self.activations)
        with torch.no_grad():  # No gradients during delta capture
            _ = self.pipe.unet(
                latents,
                timesteps,
                encoder_hidden_states=prompt_embeds,
            )

        return self.activations.copy()

    def get_deltas(
        self,
        batch: Dict,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Generate deltas from real UNet forward passes.

        Maintains delta-isolation invariant:
            Δ_lora^ℓ = h_lora^ℓ - h_base^ℓ

        Args:
            batch: Training batch with prompts, timesteps, latents

        Returns:
            Dict[layer_key -> Dict[lora_name -> delta_tensor]]
        """
        prompts = batch["prompts"]
        timesteps = batch["timesteps"].to(self.device)  # Ensure correct device
        latents = batch["latents"].to(self.device, dtype=self.dtype)  # Ensure correct device and dtype

        # 1. Encode prompts with CLIP
        prompt_embeds = self.pipe.encode_prompt(
            prompt=prompts,
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
        )[0]  # Returns (prompt_embeds, negative_prompt_embeds)

        # 2. Base pass (no LoRA) - CRITICAL for delta-isolation
        self._enable_lora(None)
        base_acts = self._run_and_capture(latents, timesteps, prompt_embeds)

        logger.debug(f"Base pass captured {len(base_acts)} layers")

        # 3. Initialize delta storage
        deltas = {}  # Dict[layer_key, Dict[lora_name, Tensor]]

        # 4. Per-LoRA passes
        for lora_name in self.lora_names:
            self._enable_lora(lora_name)

            lora_acts = self._run_and_capture(latents, timesteps, prompt_embeds)

            logger.debug(f"LoRA '{lora_name}' pass captured {len(lora_acts)} layers")

            # 5. Compute deltas (CRITICAL: maintain delta-isolation invariant)
            for layer_key in base_acts:
                if layer_key not in deltas:
                    deltas[layer_key] = {}

                if layer_key in lora_acts:
                    # Delta-isolation invariant: Δ = h_lora - h_base
                    delta = lora_acts[layer_key] - base_acts[layer_key]

                    # Safeguard assertions
                    assert delta.shape == base_acts[layer_key].shape, \
                        f"Shape mismatch at {layer_key}: {delta.shape} vs {base_acts[layer_key].shape}"

                    assert not torch.isnan(delta).any(), \
                        f"NaN detected in delta for {layer_key}/{lora_name}"

                    assert not torch.isinf(delta).any(), \
                        f"Inf detected in delta for {layer_key}/{lora_name}"

                    deltas[layer_key][lora_name] = delta

            # Free memory after each LoRA pass
            torch.cuda.empty_cache()

        # 6. Restore base state
        self._enable_lora(None)

        logger.debug(f"Generated deltas for {len(deltas)} layers")
        if deltas:
            # Debug: Show first few delta keys and shapes
            sample_keys = list(deltas.keys())[:5]
            logger.debug(f"Sample delta keys: {sample_keys}")

            # Debug: Show delta shapes for first layer
            first_key = sample_keys[0]
            first_layer_deltas = deltas[first_key]
            delta_shapes = {lora_name: delta.shape for lora_name, delta in first_layer_deltas.items()}
            logger.debug(f"Sample delta shapes for '{first_key}': {delta_shapes}")
        else:
            logger.warning("No deltas generated - possible layer key mismatch")

        return deltas

    def cleanup(self):
        """Remove hooks and free resources."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

        del self.pipe
        torch.cuda.empty_cache()

        logger.info("UNetDeltaProvider cleaned up")

    def __del__(self):
        """Cleanup on deletion."""
        if hasattr(self, 'hooks') and self.hooks:
            for hook in self.hooks:
                hook.remove()


# Factory function for convenience
def create_delta_provider(
    provider_type: str,
    adapter_registry,
    device: torch.device,
    dtype: torch.dtype,
    checkpoint_path: Optional[str] = None,
    **kwargs,
) -> DeltaProvider:
    """
    Factory function to create delta providers.

    Args:
        provider_type: "synthetic" or "unet"
        adapter_registry: AdapterRegistry instance
        device: Compute device
        dtype: Compute dtype
        checkpoint_path: Path to checkpoint (required for "unet")
        **kwargs: Additional provider-specific arguments

    Returns:
        DeltaProvider instance

    Example:
        # Synthetic (fast, no UNet)
        provider = create_delta_provider(
            "synthetic",
            adapter_registry,
            device,
            dtype
        )

        # Real UNet (Phase 3)
        provider = create_delta_provider(
            "unet",
            adapter_registry,
            device,
            dtype,
            checkpoint_path="/path/to/checkpoint.safetensors"
        )
    """
    if provider_type == "synthetic":
        return SyntheticDeltaProvider(
            adapter_registry=adapter_registry,
            device=device,
            dtype=dtype,
            **kwargs,
        )
    elif provider_type == "unet":
        if checkpoint_path is None:
            raise ValueError("checkpoint_path is required for UNetDeltaProvider")

        return UNetDeltaProvider(
            adapter_registry=adapter_registry,
            checkpoint_path=checkpoint_path,
            device=device,
            dtype=dtype,
            **kwargs,
        )
    else:
        raise ValueError(
            f"Unknown provider type: {provider_type}. "
            f"Choose 'synthetic' or 'unet'."
        )
