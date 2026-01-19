"""
Adapter registry for managing adapters across all layers.

Automatically determines adapter type (rank-space vs output-space) based on
layer characteristics and creates appropriate adapter instances.
"""

import logging
import re
from typing import Dict, List, Literal, Tuple
from pathlib import Path
import torch

from .base import BaseAdapter
from .rank_space import RankSpaceAdapter
from .output_space import OutputSpaceAdapter
from ...types import LORA_STACK
from ...utils.config import ATTENTION_LAYERS, MLP_LAYERS
from ..lora_loader import load_lora_safetensors
from ..key_utils import NormalizedKey

AdapterType = Literal["rank_space", "output_space"]


def is_attention_layer(layer_key: str) -> bool:
    """
    Determine if a layer is an attention layer.

    Checks for attention-related patterns in the layer key using word-boundary
    aware matching to avoid false positives.

    Args:
        layer_key: Layer identifier string

    Returns:
        True if layer is attention-related

    Examples:
        >>> is_attention_layer("diffusion_model.layers.0.attention.q")
        True
        >>> is_attention_layer("lora_unet_down_blocks_0_attentions_0_attn1_to_q")
        True
        >>> is_attention_layer("diffusion_model.layers.0.mlp.w1")
        False
    """
    # Handle tuple keys from ComfyUI
    key_str = str(layer_key[0]) if isinstance(layer_key, tuple) else str(layer_key)
    key_str = key_str.lower()

    # Check for attention patterns with word boundaries
    for pattern in ATTENTION_LAYERS:
        # Match pattern only at word boundaries (after . or _)
        regex = rf'(?:^|[._]){re.escape(pattern)}(?:[._]|$)'
        if re.search(regex, key_str):
            return True

    return False


def is_mlp_layer(layer_key: str) -> bool:
    """
    Determine if a layer is an MLP/feedforward layer.

    Checks for MLP-related patterns in the layer key using word-boundary
    aware matching.

    Args:
        layer_key: Layer identifier string

    Returns:
        True if layer is MLP-related

    Examples:
        >>> is_mlp_layer("diffusion_model.layers.0.mlp.w1")
        True
        >>> is_mlp_layer("lora_unet_down_blocks_0_ff_net_0_proj")
        True
        >>> is_mlp_layer("diffusion_model.layers.0.attention.q")
        False
    """
    # Handle tuple keys from ComfyUI
    key_str = str(layer_key[0]) if isinstance(layer_key, tuple) else str(layer_key)
    key_str = key_str.lower()

    # Check for MLP patterns with word boundaries
    for pattern in MLP_LAYERS:
        # Match pattern only at word boundaries (after . or _)
        regex = rf'(?:^|[._]){re.escape(pattern)}(?:[._]|$)'
        if re.search(regex, key_str):
            return True

    return False


class AdapterRegistry:
    """
    Registry for managing adapters across all layers.

    Automatically determines adapter type based on layer characteristics
    and creates appropriate adapter instances.

    Adapter Type Strategy:
    - Attention (Q/K/V/Out): rank_space (concepts encoded in low-rank subspace)
    - MLP (linear): rank_space (style & feature modulation)
    - Conv2D: output_space (no natural rank decomposition)
    - Unknown: output_space (safe fallback)
    """

    def __init__(
        self,
        lora_stack: LORA_STACK,
        semantic_maps: Dict[str, Dict[str, float]],
        feature_names: List[str],
        device: torch.device,
        dtype: torch.dtype,
    ):
        """
        Initialize adapter registry.

        Args:
            lora_stack: LoRA stack (only file paths are used, LoRAs loaded from disk)
            semantic_maps: Semantic importance maps (can be empty for inference)
            feature_names: List of feature names
            device: Compute device
            dtype: Compute dtype
        """
        self.lora_stack = lora_stack
        self.semantic_maps = semantic_maps
        self.feature_names = feature_names
        self.device = device
        self.dtype = dtype

        self.adapters: Dict[str, BaseAdapter] = {}
        self.adapter_types: Dict[str, AdapterType] = {}

        # Load LoRAs from disk using diffusers (same as GradientSemanticAnalyzer)
        self.loaded_loras: Dict[str, Dict] = {}
        self._load_loras_from_disk()

        self._build_registry()

    def _load_loras_from_disk(self):
        """
        Load LoRAs from disk using diffusers (same approach as GradientSemanticAnalyzer).

        Extracts file paths from lora_stack and loads LoRAs using load_lora_safetensors.
        """
        logging.info(f"Loading {len(self.lora_stack)} LoRAs from disk...")

        for lora_name, lora_entry in self.lora_stack.items():
            # Extract file path
            file_path = lora_entry.get("file_path")
            if not file_path:
                raise ValueError(f"LoRA '{lora_name}' is missing file_path in lora_stack")

            # Check file exists
            if not Path(file_path).exists():
                raise FileNotFoundError(f"LoRA file not found: {file_path}")

            # Load LoRA from disk
            logging.info(f"  Loading {lora_name} from {file_path}")
            loaded = load_lora_safetensors(
                path=file_path,
                device=self.device,
                dtype=self.dtype,
            )

            self.loaded_loras[lora_name] = loaded
            logging.info(
                f"    Loaded {len(loaded['up_tensors'])} layers "
                f"(normalized keys)"
            )

        logging.info(f"✓ All LoRAs loaded from disk")

    def _determine_adapter_type(self, layer_key: str) -> AdapterType:
        """
        Determine adapter type based on layer characteristics.

        Strategy (Phase 3 - UNet output-space deltas):
        - Hooked Linear attention layers (to_q, to_v, to_out): output_space
        - Other attention/MLP: rank_space (not currently hooked in Phase 3)
        - Conv2D: output_space
        - Unknown: output_space (safe fallback)

        Rationale:
        UNetDeltaProvider hooks specific Linear layers and produces output-space
        deltas (full activations). These layers require OutputSpaceAdapters that
        operate on final LoRA outputs, not intermediate rank-space signals.

        Args:
            layer_key: Layer identifier

        Returns:
            "rank_space" or "output_space"
        """
        # Phase 3: Hooked Linear attention layers use output-space adapters
        # These match the layers hooked in UNetDeltaProvider._should_hook()
        if layer_key.endswith('.to_q') or layer_key.endswith('.to_v') or layer_key.endswith('.to_out'):
            return "output_space"

        # All other layers: rank_space (not currently used in Phase 3 training)
        elif is_attention_layer(layer_key) or is_mlp_layer(layer_key):
            return "rank_space"
        else:
            return "output_space"

    def _build_registry(self):
        """Build adapter registry for all layers in LoRA stack."""
        # Get all unique layer keys (NormalizedKey) across all loaded LoRAs
        all_keys = set()
        for lora_name, loaded_lora in self.loaded_loras.items():
            all_keys.update(loaded_lora["up_tensors"].keys())

        semantic_dim = len(self.feature_names)
        num_loras = len(self.loaded_loras)

        logging.info(
            f"Building adapter registry for {len(all_keys)} layers, "
            f"{num_loras} LoRAs, {semantic_dim} features"
        )

        # Count adapter types for logging
        rank_space_count = 0
        output_space_count = 0

        # Convert NormalizedKeys to strings first for sorting
        # (NormalizedKey objects don't support direct comparison)
        key_pairs = [(norm_key, self._normalized_key_to_string(norm_key)) for norm_key in all_keys]
        key_pairs.sort(key=lambda x: x[1])  # Sort by string representation

        for norm_key, layer_key in key_pairs:

            adapter_type = self._determine_adapter_type(layer_key)
            self.adapter_types[layer_key] = adapter_type

            # Get layer dimensions from first LoRA that has this key
            up, down, alpha = None, None, None
            for loaded_lora in self.loaded_loras.values():
                if norm_key in loaded_lora["up_tensors"] and norm_key in loaded_lora["down_tensors"]:
                    up = loaded_lora["up_tensors"][norm_key]
                    down = loaded_lora["down_tensors"][norm_key]
                    alpha = loaded_lora.get("alpha_tensors", {}).get(norm_key)
                    break

            if up is None or down is None:
                logging.warning(
                    f"Layer {layer_key} not found in any LoRA, using default dimensions"
                )
                rank = 32
                output_dim = 768
            else:
                rank = down.shape[0]
                output_dim = up.shape[0]

            # Create adapter
            if adapter_type == "rank_space":
                adapter = RankSpaceAdapter(
                    layer_key=layer_key,
                    semantic_dim=semantic_dim,
                    feature_names=self.feature_names,
                    rank=rank,
                    num_loras=num_loras,
                    use_residual_mlp=True,
                ).to(device=self.device, dtype=self.dtype)
                rank_space_count += 1
            else:
                adapter = OutputSpaceAdapter(
                    layer_key=layer_key,
                    semantic_dim=semantic_dim,
                    feature_names=self.feature_names,
                    output_dim=output_dim,
                    num_loras=num_loras,
                ).to(device=self.device, dtype=self.dtype)
                output_space_count += 1

            self.adapters[layer_key] = adapter

        logging.info(
            f"Created {rank_space_count} rank-space adapters, "
            f"{output_space_count} output-space adapters"
        )

    def _normalized_key_to_string(self, norm_key: NormalizedKey) -> str:
        """
        Convert NormalizedKey to a canonical string representation.

        This creates a string key that can be used for adapter registry lookup.

        Args:
            norm_key: Normalized key from LoRA loader

        Returns:
            String representation (e.g., "down_blocks.0.attentions.0.attn1.to_q")
        """
        parts = []

        # Block type and index
        if norm_key.block_type == "down":
            parts.append(f"down_blocks.{norm_key.block_idx}")
        elif norm_key.block_type == "mid":
            parts.append("mid_block")
        elif norm_key.block_type == "up":
            parts.append(f"up_blocks.{norm_key.block_idx}")
        else:
            parts.append(f"{norm_key.block_type}")

        # Attention index (if applicable)
        if norm_key.attention_idx is not None:
            parts.append(f"attentions.{norm_key.attention_idx}")

        # Transformer block
        parts.append(f"transformer_blocks.{norm_key.transformer_idx}")

        # Attention type
        if norm_key.attn_type == "self":
            parts.append("attn1")
        elif norm_key.attn_type == "cross":
            parts.append("attn2")
        elif norm_key.attn_type:
            parts.append(norm_key.attn_type)

        # Sublayer
        parts.append(norm_key.sublayer)

        return ".".join(parts)

    def get_adapter(self, layer_key: str) -> BaseAdapter:
        """
        Get adapter for a specific layer.

        Args:
            layer_key: Layer identifier

        Returns:
            Adapter instance

        Raises:
            KeyError: If layer_key not in registry
        """
        return self.adapters[layer_key]

    def get_all_adapters(self) -> Dict[str, BaseAdapter]:
        """
        Get all adapters in registry.

        Returns:
            Dict mapping layer_key -> adapter
        """
        return self.adapters

    def get_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Get statistics for all adapters.

        Returns:
            Dict mapping layer_key -> statistics dict
        """
        stats = {}
        for layer_key, adapter in self.adapters.items():
            stats[layer_key] = adapter.get_statistics()
        return stats

    def parameters(self):
        """
        Get all trainable parameters from all adapters.

        Yields:
            Parameter tensors
        """
        for adapter in self.adapters.values():
            yield from adapter.parameters()

    def state_dict(self) -> Dict:
        """
        Get state dict for all adapters.

        Returns:
            Dict with adapter_types, adapters state dicts, and feature_names
        """
        return {
            "adapter_types": self.adapter_types,
            "adapters": {
                key: adapter.state_dict()
                for key, adapter in self.adapters.items()
            },
            "feature_names": self.feature_names,
        }

    def load_state_dict(self, state_dict: Dict):
        """
        Load state dict for all adapters.

        Args:
            state_dict: State dict from state_dict() method

        Raises:
            KeyError: If state_dict missing required keys
        """
        for key, adapter_state in state_dict["adapters"].items():
            if key in self.adapters:
                self.adapters[key].load_state_dict(adapter_state)
            else:
                logging.warning(
                    f"Layer {key} in state_dict but not in registry, skipping"
                )

    def __len__(self) -> int:
        """Return number of adapters in registry."""
        return len(self.adapters)

    def __repr__(self) -> str:
        rank_space_count = sum(
            1 for t in self.adapter_types.values() if t == "rank_space"
        )
        output_space_count = sum(
            1 for t in self.adapter_types.values() if t == "output_space"
        )
        return (
            f"AdapterRegistry("
            f"total={len(self.adapters)}, "
            f"rank_space={rank_space_count}, "
            f"output_space={output_space_count})"
        )
