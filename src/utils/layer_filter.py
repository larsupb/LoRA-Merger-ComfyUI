"""
Layer filtering utilities for LoRA Power-Merger.

Provides LayerFilter class for selective layer merging operations.
"""

from typing import Set, Optional, Dict, Any, Tuple
from ..types import LORA_KEY_DICT, LayerComponentSet
from .config import (
    ATTENTION_LAYERS,
    MLP_LAYERS,
    ATTENTION_MLP_LAYERS,
)

# CLIP layer key prefixes for detection
CLIP_KEY_PREFIXES = {
    # ComfyUI internal CLIP formats
    'clip_l.',  # CLIP-L (OpenAI CLIP Large)
    'clip_g.',  # CLIP-G (OpenCLIP bigG)
    'clip_h.',  # CLIP-H (OpenCLIP Huge)
    # LoRA file formats
    'lora_te',  # SD1.5 text encoder
    'text_encoder',  # Generic text encoder
    'lora_te1_text_model',  # SDXL text encoder 1
    'lora_te2_text_model',  # SDXL text encoder 2
    'text_model',  # Generic text model
    'transformer.text_model',  # Transformer text model
}


def detect_lora_architecture(patch_dict: LORA_KEY_DICT, sample_size: int = 20) -> Tuple[str, Dict[str, Any]]:
    """
    Detect the architecture of a LoRA by analyzing its layer keys.

    Args:
        patch_dict: Dictionary of LoRA patches
        sample_size: Number of keys to sample for detection (default: 20)

    Returns:
        Tuple of (architecture_name, metadata_dict)

    Examples:
        >>> arch, meta = detect_lora_architecture(lora_patches)
        >>> print(f"Detected {arch} architecture")
        >>> # "Detected Wan 2.2 architecture"
    """
    if not patch_dict:
        return "Unknown", {"reason": "Empty patch dictionary"}

    # Sample keys for analysis
    all_keys = list(patch_dict.keys())
    sample_keys = all_keys[:min(sample_size, len(all_keys))]

    # Convert keys to strings (handle tuple keys from ComfyUI)
    key_strings = [str(k[0]) if isinstance(k, tuple) else str(k) for k in sample_keys]
    all_keys_str = ' '.join(key_strings).lower()

    metadata = {
        "total_keys": len(all_keys),
        "sample_size": len(sample_keys),
        "confidence": "high"
    }

    # Architecture detection patterns (order matters - most specific first)

    # 1. Wan 2.2: diffusion_model.blocks.N.{self_attn,cross_attn,ffn}
    if 'diffusion_model.blocks.' in all_keys_str and ('self_attn' in all_keys_str or 'cross_attn' in all_keys_str):
        if 'ffn' in all_keys_str:
            metadata["attention_type"] = "self_attn + cross_attn"
            metadata["mlp_type"] = "ffn"
            return "Wan 2.2", metadata

    # 2. Flux: double_blocks/single_blocks with img_attn/txt_attn and img_mlp/txt_mlp
    if 'double_blocks' in all_keys_str or 'single_blocks' in all_keys_str:
        has_img_attn = 'img_attn' in all_keys_str
        has_txt_attn = 'txt_attn' in all_keys_str
        has_img_mlp = 'img_mlp' in all_keys_str
        has_txt_mlp = 'txt_mlp' in all_keys_str

        if (has_img_attn or has_txt_attn) and (has_img_mlp or has_txt_mlp):
            metadata["architecture_type"] = "Dual-stream transformer"
            metadata["has_double_blocks"] = 'double_blocks' in all_keys_str
            metadata["has_single_blocks"] = 'single_blocks' in all_keys_str
            return "Flux", metadata

    # 3. Qwen Image Edit: transformer_blocks.N.attn.* and {img_mlp,txt_mlp}
    if 'transformer_blocks.' in all_keys_str and '.attn.' in all_keys_str:
        has_img_mlp = 'img_mlp' in all_keys_str
        has_txt_mlp = 'txt_mlp' in all_keys_str

        if has_img_mlp or has_txt_mlp:
            metadata["architecture_type"] = "Image-text transformer"
            metadata["mlp_type"] = "img_mlp + txt_mlp"
            return "Qwen Image Edit", metadata

    # 4. zImage: diffusion_model.layers.N with feed_forward and adaLN_modulation
    if 'diffusion_model.layers.' in all_keys_str and 'feed_forward' in all_keys_str:
        if 'adaln_modulation' in all_keys_str:
            metadata["architecture_type"] = "Pure DiT"
            metadata["has_adaptive_norm"] = True
            metadata["attention_type"] = "attention"
            metadata["mlp_type"] = "feed_forward"
            return "zImage", metadata

    # 5. Generic DiT: diffusion_model.layers.N.{attention,feed_forward}
    if 'diffusion_model.layers.' in all_keys_str and '.attention.' in all_keys_str:
        metadata["architecture_type"] = "DiT (Diffusion Transformer)"
        metadata["attention_type"] = "attention"
        metadata["mlp_type"] = "mlp or feed_forward"
        return "DiT", metadata

    # 6. Stable Diffusion: lora_unet with attn1/attn2 and ff
    if 'lora_unet' in all_keys_str or 'unet' in all_keys_str:
        has_attn1 = 'attn1' in all_keys_str
        has_attn2 = 'attn2' in all_keys_str
        has_ff = '_ff' in all_keys_str

        if has_attn1 or has_attn2:
            # Determine if SDXL or SD1.5 based on key patterns
            if 'down_blocks' in all_keys_str or 'up_blocks' in all_keys_str:
                metadata["architecture_type"] = "UNet"
                metadata["attention_type"] = "attn1 + attn2"
                metadata["mlp_type"] = "ff"

                # Try to distinguish SDXL vs SD1.5 by complexity
                if len(all_keys) > 500:
                    return "Stable Diffusion XL", metadata
                else:
                    return "Stable Diffusion 1.5", metadata

    # 7. Fallback: Unknown architecture
    metadata["confidence"] = "low"
    metadata["reason"] = "No matching architecture pattern found"

    # Provide hints about what was found
    patterns_found = []
    if 'attention' in all_keys_str:
        patterns_found.append("attention layers")
    if 'mlp' in all_keys_str or 'feed_forward' in all_keys_str or 'ff' in all_keys_str:
        patterns_found.append("MLP/feedforward layers")

    if patterns_found:
        metadata["patterns_found"] = patterns_found

    return "Unknown", metadata


def is_clip_layer(key) -> bool:
    """
    Determine if a layer key belongs to CLIP/text encoder layers.

    Args:
        key: Layer key (can be string or ComfyUI tuple format)

    Returns:
        True if the key is a CLIP layer, False otherwise

    Examples:
        >>> is_clip_layer("lora_te_text_model_encoder_layers_0_self_attn_q_proj.lora_down.weight")
        True
        >>> is_clip_layer("lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q.lora_down.weight")
        False
        >>> is_clip_layer("text_encoder.encoder.layers.11.mlp.fc1.lora_up.weight")
        True
    """
    # Handle tuple keys from ComfyUI
    key_str = str(key[0]) if isinstance(key, tuple) else str(key)

    # Check if key starts with any CLIP prefix
    return any(key_str.startswith(prefix) for prefix in CLIP_KEY_PREFIXES)


class LayerFilter:
    """
    Layer filtering for selective LoRA merging.

    Allows filtering LoRA patches to include only specific layer types
    (e.g., attention only, attention + MLP, etc.).
    """

    # Predefined filter presets (architecture-agnostic, works for both SD and DiT)
    PRESETS = {
        "full": None,  # No filtering
        "attn-only": ATTENTION_LAYERS,
        "attn-mlp": ATTENTION_MLP_LAYERS,
        "mlp-only": MLP_LAYERS,
    }

    def __init__(self, filter_spec: str | Set[str] | None):
        """
        Initialize layer filter.

        Args:
            filter_spec: Filter specification:
                - None or "full": No filtering (all layers)
                - Preset name (str): Use predefined filter set
                - Set of strings: Custom component names to include

        Examples:
            >>> LayerFilter("attn-only")  # Only attention layers
            >>> LayerFilter(None)  # No filtering
        """
        if filter_spec is None or filter_spec == "full":
            self.filter_set = None
        elif isinstance(filter_spec, str):
            # Look up preset
            if filter_spec not in self.PRESETS:
                raise ValueError(
                    f"Unknown filter preset: {filter_spec}. "
                    f"Available: {', '.join(self.PRESETS.keys())}"
                )
            self.filter_set = self.PRESETS[filter_spec]
        elif isinstance(filter_spec, set):
            self.filter_set = filter_spec
        else:
            raise TypeError(
                f"filter_spec must be None, str, or set. Got {type(filter_spec)}"
            )

    def apply(self, patch_dict: LORA_KEY_DICT, detect_architecture: bool = True) -> LORA_KEY_DICT:
        """
        Apply filter to patch dictionary.

        Args:
            patch_dict: Dictionary of layer key -> LoRAAdapter
            detect_architecture: Whether to detect and log architecture (default: True)

        Returns:
            Filtered patch dictionary

        Note:
            Uses component-based matching (split by '.') to avoid false positives
            from substring matches (e.g., 'ff' in 'diffusion_model').

        Example:
            >>> filter = LayerFilter("attn-only")
            >>> filtered = filter.apply(lora_patches)
        """
        import logging

        # Detect architecture before filtering
        if detect_architecture and patch_dict:
            arch_name, arch_meta = detect_lora_architecture(patch_dict)
            total_keys = arch_meta.get("total_keys", len(patch_dict))

            if arch_name != "Unknown":
                logging.info(f"Detected {arch_name} architecture ({total_keys} keys)")
            else:
                logging.info(f"Processing LoRA ({total_keys} keys, architecture: {arch_name})")

        if self.filter_set is None:
            # No filtering
            return patch_dict

        original_count = len(patch_dict)

        def matches_filter(key) -> bool:
            """
            Check if key matches any filter component.

            Uses word-boundary aware matching to avoid false positives.
            For example, 'ff' will match 'ff_net' or '.ff.' but not 'diffusion'.
            """
            import re

            # Handle tuple keys from ComfyUI
            key_str = str(key[0]) if isinstance(key, tuple) else str(key)
            key_lower = key_str.lower()

            for filter_pattern in self.filter_set:
                pattern_lower = filter_pattern.lower()

                # Create regex pattern with word boundaries
                # \b doesn't work well with underscores, so use custom pattern
                # Match pattern when it appears:
                # - at start/end of string
                # - surrounded by dots, underscores, or other non-alphanumeric chars
                regex_pattern = r'(?:^|[._])' + re.escape(pattern_lower) + r'(?:[._]|$)'

                if re.search(regex_pattern, key_lower):
                    return True

            return False

        # Filter to only include keys that match filter components
        filtered_dict = {
            key: value
            for key, value in patch_dict.items()
            if matches_filter(key)
        }

        filtered_count = original_count - len(filtered_dict)

        if filtered_count > 0:
            logging.info(
                f"Layer filter: kept {len(filtered_dict)}/{original_count} keys "
                f"({filtered_count} filtered out)"
            )

        return filtered_dict

    def __str__(self) -> str:
        """String representation of filter."""
        if self.filter_set is None:
            return "LayerFilter(full)"
        return f"LayerFilter({self.filter_set})"

    def __repr__(self) -> str:
        """Detailed representation of filter."""
        return self.__str__()

    @classmethod
    def list_presets(cls) -> Dict[str, Optional[Set[str]]]:
        """
        List all available filter presets.

        Returns:
            Dictionary of preset name -> component set
        """
        return cls.PRESETS.copy()

    def is_empty(self) -> bool:
        """Check if filter would exclude all layers."""
        return self.filter_set is not None and len(self.filter_set) == 0

    def matches(self, layer_key) -> bool:
        """
        Check if a layer key matches the filter.

        Args:
            layer_key: Layer key to check (can be string or tuple)

        Returns:
            True if layer matches filter (or no filter), False otherwise

        Note:
            Uses word-boundary aware matching to avoid false positives.
            For example, 'ff' will match 'ff_net' or '.ff.' but not 'diffusion'.
        """
        if self.filter_set is None:
            return True

        import re

        # Handle tuple keys from ComfyUI
        key_str = str(layer_key[0]) if isinstance(layer_key, tuple) else str(layer_key)
        key_lower = key_str.lower()

        for filter_pattern in self.filter_set:
            pattern_lower = filter_pattern.lower()

            # Create regex pattern with word boundaries
            # Match pattern when surrounded by dots, underscores, or at start/end
            regex_pattern = r'(?:^|[._])' + re.escape(pattern_lower) + r'(?:[._]|$)'

            if re.search(regex_pattern, key_lower):
                return True

        return False
