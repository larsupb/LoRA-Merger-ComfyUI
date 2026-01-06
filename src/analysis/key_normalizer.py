"""
Normalize LoRA keys from different formats to a unified representation.

Supports:

- Kohya format: lora_unet_down_blocks_0_attentions_0_...
- Diffusers format: unet.down_blocks.0.attentions.0...

- Diffusers processor format: down_blocks.0.attentions.0.to_q.processor.to_q_lora.down.weight

"""

import re
import logging
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class NormalizedKey:
    """
    Normalized representation of a LoRA parameter location.

    Captures the logical position in the network independent of format.
    """
    component: str  # "unet", "text_encoder", "text_encoder_2"
    block_type: str  # "down", "mid", "up", "input", "output"
    block_idx: int  # Block index
    layer_type: str  # "attn", "attn1", "attn2", "ff", "resnet", "proj"
    layer_idx: Optional[int]  # Layer index within block (if applicable)
    sublayer: str  # "to_q", "to_k", "to_v", "to_out", "net.0.proj", etc.
    lora_type: str  # "down", "up", "alpha"

    def to_base_key(self) -> str:
        """
        Generate a base key for matching (without lora_type).

        Format: component.block_type.block_idx.layer_type[.layer_idx].sublayer
        """
        parts = [self.component, self.block_type, str(self.block_idx), self.layer_type]
        if self.layer_idx is not None:
            parts.append(str(self.layer_idx))
        parts.append(self.sublayer)
        return ".".join(parts)

    def __hash__(self):
        return hash(self.to_base_key())

    def __eq__(self, other):
        if not isinstance(other, NormalizedKey):
            return False
        return self.to_base_key() == other.to_base_key()


class LoRAKeyNormalizer:
    """
    Normalize LoRA keys from various formats to NormalizedKey objects.
    """

    # Regex patterns for different formats
    KOHYA_UNET_PATTERN = re.compile(
        r"lora_unet_"
        r"(?P<block_type>down_blocks|mid_block|up_blocks)_"
        r"(?P<block_idx>\d+)_"
        r"(?P<layer_type>attentions|resnets)_"
        r"(?P<layer_idx>\d+)"
        r"(?:_transformer_blocks_(?P<transformer_idx>\d+))?"
        r"(?:_(?P<attn_type>attn1|attn2|ff))?"
        r"_(?P<sublayer>to_q|to_k|to_v|to_out|net_\d+_proj|proj_in|proj_out|"
        r"conv1|conv2|conv_shortcut|time_emb_proj)"
        r"(?:_(?P<sublayer_idx>\d+))?"
        r"\.(?P<lora_type>lora_down|lora_up|alpha)"
        r"(?:\.weight)?$"
    )

    KOHYA_TE_PATTERN = re.compile(
        r"lora_te(?P<te_idx>1|2)?_"
        r"(?P<path>.+)"
        r"\.(?P<lora_type>lora_down|lora_up|alpha)"
        r"(?:\.weight)?$"
    )

    DIFFUSERS_PROCESSOR_PATTERN = re.compile(
        r"(?P<block_type>down_blocks|mid_block|up_blocks)\."
        r"(?P<block_idx>\d+)\."
        r"(?P<layer_type>attentions|resnets)\."
        r"(?P<layer_idx>\d+)\."
        r"(?:transformer_blocks\.(?P<transformer_idx>\d+)\.)?"
        r"(?:(?P<attn_type>attn1|attn2|ff)\.)?"
        r"(?P<sublayer>to_q|to_k|to_v|to_out|net\.0\.proj|net\.2|proj_in|proj_out)"
        r"\.processor\."
        r"(?P<lora_name>to_q_lora|to_k_lora|to_v_lora|to_out_lora)"
        r"\.(?P<lora_type>down|up)"
        r"(?:\.weight)?$"
    )

    DIFFUSERS_NATIVE_PATTERN = re.compile(
        r"(?:unet\.)?"
        r"(?P<block_type>down_blocks|mid_block|up_blocks)\."
        r"(?P<block_idx>\d+)\."
        r"(?P<layer_type>attentions|resnets)\."
        r"(?P<layer_idx>\d+)\."
        r"(?:transformer_blocks\.(?P<transformer_idx>\d+)\.)?"
        r"(?:(?P<attn_type>attn1|attn2|ff)\.)?"
        r"(?P<sublayer>to_q|to_k|to_v|to_out|net\.0\.proj|net\.2)"
        r"\.lora_(?P<lora_type>A|B)"
        r"(?:\.weight)?$"
    )

    @classmethod
    def detect_format(cls, keys: list[str]) -> str:
        """Detect the format of LoRA keys."""
        if not keys:
            return "unknown"

        sample_keys = keys[:10]  # Check first 10 keys

        kohya_count = sum(1 for k in sample_keys if k.startswith("lora_"))
        diffusers_count = sum(1 for k in sample_keys if ".lora_A" in k or ".lora_B" in k)
        processor_count = sum(1 for k in sample_keys if ".processor." in k)

        if kohya_count > len(sample_keys) / 2:
            return "kohya"
        elif processor_count > len(sample_keys) / 2:
            return "diffusers_processor"
        elif diffusers_count > len(sample_keys) / 2:
            return "diffusers_native"
        else:
            return "unknown"

    @classmethod
    def normalize(cls, key: str, source_format: Optional[str] = None) -> Optional[NormalizedKey]:
        """
        Normalize a LoRA key to NormalizedKey.

        Args:
            key: Original LoRA key string
            source_format: Optional format hint ("kohya", "diffusers_processor", etc.)

        Returns:
            NormalizedKey or None if parsing failed
        """
        if source_format is None:
            source_format = cls.detect_format([key])

        # Try each parser in order of likelihood
        parsers = [
            (cls._parse_kohya_unet, "kohya"),
            (cls._parse_kohya_te, "kohya"),
            (cls._parse_diffusers_processor, "diffusers_processor"),
            (cls._parse_diffusers_native, "diffusers_native"),
        ]

        for parser, fmt in parsers:
            if source_format != "unknown" and fmt != source_format:
                continue
            result = parser(key)
            if result is not None:
                return result

        # Fallback: try all parsers
        for parser, _ in parsers:
            result = parser(key)
            if result is not None:
                return result

        logger.debug(f"Could not normalize key: {key}")
        return None

    @classmethod
    def _parse_kohya_unet(cls, key: str) -> Optional[NormalizedKey]:
        """Parse Kohya UNet format."""
        match = cls.KOHYA_UNET_PATTERN.match(key)
        if not match:
            return None

        groups = match.groupdict()

        # Normalize block type
        block_type = groups["block_type"].replace("_blocks", "").replace("_block", "")

        # Normalize layer type
        layer_type = groups["layer_type"].rstrip("s")  # attentions -> attention
        if groups.get("attn_type"):
            layer_type = groups["attn_type"]

        # Normalize sublayer
        sublayer = groups["sublayer"].replace("_", ".")
        if groups.get("sublayer_idx"):
            sublayer = f"{sublayer}.{groups['sublayer_idx']}"

        # Normalize lora type
        lora_type = groups["lora_type"].replace("lora_", "")

        return NormalizedKey(
            component="unet",
            block_type=block_type,
            block_idx=int(groups["block_idx"]),
            layer_type=layer_type,
            layer_idx=int(groups["layer_idx"]) if groups.get("layer_idx") else None,
            sublayer=sublayer,
            lora_type=lora_type,
        )

    @classmethod
    def _parse_kohya_te(cls, key: str) -> Optional[NormalizedKey]:
        """Parse Kohya text encoder format."""
        match = cls.KOHYA_TE_PATTERN.match(key)
        if not match:
            return None

        groups = match.groupdict()

        te_idx = groups.get("te_idx")
        component = "text_encoder" if not te_idx or te_idx == "1" else "text_encoder_2"

        # Simplified text encoder parsing
        path = groups["path"]
        lora_type = groups["lora_type"].replace("lora_", "")

        return NormalizedKey(
            component=component,
            block_type="encoder",
            block_idx=0,
            layer_type="mlp",
            layer_idx=None,
            sublayer=path.replace("_", "."),
            lora_type=lora_type,
        )

    @classmethod
    def _parse_diffusers_processor(cls, key: str) -> Optional[NormalizedKey]:
        """Parse diffusers processor format (from gradient analyzer)."""
        match = cls.DIFFUSERS_PROCESSOR_PATTERN.match(key)
        if not match:
            return None

        groups = match.groupdict()

        block_type = groups["block_type"].replace("_blocks", "").replace("_block", "")
        layer_type = groups["layer_type"].rstrip("s")
        if groups.get("attn_type"):
            layer_type = groups["attn_type"]

        sublayer = groups["sublayer"]
        lora_type = groups["lora_type"]

        return NormalizedKey(
            component="unet",
            block_type=block_type,
            block_idx=int(groups["block_idx"]),
            layer_type=layer_type,
            layer_idx=int(groups["layer_idx"]) if groups.get("layer_idx") else None,
            sublayer=sublayer,
            lora_type=lora_type,
        )

    @classmethod
    def _parse_diffusers_native(cls, key: str) -> Optional[NormalizedKey]:
        """Parse native diffusers LoRA format."""
        match = cls.DIFFUSERS_NATIVE_PATTERN.match(key)
        if not match:
            return None

        groups = match.groupdict()

        block_type = groups["block_type"].replace("_blocks", "").replace("_block", "")
        layer_type = groups["layer_type"].rstrip("s")
        if groups.get("attn_type"):
            layer_type = groups["attn_type"]

        # Convert A/B to down/up
        lora_type = "down" if groups["lora_type"] == "A" else "up"

        return NormalizedKey(
            component="unet",
            block_type=block_type,
            block_idx=int(groups["block_idx"]),
            layer_type=layer_type,
            layer_idx=int(groups["layer_idx"]) if groups.get("layer_idx") else None,
            sublayer=groups["sublayer"],
            lora_type=lora_type,
        )


def build_base_key_index(
        keys: list[str],
        source_format: Optional[str] = None,
) -> Dict[str, list[str]]:
    """
    Build an index mapping base keys to original keys.

    This allows efficient matching between different LoRA formats.

    Args:
        keys: List of original LoRA keys
        source_format: Optional format hint

    Returns:
        Dict mapping base_key -> [original_key1, original_key2, ...]
    """
    if source_format is None:
        source_format = LoRAKeyNormalizer.detect_format(keys)

    index: Dict[str, list[str]] = {}

    for key in keys:
        normalized = LoRAKeyNormalizer.normalize(key, source_format)
        if normalized:
            base_key = normalized.to_base_key()
            if base_key not in index:
                index[base_key] = []
            index[base_key].append(key)
        else:
            # Fallback: use simplified base key extraction
            base_key = _extract_simple_base_key(key)
            if base_key not in index:
                index[base_key] = []
            index[base_key].append(key)

    return index


def _extract_simple_base_key(key: str) -> str:
    """
    Fallback: Extract a simple base key for matching.

    Removes common prefixes/suffixes to get core layer identifier.
    """
    # Remove common prefixes
    for prefix in ["lora_unet_", "lora_te_", "lora_te1_", "lora_te2_", "unet."]:
        if key.startswith(prefix):
            key = key[len(prefix):]
            break

    # Remove common suffixes
    for suffix in [".lora_down.weight", ".lora_up.weight", ".lora_A.weight",
                   ".lora_B.weight", ".alpha", ".weight",
                   ".processor.to_q_lora.down", ".processor.to_q_lora.up",
                   ".processor.to_k_lora.down", ".processor.to_k_lora.up",
                   ".processor.to_v_lora.down", ".processor.to_v_lora.up",
                   ".processor.to_out_lora.down", ".processor.to_out_lora.up"]:
        if key.endswith(suffix):
            key = key[:-len(suffix)]
            break

    return key