"""
Unified key normalization for LoRA merging.

Converts both Kohya keys (from safetensors) and Diffusers keys (from gradient analyzer)
to a common normalized format for matching.
"""

import re
import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class NormalizedKey:
    """Normalized representation of a LoRA layer location."""
    component: str  # "unet" or "text_encoder"
    block_type: str  # "down", "mid", "up"
    block_idx: int  # Block index
    attention_idx: int  # Attention layer within block
    transformer_idx: int  # Transformer block index
    attn_type: Optional[str]  # "self", "cross", or None
    sublayer: str  # Normalized: "to_q", "to_k", "to_v", "to_out", "ff.0", "ff.2"

    def __str__(self) -> str:
        parts = [self.component, self.block_type, str(self.block_idx),
                 f"attn{self.attention_idx}", f"tb{self.transformer_idx}"]
        if self.attn_type:
            parts.append(self.attn_type)
        parts.append(self.sublayer)
        return ".".join(parts)

    def __hash__(self):
        return hash((self.component, self.block_type, self.block_idx,
                     self.attention_idx, self.transformer_idx, self.attn_type,
                     self.sublayer))

    def __eq__(self, other):
        if not isinstance(other, NormalizedKey):
            return False
        return (self.component == other.component and
                self.block_type == other.block_type and
                self.block_idx == other.block_idx and
                self.attention_idx == other.attention_idx and
                self.transformer_idx == other.transformer_idx and
                self.attn_type == other.attn_type and
                self.sublayer == other.sublayer)


def _normalize_sublayer(sublayer: str) -> str:
    """
    Normalize sublayer names to a consistent format.

    Handles variations like:
    - attn1_to_q, attn2_to_k → "to_q", "to_k" (strip attn prefix)
    - to_out_0, to_out.0, to_out → "to_out" (strip index suffix)
    - ff_net_0_proj, ff.net.0.proj → "ff.0"
    - ff_net_2, ff.net.2 → "ff.2"

    """
    sl = sublayer.lower()

    # Step 1: Strip attn1_ or attn2_ prefix if present
    sl = re.sub(r"^attn[12][_.]?", "", sl)

    # Step 2: Normalize separators
    sl = sl.replace("_", ".")

    # Step 3: Handle to_out (remove trailing index like .0)
    if sl.startswith("to.out"):
        return "to_out"

    # Step 4: Normalize ff layer names
    if "ff" in sl:
        if "proj" in sl or ".0." in sl or sl.endswith(".0"):
            return "ff.0"
        elif ".2" in sl or sl.endswith("2"):
            return "ff.2"
        else:
            return "ff"

    # Step 5: Standard attention sublayers - normalize to underscore format
    if sl in ["to.q", "toq"]:
        return "to_q"
    if sl in ["to.k", "tok"]:
        return "to_k"
    if sl in ["to.v", "tov"]:
        return "to_v"
    if sl.startswith("to.out"):
        return "to_out"

    # Step 6: Projection layers
    if sl in ["proj.in", "projin"]:
        return "proj_in"
    if sl in ["proj.out", "projout"]:
        return "proj_out"

    # Fallback: replace dots with underscores for consistency
    return sl.replace(".", "_")


def normalize_kohya_key(key: str) -> Optional[NormalizedKey]:
    """
    Normalize a Kohya-format LoRA key.

    Example: lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q.lora_down.weight
    """
    key_lower = key.lower()

    # Skip non-unet keys
    if not key_lower.startswith("lora_unet_"):
        return None

    # Remove prefix and suffix
    core = key_lower.replace("lora_unet_", "")
    core = re.sub(r"\.(lora_down|lora_up|alpha)(\.weight)?$", "", core)

    # Parse components
    block_type = None
    block_idx = 0
    attention_idx = 0
    transformer_idx = 0
    attn_type = None
    sublayer = None

    # Block type and index
    if "down_blocks_" in core:
        block_type = "down"
        match = re.search(r"down_blocks_(\d+)", core)
        if match:
            block_idx = int(match.group(1))
    elif "up_blocks_" in core:
        block_type = "up"
        match = re.search(r"up_blocks_(\d+)", core)
        if match:
            block_idx = int(match.group(1))
    elif "mid_block" in core:
        block_type = "mid"
        block_idx = 0
    else:
        return None

    # Attention index
    attn_match = re.search(r"attentions_(\d+)", core)
    if attn_match:
        attention_idx = int(attn_match.group(1))

    # Transformer block index
    tb_match = re.search(r"transformer_blocks_(\d+)", core)
    if tb_match:
        transformer_idx = int(tb_match.group(1))

    # Attention type - extract BEFORE sublayer extraction
    if "_attn1_" in core or "_attn1" in core:
        attn_type = "self"
    elif "_attn2_" in core or "_attn2" in core:
        attn_type = "cross"

    # Sublayer extraction - find the last meaningful component
    # Pattern: ...transformer_blocks_0_attn1_to_q or ...transformer_blocks_0_ff_net_0_proj

    # Try to extract sublayer after attn1/attn2
    attn_sublayer_match = re.search(r"_attn[12]_(.+)$", core)
    if attn_sublayer_match:
        raw_sublayer = attn_sublayer_match.group(1)
        sublayer = _normalize_sublayer(raw_sublayer)
    else:
        # Try ff layer pattern (not inside attention)
        ff_match = re.search(r"_(ff_.+)$", core)
        if ff_match:
            sublayer = _normalize_sublayer(ff_match.group(1))
        else:
            # Try proj_in/proj_out (standalone)
            proj_match = re.search(r"_(proj_(?:in|out))$", core)
            if proj_match:
                sublayer = _normalize_sublayer(proj_match.group(1))

    if sublayer is None:
        logger.debug(f"Could not extract sublayer from Kohya key: {key} (core: {core})")
        return None

    return NormalizedKey(
        component="unet",
        block_type=block_type,
        block_idx=block_idx,
        attention_idx=attention_idx,
        transformer_idx=transformer_idx,
        attn_type=attn_type,
        sublayer=sublayer,
    )


def normalize_diffusers_key(key: str) -> Optional[NormalizedKey]:
    """
    Normalize a Diffusers-format key (from gradient analyzer).

    Examples:
    - down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.lora_A.LoraName.weight
    - down_blocks.0.attentions.0.transformer_blocks.0.attn1.processor.to_q_lora.down.weight

    """
    key_lower = key.lower()

    # Remove common prefixes
    core = key_lower
    for prefix in ["unet.", "text_encoder.", "text_encoder_2."]:
        if core.startswith(prefix):
            core = core[len(prefix):]
            break

    # Remove LoRA-specific suffixes
    # Handle processor format: .processor.to_q_lora.down.weight
    processor_match = re.search(r"\.processor\.(to_[qkvo][^.]*?)_lora\.(down|up)(\.weight)?$", core)
    if processor_match:
        # Extract sublayer from processor format and clean up
        extracted_sublayer = processor_match.group(1)  # e.g., "to_q" or "to_out"
        core = re.sub(r"\.processor\.to_[qkvo][^.]*?_lora\.(down|up)(\.weight)?$", f".{extracted_sublayer}", core)

    # Handle standard format: .lora_A.LoraName.weight or .lora_A.weight
    core = re.sub(r"\.lora_(a|b|down|up)\.[^.]+\.weight$", "", core)
    core = re.sub(r"\.lora_(a|b|down|up)\.weight$", "", core)
    core = re.sub(r"\.lora_(a|b|down|up)$", "", core)
    core = re.sub(r"\.weight$", "", core)

    block_type = None
    block_idx = 0
    attention_idx = 0
    transformer_idx = 0
    attn_type = None
    sublayer = None

    # Block type and index
    if "down_blocks" in core:
        block_type = "down"
        match = re.search(r"down_blocks\.(\d+)", core)
        if match:
            block_idx = int(match.group(1))
    elif "up_blocks" in core:
        block_type = "up"
        match = re.search(r"up_blocks\.(\d+)", core)
        if match:
            block_idx = int(match.group(1))
    elif "mid_block" in core:
        block_type = "mid"
        block_idx = 0
    else:
        return None

    # Attention index
    attn_match = re.search(r"attentions\.(\d+)", core)
    if attn_match:
        attention_idx = int(attn_match.group(1))

    # Transformer block index
    tb_match = re.search(r"transformer_blocks\.(\d+)", core)
    if tb_match:
        transformer_idx = int(tb_match.group(1))

    # Attention type
    if ".attn1" in core:
        attn_type = "self"
    elif ".attn2" in core:
        attn_type = "cross"

    # Sublayer extraction
    # Pattern 1: .attn1.to_q or .attn2.to_k
    attn_sublayer_match = re.search(r"\.attn[12]\.(to_[qkv]|to_out(?:\.\d+)?)", core)
    if attn_sublayer_match:
        sublayer = _normalize_sublayer(attn_sublayer_match.group(1))
    else:
        # Pattern 2: FF layers
        if ".ff." in core:
            if ".net.0.proj" in core:
                sublayer = "ff.0"
            elif ".net.2" in core:
                sublayer = "ff.2"
            else:
                sublayer = "ff"
        # Pattern 3: Projection layers
        elif ".proj_in" in core or "proj_in" in core:
            sublayer = "proj_in"
        elif ".proj_out" in core or "proj_out" in core:
            sublayer = "proj_out"

    if sublayer is None:
        logger.debug(f"Could not extract sublayer from diffusers key: {key} (core: {core})")
        return None

    return NormalizedKey(
        component="unet",
        block_type=block_type,
        block_idx=block_idx,
        attention_idx=attention_idx,
        transformer_idx=transformer_idx,
        attn_type=attn_type,
        sublayer=sublayer,
    )


def _extract_sublayer_from_diffusers(core: str) -> Optional[str]:
    """
    Extract and normalize sublayer from diffusers key.

    Handles:
    - attn1.to_q → to_q
    - attn1.to_out.0 → to_out
    - ff.net.0.proj → ff.0
    - ff.net.2 → ff.2
    - proj_in → proj_in
    - proj_out → proj_out

    """
    # Pattern 1: Attention sublayers (to_q, to_k, to_v, to_out)
    attn_sublayer_match = re.search(r"\.attn[12]\.(to_[qkv]|to_out)", core)
    if attn_sublayer_match:
        return _normalize_sublayer(attn_sublayer_match.group(1))

    # Pattern 2: FF layers
    if ".ff." in core or ".ff" in core:
        if ".net.0.proj" in core or "net_0_proj" in core:
            return "ff.0"
        elif ".net.2" in core or "net_2" in core:
            return "ff.2"
        else:
            return "ff"

    # Pattern 3: Projection layers (not inside attention)
    if "proj_in" in core:
        return "proj_in"
    if "proj_out" in core:
        return "proj_out"

    # Pattern 4: Fallback - try to find any known sublayer name at the end
    for sl in ["to_q", "to_k", "to_v", "to_out"]:
        if core.endswith(f".{sl}") or core.endswith(f"_{sl}"):
            return _normalize_sublayer(sl)

    return None


def build_normalized_index(
        keys: list[str],
        key_format: str = "auto",
) -> Dict[NormalizedKey, list[str]]:
    """
    Build index mapping normalized keys to original keys.

    Args:
        keys: List of original keys
        key_format: "kohya", "diffusers", or "auto"

    Returns:
        Dict mapping NormalizedKey -> list of original keys
    """
    if key_format == "auto":
        # Detect format
        if any(k.startswith("lora_unet_") for k in keys[:10]):
            key_format = "kohya"
        else:
            key_format = "diffusers"

    normalize_fn = normalize_kohya_key if key_format == "kohya" else normalize_diffusers_key

    index: Dict[NormalizedKey, list[str]] = {}

    for key in keys:
        normalized = normalize_fn(key)
        if normalized:
            if normalized not in index:
                index[normalized] = []
            index[normalized].append(key)

    return index