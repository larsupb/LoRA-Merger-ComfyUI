import re
from typing import Any


def detect_block_names(layer_key) -> dict[str, str | Any] | None:
    # With transformer_blocks
    exp_with_transformer = re.compile(r"""
        (?:diffusion_model\.)?                            # optional prefix
        (?:blocks\.)?        
        (?P<block_idx>\d+)
        \.        
        (?P<component>self_attn|cross_attn|ffn)           # top-level component
        (?:\..+)?                                         # allow nested submodules (e.g. .to_q.weight)        
    """, re.VERBOSE)

    for exp in [exp_with_transformer]:
        match = exp.search(layer_key)
        if match:
            out = {
                "block_idx": match.group("block_idx"),
                "component": match.group("component"),
                "main_block": f"blocks.{match.group("block_idx")}",
            }
            return out
    return None