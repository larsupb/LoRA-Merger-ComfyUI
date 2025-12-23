"""
Layer filtering utilities for LoRA Power-Merger.

Provides LayerFilter class for selective layer merging operations.
"""

from typing import Set, Optional, Dict, Any
from ..types import LORA_KEY_DICT, LayerComponentSet
from .config import (
    ATTENTION_LAYERS,
    MLP_LAYERS,
    ATTENTION_MLP_LAYERS,
)


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

    def apply(self, patch_dict: LORA_KEY_DICT) -> LORA_KEY_DICT:
        """
        Apply filter to patch dictionary.

        Args:
            patch_dict: Dictionary of layer key -> LoRAAdapter

        Returns:
            Filtered patch dictionary

        Example:
            >>> filter = LayerFilter("attn-only")
            >>> filtered = filter.apply(lora_patches)
        """
        if self.filter_set is None:
            # No filtering
            return patch_dict

        original_count = len(patch_dict)

        # Filter to only include keys that contain any of the filter components
        filtered_dict = {
            key: value
            for key, value in patch_dict.items()
            if any(component in key for component in self.filter_set)
        }

        filtered_count = original_count - len(filtered_dict)

        if filtered_count > 0:
            import logging
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

    def matches(self, layer_key: str) -> bool:
        """
        Check if a layer key matches the filter.

        Args:
            layer_key: Layer key to check

        Returns:
            True if layer matches filter (or no filter), False otherwise
        """
        if self.filter_set is None:
            return True

        return any(component in layer_key for component in self.filter_set)
