"""
Semantic LoRA merging nodes.

Enables text-guided, feature-aware LoRA merging using semantic importance maps.
"""

import logging
from typing import Dict, List, Tuple, Any, Set

import torch
import folder_paths
from comfy.weight_adapter import LoRAAdapter

from .types import LORA_STACK
from .architectures.sd_lora import build_lora_to_unet_key_map, invert_lora_to_unet_key_map

logger = logging.getLogger(__name__)


class PMSemanticMergeSpec:
    """
    Create a semantic merge specification from text description.

    Allows users to specify which features should come from which LoRAs using
    natural text format like: "hair from lora1, clothing from lora2"
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "specification": (
                    "STRING",
                    {
                        "default": "hair from lora1, eyes from lora1, clothing from lora2",
                        "multiline": True,
                    },
                ),
            },
        }

    RETURN_TYPES = ("MergeSpec", "STRING")
    RETURN_NAMES = ("merge_spec", "parsed_spec")
    FUNCTION = "create_spec"
    CATEGORY = "LoRA PowerMerge/Semantic"
    DESCRIPTION = "Parse text specification into semantic merge configuration"

    def create_spec(self, specification: str) -> Tuple[Dict[str, Any], str]:
        """
        Parse merge specification text.

        Format:
            "feature from lora_name [weight], feature2 from lora2 [weight], ..."

        Examples:
            "hair from character1, clothing from character2"
            "hair from lora1 1.0, eyes from lora1 0.8, clothing from lora2 1.2"

        Returns:
            (merge_spec_dict, formatted_description)
        """
        merge_spec = {}
        lines = []

        # Parse each feature assignment
        for assignment in specification.split(","):
            assignment = assignment.strip()
            if not assignment:
                continue

            # Parse "feature from lora [weight]"
            parts = assignment.split()

            if len(parts) < 3 or parts[1].lower() != "from":
                logger.warning(f"Skipping invalid assignment: {assignment}")
                continue

            feature = parts[0].lower()
            lora_name = parts[2]

            # Check for optional weight
            weight = 1.0
            if len(parts) >= 4:
                try:
                    weight = float(parts[3])
                except ValueError:
                    logger.warning(f"Invalid weight in '{assignment}', using 1.0")

            merge_spec[feature] = {
                "source": lora_name,
                "weight": weight,
            }

            lines.append(f"  {feature}: {lora_name} (weight={weight:.2f})")

        # Format description
        description = "Semantic merge specification:\n" + "\n".join(lines)

        if not merge_spec:
            raise ValueError("No valid feature assignments found in specification")

        logger.info(f"Parsed merge spec: {merge_spec}")

        return (merge_spec, description)


class PMSemanticMerger:
    """
    Merge LoRAs using semantic importance maps.

    This node performs feature-aware merging, where different features
    (hair, clothing, etc.) are preferentially taken from different source LoRAs.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "Model for building LoRA→UNet key mapping"}),
                "lora_stack": ("LoRAStack",),
                "semantic_maps": ("SemanticMaps",),
                "merge_spec": ("MergeSpec",),
            },
            "optional": {
                "lambda_value": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0}),
            },
        }

    RETURN_TYPES = ("LoRABundle",)
    FUNCTION = "merge_semantic"
    CATEGORY = "LoRA PowerMerge/Semantic"
    DESCRIPTION = "Merge LoRAs with semantic feature routing"

    def merge_semantic(
        self,
        model: Any,  # ComfyUI MODEL (ModelPatcher)
        lora_stack: LORA_STACK,
        semantic_maps: Dict[str, Dict[str, Dict[str, torch.Tensor]]],
        merge_spec: Dict[str, Any],
        lambda_value: float = 1.0,
    ) -> Tuple[Dict[str, Any]]:
        """
        Perform semantic merge.

        Args:
            model: ComfyUI MODEL for building LoRA→UNet key mapping
            lora_stack: Stack of LoRAs to merge
            semantic_maps: Semantic importance maps for each LoRA
                Format: {lora_name: {feature: {diffusers_key: importance_tensor}}}
                Note: semantic_maps now use diffusers keys (e.g., "down_blocks.0.attentions.0.to_q.processor.to_q_lora.down.weight")
            merge_spec: Feature assignment specification
                Format: {feature: {"source": lora_name, "weight": float}}
            lambda_value: Global scaling factor

        Returns:
            Merged LoRABundle
        """
        # Build key mapping: LoRA keys → diffusers keys
        # The semantic maps use diffusers keys, but lora_stack uses LoRA file keys
        forward_map = build_lora_to_unet_key_map(model)
        inverse_map = invert_lora_to_unet_key_map(forward_map)

        logger.info(f"Built key mapping with {len(forward_map)} forward entries, {len(inverse_map)} inverse entries")

        # Get list of LoRA names
        lora_names = list(lora_stack.keys())
        logger.info(f"Merging {len(lora_names)} LoRAs with semantic routing")

        # Validate semantic_maps match lora_stack
        for lora_name in lora_names:
            if lora_name not in semantic_maps:
                raise ValueError(
                    f"LoRA '{lora_name}' in stack but no semantic map found. "
                    f"Available maps: {list(semantic_maps.keys())}"
                )

        # Collect all layer keys
        all_layer_keys = set()
        for lora_name, lora_entry in lora_stack.items():
            patches = lora_entry.get("patches", {})
            all_layer_keys.update(patches.keys())

        # Initialize merged result
        merged_patches = {}

        # Process each layer
        # Handle mixed key types (tuples and strings) in ComfyUI
        for layer_key in sorted(all_layer_keys, key=str):
            # Collect weights for this layer from each LoRA
            layer_tensors = {}
            for lora_name in lora_names:
                patches = lora_stack[lora_name].get("patches", {})
                if layer_key in patches:
                    layer_tensors[lora_name] = patches[layer_key]

            if not layer_tensors:
                continue

            # Compute feature-weighted merge for this layer
            # layer_tensors contains LoRAAdapter objects
            merged_adapter = self._merge_layer_semantic(
                layer_tensors,
                semantic_maps,
                merge_spec,
                layer_key,
                lambda_value,
                inverse_map,
            )

            merged_patches[layer_key] = merged_adapter

        # Package as LoRABundle
        merged_lora = {
            "name": f"semantic_merge_{len(lora_names)}loras",
            "lora": merged_patches,
            "strength_model": 1.0,
        }

        logger.info(f"Semantic merge complete: {len(merged_patches)} layers")

        return (merged_lora,)

    def _merge_layer_semantic(
        self,
        layer_adapters: Dict[str, LoRAAdapter],
        semantic_maps: Dict[str, Dict[str, Dict[str, torch.Tensor]]],
        merge_spec: Dict[str, Any],
        layer_key: str,
        lambda_value: float,
        inverse_map: Dict[str, Set[str]],
    ) -> LoRAAdapter:
        """
        Merge a single layer using semantic routing.

        For each feature in merge_spec:
        - Look up which LoRA it should come from
        - Look up importance score for that feature in that LoRA
        - Weight the LoRA's contribution by importance

        Final merged weight = sum over features of (importance * lora_weight * feature_weight)

        Args:
            layer_adapters: Dict mapping LoRA names to LoRAAdapter objects for this layer
            semantic_maps: Semantic importance maps
            merge_spec: Feature assignment specification
            layer_key: Current layer key
            lambda_value: Global scaling factor

        Returns:
            LoRAAdapter object with merged weights
        """
        # Extract weights from LoRAAdapter objects
        # LoRAAdapter.weights is a tuple: (up, down, alpha, dora_scale, None, None)
        lora_weights = {}
        for lora_name, adapter in layer_adapters.items():
            up, down, alpha, *rest = adapter.weights
            lora_weights[lora_name] = {
                "up": up,
                "down": down,
                "alpha": alpha,
            }

        # Merge up, down, alpha separately using semantic importance
        for sublayer_name in ["up", "down"]:
            merged_tensor = None
            total_weight = None

            for feature, spec in merge_spec.items():
                source_lora = spec["source"]
                feature_weight = spec["weight"]

                # Check if this source LoRA is available
                if source_lora not in lora_weights:
                    logger.warning(
                        f"Feature '{feature}' specifies LoRA '{source_lora}' "
                        f"but it's not in the stack for layer {layer_key}"
                    )
                    continue

                # Get the tensor from this LoRA
                lora_tensor = lora_weights[source_lora][sublayer_name]

                # Check shape compatibility with existing merged_tensor
                if merged_tensor is not None and lora_tensor.shape != merged_tensor.shape:
                    logger.warning(
                        f"Skipping {source_lora} for {layer_key}/{sublayer_name}: "
                        f"shape mismatch {lora_tensor.shape} vs {merged_tensor.shape} "
                        f"(likely different LoRA ranks)"
                    )
                    continue

                # Get importance scores for this feature from this LoRA
                # Convert LoRA key to diffusers key for semantic map lookup
                importance = self._get_importance_score(
                    semantic_maps, source_lora, feature, layer_key, lora_tensor, inverse_map
                )

                # Compute weighted contribution
                # contribution = importance × lora_tensor × feature_weight
                if isinstance(importance, torch.Tensor):
                    weighted_contribution = importance * lora_tensor * feature_weight
                else:
                    weighted_contribution = lora_tensor * feature_weight * importance

                # Accumulate
                if merged_tensor is None:
                    merged_tensor = weighted_contribution
                    if isinstance(importance, torch.Tensor):
                        total_weight = importance * feature_weight
                    else:
                        total_weight = feature_weight * importance
                else:
                    merged_tensor = merged_tensor + weighted_contribution
                    if isinstance(importance, torch.Tensor):
                        total_weight = total_weight + (importance * feature_weight)
                    else:
                        total_weight = total_weight + (feature_weight * importance)

            # Normalize by total weight and apply lambda
            if merged_tensor is not None:
                if isinstance(total_weight, torch.Tensor):
                    total_weight = torch.clamp(total_weight, min=1e-8)
                    result = (merged_tensor / total_weight) * lambda_value
                else:
                    result = (merged_tensor / max(total_weight, 1e-8)) * lambda_value

                if sublayer_name == "up":
                    merged_up = result
                else:
                    merged_down = result
            else:
                # Fallback to simple average if no merge spec matched
                logger.warning(
                    f"No features matched for {layer_key}/{sublayer_name}, using average"
                )
                tensors = [w[sublayer_name] for w in lora_weights.values()]

                # Handle rank mismatches - filter to compatible tensors
                compatible_tensors = self._filter_compatible_tensors(tensors, layer_key, sublayer_name)

                if compatible_tensors:
                    result = torch.stack(compatible_tensors).mean(dim=0) * lambda_value
                else:
                    # If no compatible tensors, use first one
                    logger.warning(f"No compatible tensors for {layer_key}/{sublayer_name}, using first LoRA")
                    result = tensors[0] * lambda_value

                if sublayer_name == "up":
                    merged_up = result
                else:
                    merged_down = result

        # Merge alpha (simple average)
        alphas = [w["alpha"] for w in lora_weights.values() if w["alpha"] is not None]
        if alphas:
            # Alpha should be a scalar or small tensor, just average it
            if isinstance(alphas[0], torch.Tensor):
                merged_alpha = torch.stack(alphas).mean(dim=0)
            else:
                merged_alpha = sum(alphas) / len(alphas)
        else:
            merged_alpha = None

        # Create LoRAAdapter with merged weights
        # LoRAAdapter expects weights tuple: (up, down, alpha, dora_scale, None, None)
        merged_weights = (merged_up, merged_down, merged_alpha, None, None, None)

        # Get loaded_keys from first adapter (they should all be the same)
        first_adapter = list(layer_adapters.values())[0]
        loaded_keys = first_adapter.loaded_keys if hasattr(first_adapter, 'loaded_keys') else set()

        return LoRAAdapter(weights=merged_weights, loaded_keys=loaded_keys)

    def _get_importance_score(
        self,
        semantic_maps: Dict,
        source_lora: str,
        feature: str,
        layer_key: str,
        lora_tensor: torch.Tensor,
        inverse_map: Dict[str, Set[str]],
    ) -> float | torch.Tensor:
        """
        Get importance score for a feature, with fallback to uniform importance.

        Converts LoRA file keys to diffusers keys using inverse_map to lookup
        in the semantic map (which uses diffusers keys).
        """
        # Get importance scores for this feature from this LoRA
        if source_lora not in semantic_maps:
            return 1.0

        lora_semantic_map = semantic_maps[source_lora]
        if feature not in lora_semantic_map:
            return 1.0

        feature_map = lora_semantic_map[feature]

        # Convert layer_key (LoRA format) to diffusers keys using inverse_map
        # layer_key might be a tuple (ComfyUI format), convert to string
        layer_key_str = layer_key[0] if isinstance(layer_key, tuple) else str(layer_key)

        # The inverse_map maps from UNet keys (diffusion_model.*) to LoRA keys
        # We need to find diffusers keys that map to this UNet key
        # But semantic_maps use diffusers processor keys, not UNet keys

        # Try to find matching diffusers keys in the feature_map
        # The semantic map uses diffusers processor format like:
        # "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.processor.to_q_lora.down.weight"

        for diffusers_key in feature_map.keys():
            # Check if this diffusers key corresponds to our layer_key
            # by checking if it's in the inverse_map aliases for the same UNet parameter
            if self._keys_match(layer_key_str, diffusers_key, inverse_map):
                importance = feature_map[diffusers_key]

                # Ensure importance tensor matches lora tensor shape
                if isinstance(importance, torch.Tensor) and importance.shape != lora_tensor.shape:
                    logger.warning(
                        f"Shape mismatch for {source_lora}/{feature}/{diffusers_key}: "
                        f"importance {importance.shape} vs tensor {lora_tensor.shape}"
                    )
                    # Use scalar importance (average)
                    return importance.mean().item()

                return importance

        # Fallback to uniform importance
        logger.debug(f"No matching diffusers key found for LoRA key: {layer_key_str}")
        return 1.0

    def _keys_match(self, lora_key: str, diffusers_key: str, inverse_map: Dict[str, Set[str]]) -> bool:
        """
        Check if a LoRA key and diffusers key refer to the same parameter.

        Uses the inverse_map to check if both keys map to the same UNet weight.
        """
        # Find the UNet key that this LoRA key maps to
        unet_key_from_lora = None
        for unet_key, lora_aliases in inverse_map.items():
            if any(alias in lora_key for alias in lora_aliases):
                unet_key_from_lora = unet_key
                break

        if not unet_key_from_lora:
            return False

        # Check if the diffusers key also maps to the same UNet key
        for unet_key, lora_aliases in inverse_map.items():
            if unet_key == unet_key_from_lora:
                # Check if diffusers_key matches any of the aliases
                for alias in lora_aliases:
                    if alias in diffusers_key:
                        return True

        return False

    def _filter_compatible_tensors(
        self,
        tensors: List[torch.Tensor],
        layer_key: str,
        sublayer_name: str,
    ) -> List[torch.Tensor]:
        """
        Filter tensors to only include those with compatible shapes.

        When LoRAs have different ranks, their tensors will have different shapes.
        This method filters to the most common shape.

        Args:
            tensors: List of tensors to filter
            layer_key: Current layer key (for logging)
            sublayer_name: Sublayer name (for logging)

        Returns:
            List of tensors with the most common shape
        """
        if not tensors:
            return []

        # Count shapes
        shape_counts = {}
        for tensor in tensors:
            shape = tuple(tensor.shape)
            shape_counts[shape] = shape_counts.get(shape, 0) + 1

        # Find most common shape
        most_common_shape = max(shape_counts.keys(), key=lambda s: shape_counts[s])

        # Filter to most common shape
        compatible = [t for t in tensors if tuple(t.shape) == most_common_shape]

        # Warn about incompatible tensors
        num_incompatible = len(tensors) - len(compatible)
        if num_incompatible > 0:
            logger.warning(
                f"Layer {layer_key}/{sublayer_name}: {num_incompatible} LoRAs have "
                f"incompatible shapes (different ranks). Using {len(compatible)} LoRAs "
                f"with shape {most_common_shape}. This happens when merging LoRAs "
                f"trained with different rank settings."
            )

        return compatible


# ComfyUI node registration exports
NODE_CLASS_MAPPINGS = {
    "PM Semantic Merge Spec": PMSemanticMergeSpec,
    "PM Semantic Merger": PMSemanticMerger,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PM Semantic Merge Spec": "PM Semantic Merge Spec",
    "PM Semantic Merger": "PM Semantic Merger",
}
