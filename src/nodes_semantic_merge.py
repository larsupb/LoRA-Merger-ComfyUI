"""
Semantic LoRA merging nodes with statistics collection.
"""

import logging
from typing import Dict, List, Tuple, Any, Set, Optional

import torch
import folder_paths
from comfy.weight_adapter import LoRAAdapter

from .analysis.merge_spec import MergeSpec
from .types import LORA_STACK
from .architectures.sd_lora import build_lora_to_unet_key_map, invert_lora_to_unet_key_map
from .analysis.merge_statistics import (
    MergeStatistics,
    classify_layer_type,
    compute_relative_depth,
)

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
        """Parse merge specification text."""
        try:
            merge_spec = MergeSpec.from_text(specification)
        except ValueError as e:
            raise ValueError(f"Invalid specification: {e}")

        # Format description
        lines = [
            f"  {feature}: {spec['source']} (weight={spec['weight']:.2f})"
            for feature, spec in merge_spec.feature_assignments.items()
        ]
        description = "Semantic merge specification:\n" + "\n".join(lines)

        logger.info(f"Parsed merge spec: {merge_spec.feature_assignments}")

        return merge_spec.feature_assignments, description


class PMSemanticMerger:
    """
    Merge LoRAs using semantic importance maps.

    Now includes detailed statistics collection for merge analysis.
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
                "preference_strength": (
                    "FLOAT",
                    {
                        "default": 1.5,
                        "min": 1.0,
                        "max": 5.0,
                        "step": 0.1,
                        "tooltip": "How strongly to prefer assigned LoRA (1.0 = no preference, pure semantic blend)"
                    },
                ),
                "collect_statistics": ("BOOLEAN", {"default": True, "tooltip": "Collect detailed merge statistics"}),
                "verbose_report": ("BOOLEAN", {"default": False, "tooltip": "Include per-layer detail in report"}),
            },
        }

    RETURN_TYPES = ("LoRABundle", "STRING", "MergeStats")
    RETURN_NAMES = ("merged_lora", "merge_report", "statistics")
    FUNCTION = "merge_semantic"
    CATEGORY = "LoRA PowerMerge/Semantic"
    DESCRIPTION = "Merge LoRAs with semantic feature routing and detailed statistics"

    def merge_semantic(
            self,
            model: Any,
            lora_stack: LORA_STACK,
            semantic_maps: Dict[str, Dict[str, Dict[str, torch.Tensor]]],
            merge_spec: Dict[str, Any],
            lambda_value: float = 1.0,
            preference_strength: float = 1.5,
            collect_statistics: bool = True,
            verbose_report: bool = False,
    ) -> Tuple[Dict[str, Any], str, Optional[MergeStatistics]]:
        """
        Perform semantic merge with statistics collection.

        Args:
            model: Model for building LoRA→UNet key mapping
            lora_stack: Stack of LoRAs to merge
            semantic_maps: Semantic importance maps for each LoRA
            merge_spec: Feature assignment specification
            lambda_value: Global scaling factor for merged weights
            preference_strength: How strongly to prefer assigned LoRA
                                (1.0 = pure semantic blend, >1.0 = favor assigned LoRA)
            collect_statistics: Whether to collect detailed merge statistics
            verbose_report: Include per-layer detail in report
        """
        # Build key mapping
        forward_map = build_lora_to_unet_key_map(model)
        inverse_map = invert_lora_to_unet_key_map(forward_map)

        lora_names = list(lora_stack.keys())
        logger.info(f"Merging {len(lora_names)} LoRAs with semantic routing")
        logger.info(f"Preference strength: {preference_strength}")

        # Validate semantic_maps
        for lora_name in lora_names:
            if lora_name not in semantic_maps:
                raise ValueError(
                    f"LoRA '{lora_name}' in stack but no semantic map found. "
                    f"Available maps: {list(semantic_maps.keys())}"
                )

        # Collect all layer keys and sort for depth calculation
        all_layer_keys = set()
        for lora_name, lora_entry in lora_stack.items():
            patches = lora_entry.get("patches", {})
            all_layer_keys.update(patches.keys())

        sorted_layer_keys = sorted(all_layer_keys, key=str)
        total_layers = len(sorted_layer_keys)

        # Initialize statistics
        stats = MergeStatistics(lambda_value=lambda_value) if collect_statistics else None

        # Run diagnostic if DEBUG logging enabled
        if logger.isEnabledFor(logging.DEBUG):
            diagnostic = self._diagnose_key_matching(
                semantic_maps=semantic_maps,
                layer_keys=[str(k) for k in sorted_layer_keys[:10]],
                forward_map=forward_map,
            )
            logger.debug(diagnostic)

        # Initialize merged result
        merged_patches = {}

        # Process each layer with depth tracking
        for layer_idx, layer_key in enumerate(sorted_layer_keys):
            # Compute relative depth for this layer
            relative_depth = compute_relative_depth(layer_idx, total_layers)
            layer_type = classify_layer_type(layer_key)

            # Collect weights for this layer from each LoRA
            layer_tensors = {}
            for lora_name in lora_names:
                patches = lora_stack[lora_name].get("patches", {})
                if layer_key in patches:
                    layer_tensors[lora_name] = patches[layer_key]

            if not layer_tensors:
                continue

            # Compute feature-weighted merge for this layer
            merged_adapter = self._merge_layer_semantic(
                layer_tensors=layer_tensors,
                semantic_maps=semantic_maps,
                merge_spec=merge_spec,
                layer_key=layer_key,
                lambda_value=lambda_value,
                forward_map=forward_map,
                inverse_map=inverse_map,
                statistics=stats,
                relative_depth=relative_depth,
                layer_type=layer_type,
                preference_strength=preference_strength,  # ← Pass through
            )
            merged_patches[layer_key] = merged_adapter

        # Finalize statistics
        if stats:
            stats.finalize()
            merge_report = stats.format_report(verbose=verbose_report)
            logger.info(f"Merge statistics collected for {stats.total_layers} layers")
        else:
            merge_report = "Statistics collection disabled"

        # Package as LoRABundle
        merged_lora = {
            "name": f"semantic_merge_{len(lora_names)}loras",
            "lora": merged_patches,
            "strength_model": 1.0,
        }

        logger.info(f"Semantic merge complete: {len(merged_patches)} layers")

        return merged_lora, merge_report, stats

    def _merge_layer_semantic(
            self,
            layer_tensors: Dict[str, LoRAAdapter],
            semantic_maps: Dict[str, Dict[str, Dict[str, torch.Tensor]]],
            merge_spec: Dict[str, Any],
            layer_key: str,
            lambda_value: float,
            forward_map: Dict[str, str],
            inverse_map: Dict[str, Set[str]],
            statistics: Optional[MergeStatistics],
            relative_depth: float,
            layer_type: Any,
            preference_strength: float,  # ← New parameter
    ) -> LoRAAdapter:
        """
        Merge a single layer using TRUE semantic blending.

        For each feature, ALL LoRAs contribute based on their semantic importance
        for that feature at this specific layer. The preferred LoRA (from merge_spec)
        gets a boost based on preference_strength.

        Args:
            layer_tensors: LoRA adapters for this layer keyed by LoRA name
            semantic_maps: Semantic importance maps for all LoRAs
            merge_spec: Feature assignment specification
            layer_key: Current layer identifier
            lambda_value: Global scaling factor
            inverse_map: UNet key to LoRA key mapping
            statistics: Statistics collector (optional)
            relative_depth: Relative depth of this layer (0.0-1.0)
            layer_type: Classification of layer type
            preference_strength: Boost factor for preferred LoRA (1.0 = no boost)
        """
        # Extract weights from LoRAAdapter objects
        lora_weights = {}
        for lora_name, adapter in layer_tensors.items():
            up, down, alpha, *rest = adapter.weights
            lora_weights[lora_name] = {
                "up": up,
                "down": down,
                "alpha": alpha,
            }

        all_loras = list(lora_weights.keys())

        # Merge up and down separately
        merged_up = None
        merged_down = None

        for sublayer_name in ["up", "down"]:
            # Check shape compatibility - find most common shape
            reference_shape = self._get_reference_shape(
                [lora_weights[l][sublayer_name] for l in all_loras],
                layer_key,
                sublayer_name,
            )

            if reference_shape is None:
                continue

            merged_tensor = None
            total_weight = 0.0

            # For each feature, blend ALL LoRAs based on their semantic importance
            for feature, spec in merge_spec.items():
                preferred_lora = spec["source"]
                feature_weight = spec["weight"]

                # Get importance for this feature from EACH LoRA
                feature_importances = {}
                for lora_name in all_loras:
                    importance = self._get_importance_score(
                        semantic_maps=semantic_maps,
                        source_lora=lora_name,
                        feature=feature,
                        layer_key=layer_key,
                        forward_map=forward_map,
                        inverse_map=inverse_map,
                    )

                    # Convert to float
                    if isinstance(importance, torch.Tensor):
                        importance = importance.mean().item()
                    else:
                        importance = float(importance)

                    feature_importances[lora_name] = importance

                # Apply preference boost to the preferred LoRA
                if preferred_lora in feature_importances:
                    feature_importances[preferred_lora] *= preference_strength

                # Normalize importances for this feature (so they sum to 1)
                total_importance = sum(feature_importances.values())
                if total_importance > 0:
                    normalized_importances = {
                        l: imp / total_importance
                        for l, imp in feature_importances.items()
                    }
                else:
                    # Fallback: equal distribution
                    normalized_importances = {l: 1.0 / len(all_loras) for l in all_loras}

                # Blend LoRAs for this feature
                for lora_name in all_loras:
                    lora_tensor = lora_weights[lora_name][sublayer_name]

                    # Skip if shape mismatch
                    if lora_tensor.shape != reference_shape:
                        continue

                    lora_importance = normalized_importances[lora_name]
                    contribution_weight = feature_weight * lora_importance

                    # Record statistics (only for "up" to avoid double counting)
                    if statistics is not None and sublayer_name == "up":
                        statistics.record_contribution(
                            layer_key=str(layer_key),
                            lora_name=lora_name,
                            feature=feature,
                            importance=feature_importances[lora_name],  # Raw (with boost)
                            feature_weight=feature_weight,
                            effective_weight=contribution_weight,
                            layer_type=layer_type,
                            relative_depth=relative_depth,
                        )

                    # Accumulate weighted tensor
                    weighted_contribution = lora_tensor.float() * contribution_weight

                    if merged_tensor is None:
                        merged_tensor = weighted_contribution
                    else:
                        merged_tensor = merged_tensor + weighted_contribution

                    total_weight += contribution_weight

            # Normalize and apply lambda
            if merged_tensor is not None and total_weight > 0:
                merged_tensor = (merged_tensor / total_weight) * lambda_value

                # Cast back to original dtype
                first_tensor = list(lora_weights.values())[0][sublayer_name]
                merged_tensor = merged_tensor.to(dtype=first_tensor.dtype)

            if sublayer_name == "up":
                merged_up = merged_tensor
            else:
                merged_down = merged_tensor

        # Merge alpha (simple average)
        alphas = [w["alpha"] for w in lora_weights.values() if w["alpha"] is not None]
        if alphas:
            if isinstance(alphas[0], torch.Tensor):
                merged_alpha = torch.stack([a.float() for a in alphas]).mean(dim=0)
            else:
                merged_alpha = sum(alphas) / len(alphas)
        else:
            merged_alpha = None

        # Create merged LoRAAdapter
        merged_weights = (merged_up, merged_down, merged_alpha, None, None, None)
        first_adapter = list(layer_tensors.values())[0]
        loaded_keys = first_adapter.loaded_keys if hasattr(first_adapter, 'loaded_keys') else set()

        return LoRAAdapter(weights=merged_weights, loaded_keys=loaded_keys)

    def _get_reference_shape(
            self,
            tensors: List[torch.Tensor],
            layer_key: str,
            sublayer_name: str,
    ) -> Optional[tuple]:
        """Get the most common shape among tensors (for compatibility check)."""
        if not tensors:
            return None

        shape_counts = {}
        for t in tensors:
            if t is not None:
                shape = tuple(t.shape)
                shape_counts[shape] = shape_counts.get(shape, 0) + 1

        if not shape_counts:
            return None

        return max(shape_counts.keys(), key=lambda s: shape_counts[s])

    def _get_importance_score(
            self,
            semantic_maps: Dict,
            source_lora: str,
            feature: str,
            layer_key: str,
            forward_map: Dict[str, str],
            inverse_map: Dict[str, Set[str]],
    ) -> float:
        """
        Get importance score for a feature at a specific layer.

        Tries multiple key formats to find a match in the semantic map.
        Falls back to 1.0 (uniform importance) if no match found.
        """
        # Check if LoRA has semantic map
        if source_lora not in semantic_maps:
            logger.debug(f"No semantic map for LoRA: {source_lora}")
            return 1.0

        lora_semantic_map = semantic_maps[source_lora]

        # Check if feature exists in this LoRA's map
        if feature not in lora_semantic_map:
            logger.debug(f"No feature '{feature}' in semantic map for {source_lora}")
            return 1.0

        feature_map = lora_semantic_map[feature]

        # Normalize layer_key to string
        layer_key_str = layer_key[0] if isinstance(layer_key, tuple) else str(layer_key)

        # Strategy 1: Direct lookup with original layer_key
        if layer_key in feature_map:
            return self._extract_importance(feature_map[layer_key])

        # Strategy 2: Direct lookup with string version
        if layer_key_str in feature_map:
            return self._extract_importance(feature_map[layer_key_str])

        # Strategy 3: Convert LoRA key → UNet key, then try matching
        unet_key = forward_map.get(layer_key_str)
        if unet_key and unet_key in feature_map:
            return self._extract_importance(feature_map[unet_key])

        # Strategy 4: Semantic component matching (ComfyUI ↔ Diffusers)
        matched_importance = self._match_by_semantic_components(
            layer_key_str, feature_map
        )
        if matched_importance is not None:
            return matched_importance

        # Strategy 5: Fallback heuristic matching
        for map_key in feature_map.keys():
            if self._keys_likely_match(layer_key_str, map_key):
                logger.debug(f"Fuzzy match: {layer_key_str} → {map_key}")
                return self._extract_importance(feature_map[map_key])

        # Log failed lookup for debugging
        if logger.isEnabledFor(logging.DEBUG):
            sample_keys = list(feature_map.keys())[:3]
            logger.debug(
                f"No importance match | layer={layer_key_str} | "
                f"feature={feature} | lora={source_lora} | "
                f"sample_map_keys={sample_keys}"
            )

        return 1.0

    def _extract_importance(self, importance: Any) -> float:
        """Extract scalar importance value."""
        if isinstance(importance, torch.Tensor):
            return importance.mean().item()
        return float(importance)

    def _match_by_semantic_components(
            self,
            layer_key: str,
            feature_map: Dict[str, Any],
    ) -> Optional[float]:
        """
        Match ComfyUI layer key to Diffusers keys by extracting semantic components.

        Handles the translation between naming conventions:
        - ComfyUI: diffusion_model.input_blocks.1.1.transformer_blocks.0.attn1.to_q
        - Diffusers: down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.processor...

        """
        # Parse the ComfyUI key
        comfy_components = self._parse_comfy_key(layer_key)
        if not comfy_components:
            return None

        # Try to match against each key in feature_map
        for diffusers_key, importance in feature_map.items():
            diffusers_components = self._parse_diffusers_key(diffusers_key)
            if not diffusers_components:
                continue

            # Match on semantic components
            if self._components_match(comfy_components, diffusers_components):
                logger.debug(
                    f"Semantic match: {layer_key} → {diffusers_key} | "
                    f"components: {comfy_components}"
                )
                return self._extract_importance(importance)

        return None

    def _parse_comfy_key(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Parse a ComfyUI/LoRA layer key into semantic components.

        Example: diffusion_model.input_blocks.1.1.transformer_blocks.0.attn1.to_q

        For SD1.5 input_blocks:
        - 0: stem (conv_in)
        - 1, 2: down_block_0 attention layers → attentions.0, attentions.1
        - 3: downsample
        - 4, 5: down_block_1 attention layers
        - 6: downsample
        - 7, 8: down_block_2 attention layers
        - ...

        """
        import re

        components = {}
        key_lower = key.lower()

        # === Block type and indices ===
        if "input_blocks" in key_lower:
            components["block_type"] = "down"
            # Parse input_blocks.X.Y where Y=1 indicates attention
            match = re.search(r"input_blocks\.(\d+)\.(\d+)", key_lower)
            if match:
                input_block_num = int(match.group(1))
                sub_idx = int(match.group(2))

                # Only attention layers (sub_idx = 1) have transformer blocks
                if sub_idx == 1 and input_block_num >= 1:
                    # Convert to diffusers indexing
                    # input_blocks 1,2 → down_blocks.0.attentions.0,1
                    # input_blocks 4,5 → down_blocks.1.attentions.0,1
                    # (skipping indices 0, 3, 6, 9... which are stem/downsamples)
                    components["block_idx"] = (input_block_num - 1) // 3
                    components["attention_idx"] = (input_block_num - 1) % 3
                else:
                    return None  # Not an attention layer

        elif "output_blocks" in key_lower:
            components["block_type"] = "up"
            match = re.search(r"output_blocks\.(\d+)\.(\d+)", key_lower)
            if match:
                output_block_num = int(match.group(1))
                sub_idx = int(match.group(2))

                # For output_blocks: structure is [resnet, resnet, attention, upsample]
                # This mapping is more complex, approximate:
                components["block_idx"] = output_block_num // 3
                components["attention_idx"] = output_block_num % 3

        elif "middle_block" in key_lower or "mid_block" in key_lower:
            components["block_type"] = "mid"
            components["block_idx"] = 0
            components["attention_idx"] = 0
        else:
            return None

        # === Transformer block index ===
        transformer_match = re.search(r"transformer_blocks[._](\d+)", key_lower)
        if transformer_match:
            components["transformer_idx"] = int(transformer_match.group(1))
        else:
            components["transformer_idx"] = 0

        # === Attention type ===
        if "attn1" in key_lower:
            components["attn_type"] = "self"
        elif "attn2" in key_lower:
            components["attn_type"] = "cross"
        else:
            components["attn_type"] = None

        # === Sublayer ===
        for sublayer in ["to_q", "to_k", "to_v", "to_out", "proj_in", "proj_out", "ff"]:
            if sublayer in key_lower:
                components["sublayer"] = sublayer
                break

        return components if "sublayer" in components else None


    def _parse_diffusers_key(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Parse a Diffusers key into semantic components.

        Example: down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.lora_A.LoraName.weight
        """
        import re

        components = {}
        key_lower = key.lower()

        # === Block type and indices ===
        if "down_blocks" in key_lower:
            components["block_type"] = "down"
            match = re.search(r"down_blocks\.(\d+)\.attentions\.(\d+)", key_lower)
            if match:
                components["block_idx"] = int(match.group(1))
                components["attention_idx"] = int(match.group(2))
            else:
                return None

        elif "up_blocks" in key_lower:
            components["block_type"] = "up"
            match = re.search(r"up_blocks\.(\d+)\.attentions\.(\d+)", key_lower)
            if match:
                components["block_idx"] = int(match.group(1))
                components["attention_idx"] = int(match.group(2))
            else:
                return None

        elif "mid_block" in key_lower:
            components["block_type"] = "mid"
            components["block_idx"] = 0
            match = re.search(r"mid_block\.attentions\.(\d+)", key_lower)
            components["attention_idx"] = int(match.group(1)) if match else 0
        else:
            return None

        # === Transformer block index ===
        transformer_match = re.search(r"transformer_blocks\.(\d+)", key_lower)
        if transformer_match:
            components["transformer_idx"] = int(transformer_match.group(1))
        else:
            components["transformer_idx"] = 0

        # === Attention type ===
        if "attn1" in key_lower:
            components["attn_type"] = "self"
        elif "attn2" in key_lower:
            components["attn_type"] = "cross"
        else:
            components["attn_type"] = None

        # === Sublayer ===
        for sublayer in ["to_q", "to_k", "to_v", "to_out", "proj_in", "proj_out", "ff"]:
            if sublayer in key_lower:
                components["sublayer"] = sublayer
                break

        return components if "sublayer" in components else None

    def _components_match(
            self,
            comfy: Dict[str, Any],
            diffusers: Dict[str, Any],
    ) -> bool:
        """
        Check if two component dictionaries represent the same layer.

        All specified components must match exactly.
        """
        # Must match block type
        if comfy.get("block_type") != diffusers.get("block_type"):
            return False

        # Must match block index
        if comfy.get("block_idx") != diffusers.get("block_idx"):
            return False

        # Must match attention index within the block
        if comfy.get("attention_idx") != diffusers.get("attention_idx"):
            return False

        # Must match transformer block index
        if comfy.get("transformer_idx") != diffusers.get("transformer_idx"):
            return False

        # Must match attention type if both have it
        comfy_attn = comfy.get("attn_type")
        diff_attn = diffusers.get("attn_type")
        if comfy_attn and diff_attn and comfy_attn != diff_attn:
            return False

        # Must match sublayer
        if comfy.get("sublayer") != diffusers.get("sublayer"):
            return False

        return True

    def _keys_likely_match(self, key1: str, key2: str) -> bool:
        """
        Fallback heuristic: check if two keys likely refer to the same layer.
        """
        import re

        k1 = key1.lower()
        k2 = key2.lower()

        # Must have same attention type
        k1_attn = "attn1" in k1 or "attn2" in k1
        k2_attn = "attn1" in k2 or "attn2" in k2
        if k1_attn and k2_attn:
            if ("attn1" in k1) != ("attn1" in k2):
                return False

        # Must have same sublayer type
        sublayers = ["to_q", "to_k", "to_v", "to_out", "proj_in", "proj_out", "ff"]
        k1_sublayer = next((s for s in sublayers if s in k1), None)
        k2_sublayer = next((s for s in sublayers if s in k2), None)
        if k1_sublayer and k2_sublayer and k1_sublayer != k2_sublayer:
            return False

        # Must have same general block type
        k1_down = "input" in k1 or "down" in k1
        k1_up = "output" in k1 or "up" in k1
        k1_mid = "middle" in k1 or "mid" in k1

        k2_down = "down" in k2
        k2_up = "up" in k2
        k2_mid = "mid" in k2

        if k1_down and not k2_down:
            return False
        if k1_up and not k2_up:
            return False
        if k1_mid and not k2_mid:
            return False

        # Check for shared numbers (block indices)
        nums1 = set(re.findall(r'\d+', k1))
        nums2 = set(re.findall(r'\d+', k2))

        # At least one shared number
        if nums1 and nums2 and not (nums1 & nums2):
            return False

        return True

    def _diagnose_key_matching(
            self,
            semantic_maps: Dict[str, Dict[str, Dict[str, Any]]],
            layer_keys: List[str],
            forward_map: Dict[str, str],
    ) -> str:
        """Diagnose key matching with detailed component analysis."""
        lines = ["=" * 60, "KEY MATCHING DIAGNOSTIC", "=" * 60, ""]

        # Sample layer keys with parsed components
        lines.append(f"Sample layer keys from patches ({len(layer_keys)} provided):")
        for k in layer_keys[:5]:
            comfy_components = self._parse_comfy_key(k)
            lines.append(f"  {k[:70]}...")
            lines.append(f"    → {comfy_components}")

        lines.append("")

        # Sample semantic map keys with parsed components
        for lora_name, lora_map in list(semantic_maps.items())[:1]:
            lines.append(f"Sample semantic map keys for '{lora_name}':")
            for feature, feature_map in list(lora_map.items())[:1]:
                for map_key in list(feature_map.keys())[:3]:
                    diff_components = self._parse_diffusers_key(map_key)
                    lines.append(f"  {map_key[:70]}...")
                    lines.append(f"    → {diff_components}")

        lines.append("")

        # Test matching
        lines.append("Matching tests:")
        for layer_key in layer_keys[:5]:
            layer_key_str = layer_key[0] if isinstance(layer_key, tuple) else str(layer_key)
            comfy_components = self._parse_comfy_key(layer_key_str)

            if not comfy_components:
                lines.append(f"  ✗ Could not parse: {layer_key_str[:50]}...")
                continue

            lines.append(f"  Testing: {layer_key_str[:50]}...")
            lines.append(f"    Parsed: {comfy_components}")

            # Try each LoRA's semantic map
            for lora_name, lora_map in semantic_maps.items():
                first_feature = list(lora_map.keys())[0]
                feature_map = lora_map[first_feature]

                matches = []
                for map_key in feature_map.keys():
                    diff_components = self._parse_diffusers_key(map_key)
                    if diff_components and self._components_match(comfy_components, diff_components):
                        matches.append((map_key[:60], diff_components))

                if matches:
                    lines.append(f"    ✓ {lora_name}: {len(matches)} match(es)")
                    for match_key, match_comp in matches[:2]:
                        lines.append(f"      → {match_key}...")
                        lines.append(f"        {match_comp}")
                else:
                    lines.append(f"    ✗ {lora_name}: No matches")

                    # Show why - compare with first few semantic map keys
                    lines.append(f"      Closest candidates:")
                    for map_key in list(feature_map.keys())[:2]:
                        diff_components = self._parse_diffusers_key(map_key)
                        if diff_components:
                            lines.append(f"        {map_key[:50]}...")
                            lines.append(f"        → {diff_components}")

        return "\n".join(lines)

    def _extract_importance(self, importance: Any) -> float:
        """Extract scalar importance value."""
        if isinstance(importance, torch.Tensor):
            return importance.mean().item()
        return float(importance)

    def _filter_compatible_tensors(
        self,
        tensors: List[torch.Tensor],
        layer_key: str,
        sublayer_name: str,
    ) -> List[torch.Tensor]:
        """Filter tensors to only include those with compatible shapes."""
        if not tensors:
            return []

        shape_counts = {}
        for tensor in tensors:
            shape = tuple(tensor.shape)
            shape_counts[shape] = shape_counts.get(shape, 0) + 1

        most_common_shape = max(shape_counts.keys(), key=lambda s: shape_counts[s])
        return [t for t in tensors if tuple(t.shape) == most_common_shape]


class PMSemanticMergeStatsViewer:
    """
    View and export merge statistics.

    Allows querying specific features or layers from merge statistics.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "statistics": ("MergeStats",),
                "view_type": (["summary", "by_feature", "by_layer", "export_json"], {"default": "summary"}),
            },
            "optional": {
                "query": ("STRING", {"default": "", "tooltip": "Feature name or layer key to query"}),
                "verbose": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output",)
    FUNCTION = "view_stats"
    CATEGORY = "LoRA PowerMerge/Semantic"
    DESCRIPTION = "View and query merge statistics"

    def view_stats(
        self,
        statistics: MergeStatistics,
        view_type: str,
        query: str = "",
        verbose: bool = False,
    ) -> Tuple[str]:
        """View statistics in different formats."""
        if statistics is None:
            return ("No statistics available. Enable 'collect_statistics' in merger.",)

        if view_type == "summary":
            return (statistics.format_report(verbose=verbose),)

        elif view_type == "by_feature":
            if not query:
                return (f"Specify a feature name. Available: {', '.join(statistics.features)}",)
            breakdown = statistics.get_feature_breakdown(query)
            if "error" in breakdown:
                return (breakdown["error"],)
            return (self._format_feature_breakdown(breakdown),)

        elif view_type == "by_layer":
            if not query:
                sample_keys = list(statistics.layer_contributions.keys())[:5]
                return (f"Specify a layer key. Sample keys: {sample_keys}",)
            breakdown = statistics.get_layer_breakdown(query)
            if "error" in breakdown:
                return (breakdown["error"],)
            return (self._format_layer_breakdown(breakdown),)

        elif view_type == "export_json":
            return (statistics.to_json(indent=2),)

        return ("Unknown view_type",)

    def _format_feature_breakdown(self, breakdown: Dict[str, Any]) -> str:
        """Format feature breakdown as readable string."""
        lines = [
            f"Feature: {breakdown['feature']}",
            "",
            "Overall LoRA contributions:",
        ]
        for lora, pct in sorted(breakdown['overall'].items(), key=lambda x: -x[1]):
            lines.append(f"  {lora}: {pct*100:.1f}%")

        lines.append("")
        lines.append("By depth:")
        for depth, contribs in breakdown.get('by_depth', {}).items():
            contrib_str = ", ".join(f"{l}: {p*100:.0f}%" for l, p in sorted(contribs.items(), key=lambda x: -x[1]))
            lines.append(f"  {depth}: {contrib_str}")

        lines.append("")
        lines.append("By layer type:")
        for lt, contribs in breakdown.get('by_layer_type', {}).items():
            contrib_str = ", ".join(f"{l}: {p*100:.0f}%" for l, p in sorted(contribs.items(), key=lambda x: -x[1]))
            lines.append(f"  {lt}: {contrib_str}")

        lines.append("")
        lines.append("Coverage (layers contributed):")
        for lora, count in breakdown.get('coverage', {}).items():
            lines.append(f"  {lora}: {count} layers")

        return "\n".join(lines)

    def _format_layer_breakdown(self, breakdown: Dict[str, Any]) -> str:
        """Format layer breakdown as readable string."""
        lines = [
            f"Layer: {breakdown['layer']}",
            f"Type: {breakdown['layer_type']}",
            f"Depth: {breakdown['depth']:.2f}",
            f"Dominant LoRA: {breakdown.get('dominant', 'none (contested)')}",
            "",
            "LoRA shares:",
        ]
        for lora, share in sorted(breakdown['lora_shares'].items(), key=lambda x: -x[1]):
            lines.append(f"  {lora}: {share*100:.1f}%")

        lines.append("")
        lines.append("By feature:")
        for feature, contribs in breakdown.get('by_feature', {}).items():
            lines.append(f"  {feature}:")
            for c in contribs:
                lines.append(f"    {c['lora']}: importance={c['importance']:.3f}, "
                           f"weight={c['feature_weight']:.2f}, share={c['share']*100:.1f}%")

        return "\n".join(lines)


# Node registration

NODE_CLASS_MAPPINGS = {
    "PM Semantic Merge Spec": PMSemanticMergeSpec,
    "PM Semantic Merger": PMSemanticMerger,
    "PM Semantic Merge Stats Viewer": PMSemanticMergeStatsViewer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PM Semantic Merge Spec": "PM Semantic Merge Spec",
    "PM Semantic Merger": "PM Semantic Merger",
    "PM Semantic Merge Stats Viewer": "PM Merge Stats Viewer",
}