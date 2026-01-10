"""
Semantic LoRA merger - Pure diffusers implementation.

No ComfyUI dependencies. Works directly with safetensors files.
"""

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

import torch

from .lora_loader import load_lora_safetensors, save_lora_safetensors
from .key_utils import NormalizedKey, build_normalized_index, normalize_diffusers_key, _normalize_sublayer
from .merge_statistics import MergeStatistics, classify_layer_type, compute_relative_depth

logger = logging.getLogger(__name__)


@dataclass
class MergeSpec:
    """
    Specification for semantic merge.

    Each feature assignment has:
    - source: Which LoRA to prefer for this feature
    - weight: How strongly this feature influences the merge (default 1.0)
    - exclusive: If True, ONLY use the source LoRA for this feature (default False)

    """
    feature_assignments: Dict[str, Dict[str, Any]]

    @classmethod
    def from_text(cls, specification: str) -> "MergeSpec":
        """
        Parse merge specification from text.

        Format: "feature from lora_name [weight] [exclusive], ..."

        Examples:
            Basic:
                "hair from lora1, face from lora2"

            With weights (higher = more influence):
                "hair from lora1 1.5, face from lora2 0.8"

            Exclusive mode (only use specified LoRA):
                "hair from lora1 exclusive, face from lora2"

            Combined:
                "hair from lora1 2.0 exclusive, face from lora2 1.0, style from lora1 0.5"

        Weight interpretation:
            - 1.0 = normal influence
            - 2.0 = double influence (this feature matters twice as much)
            - 0.5 = half influence (this feature matters less)

        Exclusive mode:
            - When set, only the specified LoRA contributes to this feature
            - Other LoRAs are completely excluded for this feature
            - Useful for features that should come entirely from one source

        """
        assignments = {}

        for part in specification.split(","):
            part = part.strip()
            if not part:
                continue

            tokens = part.split()

            if len(tokens) < 3 or tokens[1].lower() != "from":
                logger.warning(f"Skipping invalid assignment: '{part}'")
                continue

            feature = tokens[0].lower()
            source = tokens[2]
            weight = 1.0
            exclusive = False

            # Parse remaining tokens for weight and/or exclusive
            for token in tokens[3:]:
                if token.lower() == "exclusive":
                    exclusive = True
                else:
                    try:
                        weight = float(token)
                    except ValueError:
                        logger.warning(f"Unknown token '{token}' in '{part}'")

            assignments[feature] = {
                "source": source,
                "weight": weight,
                "exclusive": exclusive,
            }

            logger.debug(
                f"Parsed feature: {feature} -> source={source}, "
                f"weight={weight}, exclusive={exclusive}"
            )

        if not assignments:
            raise ValueError("No valid feature assignments found in specification")

        return cls(feature_assignments=assignments)

    def format_description(self) -> str:
        """Format human-readable description of the merge spec."""
        lines = ["Semantic merge specification:"]
        for feature, spec in self.feature_assignments.items():
            excl_str = " [EXCLUSIVE]" if spec.get("exclusive") else ""
            lines.append(
                f"  {feature}: {spec['source']} "
                f"(weight={spec['weight']:.2f}){excl_str}"
            )
        return "\n".join(lines)


class SemanticMerger:
    """
    Merge LoRAs using semantic importance maps.

    Pure diffusers implementation - no ComfyUI dependencies.
    """

    def __init__(
        self,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype or torch.float32

    def merge(
        self,
        lora_paths: Dict[str, str],
        semantic_maps: Dict[str, Dict[str, Dict[str, Any]]],
        merge_spec: MergeSpec,
        output_path: str,
        lambda_value: float = 1.0,
        preference_strength: float = 1.5,
        collect_statistics: bool = True,
    ) -> Tuple[str, Optional[MergeStatistics], str]:
        """
        Merge LoRAs with semantic routing.

        Args:
            lora_paths: {lora_name: path_to_safetensors}
            semantic_maps: {lora_name: {feature: {diffusers_key: importance}}}
            merge_spec: Feature assignment specification
            output_path: Where to save merged LoRA
            lambda_value: Global scaling factor
            preference_strength: Boost for preferred LoRA (1.0 = pure semantic)
            collect_statistics: Whether to collect detailed stats

        Returns:
            (output_path, statistics, report_string)
        """
        logger.info(f"Starting semantic merge of {len(lora_paths)} LoRAs")
        logger.info(f"Preference strength: {preference_strength}")

        """Merge LoRAs with semantic routing."""

        logger.info(f"Starting semantic merge of {len(lora_paths)} LoRAs")

        # Load all LoRAs
        loras = {}
        for name, path in lora_paths.items():
            loras[name] = load_lora_safetensors(path, device=self.device, dtype=self.dtype)

        # Use first LoRA's original keys as template
        first_lora = list(loras.values())[0]
        original_keys_template = first_lora["original_keys"]

        # Normalize semantic maps
        normalized_semantic_maps = self._normalize_semantic_maps(semantic_maps)

        # Diagnostic: Show sample keys from both sides
        logger.info("=" * 60)
        logger.info("KEY FORMAT DIAGNOSTIC")
        logger.info("=" * 60)

        # Sample LoRA normalized keys
        first_lora = list(loras.values())[0]
        sample_lora_keys = list(first_lora["down_tensors"].keys())[:5]
        logger.info("Sample NormalizedKeys from LoRA:")
        for nk in sample_lora_keys:
            logger.info(f"  {nk}")

        # Sample semantic map normalized keys
        first_lora_name = list(normalized_semantic_maps.keys())[0]
        first_feature = list(normalized_semantic_maps[first_lora_name].keys())[0]
        sample_semantic_keys = list(normalized_semantic_maps[first_lora_name][first_feature].keys())[:5]
        logger.info(f"\nSample NormalizedKeys from semantic map ({first_lora_name}/{first_feature}):")
        for nk in sample_semantic_keys:
            logger.info(f"  {nk}")

        # Check for exact matches
        lora_key_set = set(first_lora["down_tensors"].keys())
        semantic_key_set = set(normalized_semantic_maps[first_lora_name][first_feature].keys())

        matches = lora_key_set & semantic_key_set
        logger.info(f"\nDirect matches: {len(matches)} / {len(lora_key_set)} LoRA keys")

        if len(matches) < len(lora_key_set):
            # Show examples of non-matching keys
            only_in_lora = list(lora_key_set - semantic_key_set)[:3]
            only_in_semantic = list(semantic_key_set - lora_key_set)[:3]

            logger.info("\nSample keys only in LoRA (no semantic match):")
            for nk in only_in_lora:
                logger.info(f"  {nk}")

            logger.info("\nSample keys only in semantic map (no LoRA match):")
            for nk in only_in_semantic:
                logger.info(f"  {nk}")

        logger.info("=" * 60)

        # Collect all normalized keys
        all_normalized_keys = set()
        for lora_data in loras.values():
            all_normalized_keys.update(lora_data["down_tensors"].keys())
            all_normalized_keys.update(lora_data["up_tensors"].keys())

        sorted_keys = sorted(all_normalized_keys, key=str)
        total_layers = len(sorted_keys)

        logger.info(f"Total normalized layers: {total_layers}")

        # Initialize statistics
        stats = MergeStatistics(lambda_value=lambda_value) if collect_statistics else None

        # Merge each normalized key
        merged_up: Dict[NormalizedKey, torch.Tensor] = {}
        merged_down: Dict[NormalizedKey, torch.Tensor] = {}
        merged_alpha: Dict[NormalizedKey, torch.Tensor] = {}

        # Merge original_keys from all LoRAs (in case some keys exist in one but not another)
        combined_original_keys: Dict[NormalizedKey, Dict[str, str]] = {}
        for lora_data in loras.values():
            for norm_key, key_dict in lora_data["original_keys"].items():
                if norm_key not in combined_original_keys:
                    combined_original_keys[norm_key] = {}
                combined_original_keys[norm_key].update(key_dict)

        for layer_idx, norm_key in enumerate(sorted_keys):
            relative_depth = compute_relative_depth(layer_idx, total_layers)
            layer_type = classify_layer_type(str(norm_key))

            up, down, alpha = self._merge_layer(
                norm_key=norm_key,
                loras=loras,
                semantic_maps=normalized_semantic_maps,
                merge_spec=merge_spec,
                lambda_value=lambda_value,
                preference_strength=preference_strength,
                statistics=stats,
                relative_depth=relative_depth,
                layer_type=layer_type,
            )

            if up is not None:
                merged_up[norm_key] = up
            if down is not None:
                merged_down[norm_key] = down
            if alpha is not None:
                merged_alpha[norm_key] = alpha

        # Finalize statistics
        if stats:
            stats.finalize()
            report = stats.format_report()
            logger.info(f"Merge statistics collected for {stats.total_layers} layers")
        else:
            report = "Statistics collection disabled"

        # Save merged LoRA using original key templates
        output_path = save_lora_safetensors(
            up_tensors=merged_up,
            down_tensors=merged_down,
            alpha_tensors=merged_alpha,
            original_keys=combined_original_keys,  # Pass original keys
            output_path=output_path,
            metadata={
                "merge_type": "semantic",
                "source_loras": ",".join(lora_paths.keys()),
                "features": ",".join(merge_spec.feature_assignments.keys()),
            },
        )

        logger.info(f"Semantic merge complete: {output_path}")

        return output_path, stats, report

    def _normalize_semantic_maps(
        self,
        semantic_maps: Dict[str, Dict[str, Dict[str, Any]]],
    ) -> Dict[str, Dict[str, Dict[NormalizedKey, float]]]:
        """Convert semantic map keys from diffusers format to NormalizedKey."""
        normalized = {}

        for lora_name, lora_map in semantic_maps.items():
            normalized[lora_name] = {}

            for feature, feature_map in lora_map.items():
                normalized[lora_name][feature] = {}

                for diffusers_key, importance in feature_map.items():
                    norm_key = normalize_diffusers_key(diffusers_key)
                    if norm_key:
                        # Convert importance to float
                        if isinstance(importance, torch.Tensor):
                            importance = importance.mean().item()
                        normalized[lora_name][feature][norm_key] = float(importance)

        return normalized

    def _merge_layer(
        self,
        norm_key: NormalizedKey,
        loras: Dict[str, Dict[str, Any]],
        semantic_maps: Dict[str, Dict[str, Dict[NormalizedKey, float]]],
        merge_spec: MergeSpec,
        lambda_value: float,
        preference_strength: float,
        statistics: Optional[MergeStatistics],
        relative_depth: float,
        layer_type: Any,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Merge a single layer across all LoRAs."""

        all_loras = list(loras.keys())

        # Collect tensors for this layer
        up_by_lora = {}
        down_by_lora = {}
        alpha_by_lora = {}

        for lora_name, lora_data in loras.items():
            if norm_key in lora_data["up_tensors"]:
                up_by_lora[lora_name] = lora_data["up_tensors"][norm_key]
            if norm_key in lora_data["down_tensors"]:
                down_by_lora[lora_name] = lora_data["down_tensors"][norm_key]
            if norm_key in lora_data["alpha_tensors"]:
                alpha_by_lora[lora_name] = lora_data["alpha_tensors"][norm_key]

        if not up_by_lora and not down_by_lora:
            return None, None, None

        # Merge up and down weights
        merged_up = self._merge_weight_type(
            weight_type="up",
            tensors_by_lora=up_by_lora,
            norm_key=norm_key,
            semantic_maps=semantic_maps,
            merge_spec=merge_spec,
            lambda_value=lambda_value,
            preference_strength=preference_strength,
            statistics=statistics,
            relative_depth=relative_depth,
            layer_type=layer_type,
            all_loras=all_loras,
        )

        merged_down = self._merge_weight_type(
            weight_type="down",
            tensors_by_lora=down_by_lora,
            norm_key=norm_key,
            semantic_maps=semantic_maps,
            merge_spec=merge_spec,
            lambda_value=lambda_value,
            preference_strength=preference_strength,
            statistics=None,  # Only record stats once (for "up")
            relative_depth=relative_depth,
            layer_type=layer_type,
            all_loras=all_loras,
        )

        # Merge alpha (simple average)
        merged_alpha = None
        if alpha_by_lora:
            alphas = list(alpha_by_lora.values())
            if isinstance(alphas[0], torch.Tensor):
                merged_alpha = torch.stack([a.float() for a in alphas]).mean(dim=0)
            else:
                merged_alpha = torch.tensor(sum(alphas) / len(alphas))

        return merged_up, merged_down, merged_alpha

    def _merge_weight_type(
            self,
            weight_type: str,
            tensors_by_lora: Dict[str, torch.Tensor],
            norm_key: NormalizedKey,
            semantic_maps: Dict[str, Dict[str, Dict[NormalizedKey, float]]],
            merge_spec: MergeSpec,
            lambda_value: float,
            preference_strength: float,
            statistics: Optional[MergeStatistics],
            relative_depth: float,
            layer_type: Any,
            all_loras: List[str],
    ) -> Optional[torch.Tensor]:
        """
        Merge up or down weights with semantic routing.

        For each feature:
        1. Get semantic importance from each LoRA
        2. Apply preference boost to preferred LoRA
        3. If exclusive mode, zero out other LoRAs
        4. Normalize and apply feature weight
        5. Blend tensors

        """
        if not tensors_by_lora:
            return None

        # Check shape compatibility
        shapes = {name: t.shape for name, t in tensors_by_lora.items()}
        if len(set(shapes.values())) > 1:
            logger.warning(f"Shape mismatch at {norm_key}: {shapes}")
            shape_counts = {}
            for s in shapes.values():
                shape_counts[s] = shape_counts.get(s, 0) + 1
            target_shape = max(shape_counts.keys(), key=lambda s: shape_counts[s])
            tensors_by_lora = {n: t for n, t in tensors_by_lora.items() if t.shape == target_shape}

        merged_tensor = None
        total_weight = 0.0

        # Process each feature
        for feature, spec in merge_spec.feature_assignments.items():
            preferred_lora = spec["source"]
            feature_weight = spec["weight"]
            is_exclusive = spec.get("exclusive", False)

            # Get importance for each LoRA at this layer for this feature
            feature_importances = {}
            match_status = {}

            for lora_name in all_loras:
                # In exclusive mode, skip non-preferred LoRAs entirely
                if is_exclusive and lora_name != preferred_lora:
                    feature_importances[lora_name] = 0.0
                    match_status[lora_name] = True  # Mark as "handled"
                    continue

                importance, was_matched = self._get_importance(
                    semantic_maps=semantic_maps,
                    lora_name=lora_name,
                    feature=feature,
                    norm_key=norm_key,
                    statistics=statistics,
                )
                feature_importances[lora_name] = importance
                match_status[lora_name] = was_matched

            # Apply preference boost (only in non-exclusive mode)
            if not is_exclusive and preferred_lora in feature_importances:
                feature_importances[preferred_lora] *= preference_strength

            # Normalize importances for this feature
            total_importance = sum(feature_importances.values())
            if total_importance > 0:
                normalized_importances = {
                    l: imp / total_importance
                    for l, imp in feature_importances.items()
                }
            else:
                # Fallback: if all zeros (shouldn't happen), give all to preferred
                normalized_importances = {l: 0.0 for l in all_loras}
                if preferred_lora in normalized_importances:
                    normalized_importances[preferred_lora] = 1.0

            # Blend tensors for this feature
            for lora_name, tensor in tensors_by_lora.items():
                lora_importance = normalized_importances.get(lora_name, 0.0)

                # Skip if this LoRA doesn't contribute
                if lora_importance == 0.0:
                    continue

                contribution_weight = feature_weight * lora_importance

                # Record statistics
                if statistics is not None:
                    statistics.record_contribution(
                        layer_key=str(norm_key),
                        lora_name=lora_name,
                        feature=feature,
                        importance=feature_importances.get(lora_name, 0.0),
                        feature_weight=feature_weight,
                        effective_weight=contribution_weight,
                        layer_type=layer_type,
                        relative_depth=relative_depth,
                        was_matched=match_status.get(lora_name, False),
                        is_exclusive=is_exclusive,
                    )

                # Accumulate weighted tensor
                weighted = tensor.float() * contribution_weight

                if merged_tensor is None:
                    merged_tensor = weighted
                else:
                    merged_tensor = merged_tensor + weighted

                total_weight += contribution_weight

        # Normalize and apply lambda
        if merged_tensor is not None and total_weight > 0:
            merged_tensor = (merged_tensor / total_weight) * lambda_value

            # Cast back to original dtype
            first_tensor = list(tensors_by_lora.values())[0]
            merged_tensor = merged_tensor.to(dtype=first_tensor.dtype)

        return merged_tensor

    def _get_importance(
            self,
            semantic_maps: Dict[str, Dict[str, Dict[NormalizedKey, float]]],
            lora_name: str,
            feature: str,
            norm_key: NormalizedKey,
            statistics: Optional[MergeStatistics] = None,  # Add this parameter
    ) -> Tuple[float, bool]:
        """Get importance score using NormalizedKey lookup."""

        if lora_name not in semantic_maps:
            if statistics:
                statistics.record_unmatched(norm_key, lora_name, feature)
            return 1.0, False

        lora_map = semantic_maps[lora_name]
        if feature not in lora_map:
            if statistics:
                statistics.record_unmatched(norm_key, lora_name, feature)
            return 1.0, False

        feature_map = lora_map[feature]

        # Direct lookup
        if norm_key in feature_map:
            return feature_map[norm_key], True

        # Fallback matching strategies...
        # ... existing fallback code ...

        # If still no match, record for debugging
        if statistics:
            statistics.record_unmatched(norm_key, lora_name, feature)

        return 1.0, False

    @staticmethod
    def _sublayers_equivalent(sl1: str, sl2: str) -> bool:
        """Check if two sublayer names are equivalent."""
        # Normalize both
        n1 = _normalize_sublayer(sl1)
        n2 = _normalize_sublayer(sl2)
        return n1 == n2