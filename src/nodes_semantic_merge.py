"""
Semantic LoRA merging nodes with statistics collection.
"""

import logging
import os
from typing import Dict, Tuple, Any, Optional

import folder_paths
import torch
from comfy.weight_adapter import LoRAAdapter

from .analysis.merge_statistics import MergeStatistics
from .analysis.semantic_merger import MergeSpec, SemanticMerger

logger = logging.getLogger(__name__)


class PMSemanticMergeSpec:
    """Create a semantic merge specification from text description."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "specification": (
                    "STRING",
                    {
                        "default": "hair from lora1 1.0, face from lora1 1.0, clothing from lora2 1.5 exclusive",
                        "multiline": True,
                        "tooltip": (
                            "Format: 'feature from lora_name [weight] [exclusive], ...'\n"
                            "Examples:\n"
                            "  hair from lora1\n"
                            "  hair from lora1 2.0 (double influence)\n"
                            "  hair from lora1 exclusive (only use lora1)\n"
                            "  hair from lora1 2.0 exclusive"
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("MergeSpec", "STRING")
    RETURN_NAMES = ("merge_spec", "parsed_spec")
    FUNCTION = "create_spec"
    CATEGORY = "LoRA PowerMerge/Semantic"
    DESCRIPTION = """
Parse text specification into semantic merge configuration.

Format: feature from lora_name [weight] [exclusive], ...

Weight (optional):
  - 1.0 = normal (default)
  - 2.0 = double influence
  - 0.5 = half influence

Exclusive (optional):
  - Adds 'exclusive' to only use that LoRA for the feature
  - Other LoRAs contribute 0% to this feature

Examples:
  hair from charLoRA, clothing from styleLoRA
  hair from charLoRA 2.0, face from charLoRA 1.0
  hair from charLoRA exclusive, background from styleLoRA
"""

    def create_spec(self, specification: str) -> Tuple[MergeSpec, str]:
        """Parse merge specification text."""
        try:
            merge_spec = MergeSpec.from_text(specification)
        except ValueError as e:
            raise ValueError(f"Invalid specification: {e}")

        description = merge_spec.format_description()
        logger.info(f"Parsed merge spec:\n{description}")

        return (merge_spec, description)


class PMSemanticMerger:
    """ComfyUI node for semantic LoRA merging."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lora_paths": ("LoRAPathDict",),  # {name: path}
                "semantic_maps": ("SemanticMaps",),
                "specification": ("STRING", {"default": "hair from lora1, face from lora2"}),
                "output_name": ("STRING", {"default": "semantic_merged"}),
            },
            "optional": {
                "lambda_value": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0}),
                "preference_strength": ("FLOAT", {"default": 1.5, "min": 1.0, "max": 5.0}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "MergeStats")
    RETURN_NAMES = ("output_path", "report", "statistics")
    FUNCTION = "merge"
    CATEGORY = "LoRA PowerMerge/Semantic"

    def merge(
            self,
            lora_paths: Dict[str, str],
            semantic_maps: Dict[str, Dict[str, Dict[str, Any]]],
            specification: str,
            output_name: str,
            lambda_value: float = 1.0,
            preference_strength: float = 1.5,
    ) -> Tuple[str, str, Optional[MergeStatistics]]:
        merge_spec = MergeSpec.from_text(specification)

        output_dir = folder_paths.get_output_directory()
        output_path = os.path.join(output_dir, f"{output_name}.safetensors")

        merger = SemanticMerger(
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            dtype=torch.float32,
        )

        result_path, stats, report = merger.merge(
            lora_paths=lora_paths,
            semantic_maps=semantic_maps,
            merge_spec=merge_spec,
            output_path=output_path,
            lambda_value=lambda_value,
            preference_strength=preference_strength,
        )

        return result_path, report, stats


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