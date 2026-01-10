"""
Merge statistics and importance distribution reporting.

Captures detailed information about which LoRAs contributed to which
features/layers during semantic merge, with multiple aggregation views.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from collections import defaultdict
from enum import Enum

from src.semantic.key_utils import NormalizedKey

logger = logging.getLogger(__name__)


class LayerType(Enum):
    """Classification of layer types."""
    ATTENTION_SELF = "self_attention"
    ATTENTION_CROSS = "cross_attention"
    MLP = "mlp"
    CONV = "conv"
    PROJ = "projection"
    OTHER = "other"


class DepthBucket(Enum):
    """Depth classification buckets."""
    EARLY = "early"      # 0-33%: composition, structure
    MIDDLE = "middle"    # 33-66%: main features
    LATE = "late"        # 66-100%: fine details


@dataclass
class LayerContribution:
    """Single layer contribution from one LoRA for one feature."""
    lora_name: str
    feature: str
    importance: float
    feature_weight: float
    effective_weight: float
    normalized_share: float
    layer_type: LayerType
    depth_bucket: DepthBucket
    relative_depth: float
    was_matched: bool = True
    is_exclusive: bool = False  # NEW


def record_contribution(
        self,
        layer_key: str,
        lora_name: str,
        feature: str,
        importance: float,
        feature_weight: float,
        effective_weight: float,  # ← NEW: pass pre-computed effective weight
        layer_type: LayerType,
        relative_depth: float,
) -> None:
    """Record a single contribution during merge."""
    if relative_depth < 0.33:
        depth_bucket = DepthBucket.EARLY
    elif relative_depth < 0.66:
        depth_bucket = DepthBucket.MIDDLE
    else:
        depth_bucket = DepthBucket.LATE

    contribution = LayerContribution(
        lora_name=lora_name,
        feature=feature,
        importance=importance,
        feature_weight=feature_weight,
        effective_weight=effective_weight,  # Use passed value
        normalized_share=0.0,
        layer_type=layer_type,
        depth_bucket=depth_bucket,
        relative_depth=relative_depth,
    )

    layer_key_str = str(layer_key)
    self.layer_contributions[layer_key_str].append(contribution)

    if lora_name not in self.source_loras:
        self.source_loras.append(lora_name)
    if feature not in self.features:
        self.features.append(feature)


@dataclass
class MergeStatistics:
    """
    Complete statistics for a semantic merge operation.

    Captures per-layer detail and provides multiple aggregation views.
    """
    # Metadata
    source_loras: List[str] = field(default_factory=list)
    features: List[str] = field(default_factory=list)
    total_layers: int = 0
    lambda_value: float = 1.0

    # Per-layer detail
    # Structure: {layer_key: [LayerContribution, ...]}
    layer_contributions: Dict[str, List[LayerContribution]] = field(
        default_factory=lambda: defaultdict(list)
    )

    # Tracking for aggregation
    _finalized: bool = False

    # Aggregated views (computed on finalize)
    by_feature: Dict[str, Dict[str, float]] = field(default_factory=dict)
    by_lora: Dict[str, Dict[str, float]] = field(default_factory=dict)
    by_layer_type: Dict[str, Dict[str, float]] = field(default_factory=dict)
    by_depth: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Additional insights
    dominant_layers: Dict[str, str] = field(default_factory=dict)  # {layer: dominant_lora}
    contested_layers: List[str] = field(default_factory=list)  # Layers with no clear winner
    coverage: Dict[str, Dict[str, int]] = field(default_factory=dict)  # {lora: {feature: count}}

    # Track importance values
    importance_histogram: Dict[str, list] = field(default_factory=lambda: defaultdict(list))
    match_count: int = 0
    fallback_count: int = 0

    # Track unmatched keys for debugging
    unmatched_samples: Dict[str, list] = field(default_factory=lambda: defaultdict(list))

    # NEW: Track exclusive features
    exclusive_features: Dict[str, str] = field(default_factory=dict)  # {feature: lora}
    feature_weights: Dict[str, float] = field(default_factory=dict)  # {feature: weight}

    def record_unmatched(self, norm_key: 'NormalizedKey', lora_name: str, feature: str):
        """Record a sample of unmatched keys for debugging."""
        key = f"{lora_name}_{feature}"
        if len(self.unmatched_samples[key]) < 5:  # Keep only 5 samples per combination
            self.unmatched_samples[key].append(str(norm_key))

    def record_contribution(
            self,
            layer_key: str,
            lora_name: str,
            feature: str,
            importance: float,
            feature_weight: float,
            effective_weight: float,
            layer_type: LayerType,
            relative_depth: float,
            was_matched: bool = True,
            is_exclusive: bool = False,  # NEW
    ) -> None:
        """Record a single contribution during merge."""
        if relative_depth < 0.33:
            depth_bucket = DepthBucket.EARLY
        elif relative_depth < 0.66:
            depth_bucket = DepthBucket.MIDDLE
        else:
            depth_bucket = DepthBucket.LATE

        contribution = LayerContribution(
            lora_name=lora_name,
            feature=feature,
            importance=importance,
            feature_weight=feature_weight,
            effective_weight=effective_weight,
            normalized_share=0.0,
            layer_type=layer_type,
            depth_bucket=depth_bucket,
            relative_depth=relative_depth,
            was_matched=was_matched,
            is_exclusive=is_exclusive,
        )

        layer_key_str = str(layer_key)
        self.layer_contributions[layer_key_str].append(contribution)

        # Track metadata
        if lora_name not in self.source_loras:
            self.source_loras.append(lora_name)
        if feature not in self.features:
            self.features.append(feature)

        # Track feature weights
        self.feature_weights[feature] = feature_weight

        # Track exclusive features
        if is_exclusive and effective_weight > 0:
            self.exclusive_features[feature] = lora_name

        # Track matching
        if was_matched:
            self.match_count += 1
        else:
            self.fallback_count += 1

        # Track importance distribution
        self.importance_histogram[f"{lora_name}_{feature}"].append(importance)

    def finalize(self) -> None:
        """
        Compute all aggregated statistics after merge is complete.
        """
        if self._finalized:
            return

        self.total_layers = len(self.layer_contributions)

        # Compute normalized shares per layer
        self._compute_normalized_shares()

        # Compute aggregations
        self._compute_by_feature()
        self._compute_by_lora()
        self._compute_by_layer_type()
        self._compute_by_depth()
        self._compute_dominance()
        self._compute_coverage()

        self._finalized = True

    def _compute_normalized_shares(self) -> None:
        """Compute normalized share (0-1) for each contribution within its layer."""
        for layer_key, contributions in self.layer_contributions.items():
            total = sum(c.effective_weight for c in contributions)
            if total > 0:
                for c in contributions:
                    # Modify in place (dataclass is mutable)
                    object.__setattr__(c, 'normalized_share', c.effective_weight / total)

    def _compute_by_feature(self) -> None:
        """Aggregate: LoRA contribution percentage per feature."""
        # {feature: {lora: sum_of_normalized_shares}}
        raw = {f: defaultdict(float) for f in self.features}
        counts = {f: defaultdict(int) for f in self.features}

        for contributions in self.layer_contributions.values():
            for c in contributions:
                raw[c.feature][c.lora_name] += c.normalized_share
                counts[c.feature][c.lora_name] += 1

        # Average per feature (divide by number of layers this lora contributed to)
        self.by_feature = {}
        for feature in self.features:
            if not raw[feature]:
                continue
            # Normalize so percentages sum to 100%
            total = sum(raw[feature].values())
            if total > 0:
                self.by_feature[feature] = {
                    lora: share / total
                    for lora, share in raw[feature].items()
                }

    def _compute_by_lora(self) -> None:
        """Aggregate: Feature distribution per LoRA."""
        # {lora: {feature: sum_of_effective_weights}}
        raw = {lora: defaultdict(float) for lora in self.source_loras}

        for contributions in self.layer_contributions.values():
            for c in contributions:
                raw[c.lora_name][c.feature] += c.effective_weight

        # Normalize per LoRA
        self.by_lora = {}
        for lora in self.source_loras:
            total = sum(raw[lora].values())
            if total > 0:
                self.by_lora[lora] = {
                    feature: weight / total
                    for feature, weight in raw[lora].items()
                }

    def _compute_by_layer_type(self) -> None:
        """Aggregate: LoRA contribution per layer type."""
        raw = {lt.value: defaultdict(float) for lt in LayerType}

        for contributions in self.layer_contributions.values():
            for c in contributions:
                raw[c.layer_type.value][c.lora_name] += c.effective_weight

        self.by_layer_type = {}
        for lt, lora_weights in raw.items():
            total = sum(lora_weights.values())
            if total > 0:
                self.by_layer_type[lt] = {
                    lora: w / total for lora, w in lora_weights.items()
                }

    def _compute_by_depth(self) -> None:
        """Aggregate: LoRA contribution per depth bucket."""
        raw = {db.value: defaultdict(float) for db in DepthBucket}

        for contributions in self.layer_contributions.values():
            for c in contributions:
                raw[c.depth_bucket.value][c.lora_name] += c.effective_weight

        self.by_depth = {}
        for db, lora_weights in raw.items():
            total = sum(lora_weights.values())
            if total > 0:
                self.by_depth[db] = {
                    lora: w / total for lora, w in lora_weights.items()
                }

    def _compute_dominance(self) -> None:
        """Find layers dominated by one LoRA (>60% share)."""
        self.dominant_layers = {}
        self.contested_layers = []

        for layer_key, contributions in self.layer_contributions.items():
            # Sum shares by LoRA
            lora_shares = defaultdict(float)
            for c in contributions:
                lora_shares[c.lora_name] += c.normalized_share

            # Find dominant LoRA
            if lora_shares:
                max_lora = max(lora_shares.keys(), key=lambda l: lora_shares[l])
                max_share = lora_shares[max_lora]

                if max_share > 0.6:
                    self.dominant_layers[layer_key] = max_lora
                elif max_share < 0.4:
                    self.contested_layers.append(layer_key)

    def _compute_coverage(self) -> None:
        """Count layers each LoRA contributed to per feature."""
        self.coverage = {lora: defaultdict(int) for lora in self.source_loras}

        for contributions in self.layer_contributions.values():
            seen = set()  # Avoid double counting same (lora, feature) in one layer
            for c in contributions:
                key = (c.lora_name, c.feature)
                if key not in seen and c.effective_weight > 0.01:
                    self.coverage[c.lora_name][c.feature] += 1
                    seen.add(key)

    # === Query Methods ===

    def get_feature_breakdown(self, feature: str) -> Dict[str, Any]:
        """Get detailed breakdown for a specific feature."""
        if not self._finalized:
            self.finalize()

        if feature not in self.by_feature:
            return {"error": f"Feature '{feature}' not found"}

        # Detailed breakdown by layer type and depth for this feature
        by_layer_type = defaultdict(lambda: defaultdict(float))
        by_depth = defaultdict(lambda: defaultdict(float))

        for contributions in self.layer_contributions.values():
            for c in contributions:
                if c.feature == feature:
                    by_layer_type[c.layer_type.value][c.lora_name] += c.effective_weight
                    by_depth[c.depth_bucket.value][c.lora_name] += c.effective_weight

        # Normalize
        for lt in by_layer_type:
            total = sum(by_layer_type[lt].values())
            if total > 0:
                by_layer_type[lt] = {l: w/total for l, w in by_layer_type[lt].items()}

        for db in by_depth:
            total = sum(by_depth[db].values())
            if total > 0:
                by_depth[db] = {l: w/total for l, w in by_depth[db].items()}

        return {
            "feature": feature,
            "overall": self.by_feature[feature],
            "by_layer_type": dict(by_layer_type),
            "by_depth": dict(by_depth),
            "coverage": {lora: self.coverage[lora].get(feature, 0) for lora in self.source_loras},
        }

    def get_layer_breakdown(self, layer_key: str) -> Dict[str, Any]:
        """Get detailed breakdown for a specific layer."""
        if not self._finalized:
            self.finalize()

        layer_key_str = str(layer_key)
        if layer_key_str not in self.layer_contributions:
            return {"error": f"Layer '{layer_key}' not found"}

        contributions = self.layer_contributions[layer_key_str]

        # Group by feature
        by_feature = defaultdict(list)
        for c in contributions:
            by_feature[c.feature].append({
                "lora": c.lora_name,
                "importance": c.importance,
                "feature_weight": c.feature_weight,
                "effective_weight": c.effective_weight,
                "share": c.normalized_share,
            })

        # Overall lora shares
        lora_totals = defaultdict(float)
        for c in contributions:
            lora_totals[c.lora_name] += c.normalized_share

        return {
            "layer": layer_key_str,
            "layer_type": contributions[0].layer_type.value if contributions else "unknown",
            "depth": contributions[0].relative_depth if contributions else 0.0,
            "lora_shares": dict(lora_totals),
            "by_feature": dict(by_feature),
            "dominant": self.dominant_layers.get(layer_key_str),
        }

    # === Formatting ===

    def format_report(self, verbose: bool = False, top_n_layers: int = 10) -> str:
        """Format human-readable merge report."""
        if not self._finalized:
            self.finalize()

        lines = [
            "=" * 70,
            "SEMANTIC MERGE STATISTICS REPORT",
            "=" * 70,
            "",
            f"Source LoRAs: {', '.join(self.source_loras)}",
            f"Features: {', '.join(self.features)}",
            f"Total Layers Merged: {self.total_layers}",
            f"Lambda: {self.lambda_value}",
            "",
        ]

        # === FEATURE CONFIGURATION ===
        lines.append("-" * 70)
        lines.append("FEATURE CONFIGURATION")
        lines.append("-" * 70)

        for feature in self.features:
            weight = self.feature_weights.get(feature, 1.0)
            exclusive_lora = self.exclusive_features.get(feature)

            if exclusive_lora:
                lines.append(f"  {feature}: weight={weight:.2f} [EXCLUSIVE → {exclusive_lora}]")
            else:
                lines.append(f"  {feature}: weight={weight:.2f}")

        lines.append("")

        # === SEMANTIC MAP MATCHING ===
        lines.append("-" * 70)
        lines.append("SEMANTIC MAP MATCHING")
        lines.append("-" * 70)

        total_contributions = self.match_count + self.fallback_count
        if total_contributions > 0:
            match_pct = self.match_count / total_contributions * 100
            lines.append(f"  Matched from semantic maps: {self.match_count} ({match_pct:.1f}%)")
            lines.append(f"  Fallback to default (1.0): {self.fallback_count} ({100 - match_pct:.1f}%)")

        lines.append("")
        lines.append("  Importance value distribution:")
        for key, values in sorted(self.importance_histogram.items()):
            if values:
                # Filter out zero values (from exclusive mode)
                non_zero = [v for v in values if v > 0]
                if non_zero:
                    min_v, max_v = min(non_zero), max(non_zero)
                    avg_v = sum(non_zero) / len(non_zero)
                    lines.append(f"    {key}: min={min_v:.3f}, max={max_v:.3f}, avg={avg_v:.3f}")
                else:
                    lines.append(f"    {key}: excluded (exclusive mode)")

        lines.append("")

        # === BY FEATURE ===
        lines.append("-" * 70)
        lines.append("BY FEATURE (which LoRA dominates each feature)")
        lines.append("-" * 70)

        for feature in self.features:
            if feature not in self.by_feature:
                continue

            contribs = self.by_feature[feature]
            sorted_contribs = sorted(contribs.items(), key=lambda x: -x[1])

            # Visual bar
            bar = self._make_bar(sorted_contribs)
            contrib_str = ", ".join(f"{lora}: {pct * 100:.1f}%" for lora, pct in sorted_contribs)

            # Add weight and exclusive indicator
            weight = self.feature_weights.get(feature, 1.0)
            excl = " [EXCLUSIVE]" if feature in self.exclusive_features else ""
            weight_str = f" (weight={weight:.1f})" if weight != 1.0 else ""

            lines.append(f"  {feature}{weight_str}{excl}:")
            lines.append(f"    {bar}")
            lines.append(f"    {contrib_str}")

        lines.append("")

        # === BY LORA ===
        lines.append("-" * 70)
        lines.append("BY LORA (feature distribution for each LoRA)")
        lines.append("-" * 70)

        for lora in self.source_loras:
            if lora not in self.by_lora:
                continue
            dist = self.by_lora[lora]
            sorted_dist = sorted(dist.items(), key=lambda x: -x[1])
            dist_str = ", ".join(f"{feat}: {pct*100:.1f}%" for feat, pct in sorted_dist)
            lines.append(f"  {lora}: {dist_str}")

        lines.append("")

        # === BY DEPTH ===
        lines.append("-" * 70)
        lines.append("BY DEPTH (LoRA influence at different network depths)")
        lines.append("-" * 70)
        lines.append("  early (0-33%):  composition, overall structure")
        lines.append("  middle (33-66%): main features (face, hair)")
        lines.append("  late (66-100%):  fine details, textures")
        lines.append("")

        for depth in ["early", "middle", "late"]:
            if depth not in self.by_depth:
                continue
            contribs = self.by_depth[depth]
            sorted_contribs = sorted(contribs.items(), key=lambda x: -x[1])
            bar = self._make_bar(sorted_contribs)
            contrib_str = ", ".join(f"{lora}: {pct*100:.1f}%" for lora, pct in sorted_contribs)
            lines.append(f"  {depth}:")
            lines.append(f"    {bar}")
            lines.append(f"    {contrib_str}")

        lines.append("")

        # === BY LAYER TYPE ===
        lines.append("-" * 70)
        lines.append("BY LAYER TYPE")
        lines.append("-" * 70)

        for lt in ["self_attention", "cross_attention", "mlp", "projection", "conv"]:
            if lt not in self.by_layer_type:
                continue
            contribs = self.by_layer_type[lt]
            sorted_contribs = sorted(contribs.items(), key=lambda x: -x[1])
            contrib_str = ", ".join(f"{lora}: {pct*100:.1f}%" for lora, pct in sorted_contribs)
            lines.append(f"  {lt}: {contrib_str}")

        lines.append("")

        # === DOMINANCE ===
        lines.append("-" * 70)
        lines.append("LAYER DOMINANCE (>60% contribution from single LoRA)")
        lines.append("-" * 70)

        dom_counts = defaultdict(int)
        for lora in self.dominant_layers.values():
            dom_counts[lora] += 1

        for lora, count in sorted(dom_counts.items(), key=lambda x: -x[1]):
            pct = count / max(self.total_layers, 1) * 100
            lines.append(f"  {lora}: {count}/{self.total_layers} layers ({pct:.1f}%)")

        contested = len(self.contested_layers)
        mixed = self.total_layers - len(self.dominant_layers) - contested
        lines.append(f"  (mixed 40-60%): {mixed} layers")
        lines.append(f"  (contested <40%): {contested} layers")

        lines.append("")

        # === COVERAGE ===
        lines.append("-" * 70)
        lines.append("COVERAGE (layers where each LoRA contributed per feature)")
        lines.append("-" * 70)

        for lora in self.source_loras:
            cov = self.coverage.get(lora, {})
            cov_str = ", ".join(f"{feat}: {count}" for feat, count in sorted(cov.items()))
            lines.append(f"  {lora}: {cov_str}")

        lines.append("")
        lines.append("=" * 70)

        # === VERBOSE: Top layers by controversy ===
        if verbose:
            lines.append("")
            lines.append("TOP CONTESTED LAYERS (most evenly split)")
            lines.append("-" * 70)

            # Find layers with most even splits
            layer_scores = []
            for layer_key, contributions in self.layer_contributions.items():
                lora_shares = defaultdict(float)
                for c in contributions:
                    lora_shares[c.lora_name] += c.normalized_share

                if len(lora_shares) >= 2:
                    shares = sorted(lora_shares.values(), reverse=True)
                    # Score: closer to 0.5/0.5 = higher score
                    evenness = 1.0 - abs(shares[0] - shares[1])
                    layer_scores.append((layer_key, evenness, dict(lora_shares)))

            layer_scores.sort(key=lambda x: -x[1])

            for layer_key, evenness, shares in layer_scores[:top_n_layers]:
                shares_str = ", ".join(f"{l}: {s*100:.1f}%" for l, s in sorted(shares.items(), key=lambda x: -x[1]))
                lines.append(f"  {layer_key}")
                lines.append(f"    {shares_str}")
        return "\n".join(lines)

    def _make_bar(self, sorted_contribs: List[tuple], width: int = 40) -> str:
        """Create ASCII bar visualization."""
        if not sorted_contribs:
            return ""

        # Use different characters for different LoRAs
        chars = ["█", "▓", "▒", "░", "·"]

        bar = ""
        for i, (lora, pct) in enumerate(sorted_contribs):
            char = chars[i % len(chars)]
            segment_len = int(pct * width)
            bar += char * segment_len

        # Pad to full width
        bar = bar[:width].ljust(width, "·")
        return f"[{bar}]"

    def to_dict(self) -> Dict[str, Any]:
        """Export statistics as dictionary (JSON-serializable)."""
        if not self._finalized:
            self.finalize()

        # Convert layer_contributions to serializable format
        layer_details = {}
        for layer_key, contributions in self.layer_contributions.items():
            layer_details[layer_key] = [
                {
                    "lora": c.lora_name,
                    "feature": c.feature,
                    "importance": c.importance,
                    "feature_weight": c.feature_weight,
                    "effective_weight": c.effective_weight,
                    "normalized_share": c.normalized_share,
                    "layer_type": c.layer_type.value,
                    "depth_bucket": c.depth_bucket.value,
                    "relative_depth": c.relative_depth,
                }
                for c in contributions
            ]

        return {
            "metadata": {
                "source_loras": self.source_loras,
                "features": self.features,
                "total_layers": self.total_layers,
                "lambda_value": self.lambda_value,
            },
            "aggregations": {
                "by_feature": self.by_feature,
                "by_lora": self.by_lora,
                "by_layer_type": self.by_layer_type,
                "by_depth": self.by_depth,
            },
            "insights": {
                "dominant_layers": self.dominant_layers,
                "contested_layers": self.contested_layers,
                "coverage": {k: dict(v) for k, v in self.coverage.items()},
            },
            "layer_details": layer_details,
        }

    def to_json(self, indent: int = 2) -> str:
        """Export as JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


# === Helper Functions ===

def classify_layer_type(layer_key: str) -> LayerType:
    """Classify a layer key into LayerType."""
    key_str = str(layer_key).lower()

    if "attn1" in key_str or "self_attn" in key_str:
        return LayerType.ATTENTION_SELF
    elif "attn2" in key_str or "cross_attn" in key_str:
        return LayerType.ATTENTION_CROSS
    elif any(x in key_str for x in ["attn", "attention"]):
        return LayerType.ATTENTION_SELF  # Default attention type
    elif any(x in key_str for x in ["mlp", "ff", "feed_forward", "ffn"]):
        return LayerType.MLP
    elif any(x in key_str for x in ["conv", "downsample", "upsample"]):
        return LayerType.CONV
    elif any(x in key_str for x in ["proj_in", "proj_out", "to_q", "to_k", "to_v", "to_out"]):
        return LayerType.PROJ
    else:
        return LayerType.OTHER


def compute_relative_depth(layer_idx: int, total_layers: int) -> float:
    """Compute relative depth (0.0 to 1.0) from layer index."""
    if total_layers <= 1:
        return 0.5
    return layer_idx / (total_layers - 1)