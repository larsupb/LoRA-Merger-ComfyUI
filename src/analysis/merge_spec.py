from dataclasses import dataclass
from typing import Dict, Any

import logging

logger = logging.getLogger(__name__)

@dataclass
class MergeSpec:
    """Specification for semantic merge."""
    feature_assignments: Dict[str, Dict[str, Any]]  # {feature: {"source": lora_name, "weight": float}}

    @classmethod
    def from_text(cls, specification: str) -> "MergeSpec":
        """
        Parse merge specification from text.

        Format: "feature from lora_name [weight], feature2 from lora2 [weight], ..."

        Examples:
            "hair from character1, clothing from character2"
            "hair from lora1 1.0, eyes from lora1 0.8, clothing from lora2 1.2"
        """
        assignments = {}

        for assignment in specification.split(","):
            assignment = assignment.strip()
            if not assignment:
                continue

            parts = assignment.split()

            if len(parts) < 3 or parts[1].lower() != "from":
                logger.warning(f"Skipping invalid assignment: {assignment}")
                continue

            feature = parts[0].lower()
            lora_name = parts[2]

            weight = 1.0
            if len(parts) >= 4:
                try:
                    weight = float(parts[3])
                except ValueError:
                    logger.warning(f"Invalid weight in '{assignment}', using 1.0")

            assignments[feature] = {
                "source": lora_name,
                "weight": weight,
            }

        if not assignments:
            raise ValueError("No valid feature assignments found in specification")

        return cls(feature_assignments=assignments)