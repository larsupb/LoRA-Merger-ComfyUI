"""
Semantic importance map caching system.

Caches gradient-based attribution results to avoid recomputing expensive analysis.
"""

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Dict, Optional, Any

import torch

logger = logging.getLogger(__name__)


class SemanticMapCache:
    """
    Cache for semantic importance maps.

    Stores and retrieves semantic attribution results for LoRAs.
    Maps are indexed by LoRA hash and feature configuration.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize cache.

        Args:
            cache_dir: Directory to store cache files. If None, uses default location.
        """
        if cache_dir is None:
            # Default: custom_nodes/LoRA-Merger-ComfyUI/cache/semantic_maps/
            cache_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                "cache",
                "semantic_maps",
            )

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Semantic map cache initialized at: {self.cache_dir}")

    def _compute_lora_hash(self, lora_path: str) -> str:
        """
        Compute unique hash for a LoRA.

        Args:
            lora_path: Path to LoRA file

        Returns:
            SHA256 hash string
        """
        # Hash based on file path and sum of all weight values
        hasher = hashlib.sha256()

        # Add file path
        hasher.update(lora_path.encode("utf-8"))

        return hasher.hexdigest()

    def _compute_config_hash(self, config: Dict[str, Any]) -> str:
        """
        Compute hash for analysis configuration.

        Args:
            config: Configuration dict (features, prompts, etc.)

        Returns:
            SHA256 hash string
        """
        # Sort keys for deterministic hashing
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.sha256(config_str.encode("utf-8")).hexdigest()[:16]

    def _get_cache_path(self, lora_hash: str, config_hash: str) -> Path:
        """Get path to cache file."""
        filename = f"{lora_hash}_{config_hash}.pt"
        return self.cache_dir / filename

    def _get_metadata_path(self, lora_hash: str, config_hash: str) -> Path:
        """Get path to metadata file."""
        filename = f"{lora_hash}_{config_hash}_meta.json"
        return self.cache_dir / filename

    def get(
        self,
        lora_path: str,
        config: Dict[str, Any],
    ) -> Optional[Dict[str, Dict[str, torch.Tensor]]]:
        """
        Retrieve cached semantic map.

        Args:
            lora_path: Path to LoRA file
            config: Analysis configuration

        Returns:
            Semantic importance map if cached, None otherwise
            Format: {feature: {layer_key: importance_tensor}}
        """
        lora_hash = self._compute_lora_hash(lora_path)
        config_hash = self._compute_config_hash(config)

        cache_path = self._get_cache_path(lora_hash, config_hash)

        if not cache_path.exists():
            logger.debug(f"Cache miss for {lora_path}")
            return None

        try:
            semantic_map = torch.load(cache_path, map_location="cpu", weights_only=True)
            logger.info(f"Cache hit for {lora_path}")
            return semantic_map
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            return None

    def set(
        self,
        lora_path: str,
        config: Dict[str, Any],
        semantic_map: Dict[str, Dict[str, torch.Tensor]],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Store semantic map in cache.

        Args:
            lora_path: Path to LoRA file
            config: Analysis configuration
            semantic_map: Semantic importance map to cache
            metadata: Optional metadata (analysis stats, timing, etc.)
        """
        lora_hash = self._compute_lora_hash(lora_path)
        config_hash = self._compute_config_hash(config)

        cache_path = self._get_cache_path(lora_hash, config_hash)
        metadata_path = self._get_metadata_path(lora_hash, config_hash)

        try:
            # Save semantic map
            torch.save(semantic_map, cache_path)

            # Save metadata
            if metadata is not None:
                with open(metadata_path, "w") as f:
                    json.dump(metadata, f, indent=2)

            logger.info(f"Cached semantic map for {lora_path}")
        except Exception as e:
            logger.error(f"Failed to cache semantic map: {e}")

    def clear(self) -> None:
        """Clear all cached semantic maps."""
        for file in self.cache_dir.glob("*.pt"):
            file.unlink()
        for file in self.cache_dir.glob("*.json"):
            file.unlink()
        logger.info("Cleared semantic map cache")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        pt_files = list(self.cache_dir.glob("*.pt"))
        total_size = sum(f.stat().st_size for f in pt_files)

        return {
            "num_cached_maps": len(pt_files),
            "total_size_mb": total_size / (1024 * 1024),
            "cache_dir": str(self.cache_dir),
        }


# Global cache instance
_global_cache: Optional[SemanticMapCache] = None


def get_cache() -> SemanticMapCache:
    """Get global cache instance."""
    global _global_cache
    if _global_cache is None:
        _global_cache = SemanticMapCache()
    return _global_cache
