"""
Data loader for semantic adapter training.

Implements the data pipeline defined in semantic_adapter_trainer_data_spec.md:
- PromptDataset: Text + feature associations
- SemanticBatchSampler: Balanced sampling with timestep generation
- HybridSemanticDataLoader: Main loader with pre-generated latents
"""

import logging
from typing import Dict, List, Optional, Tuple
import torch
import numpy as np

from ..feature_prompts import get_feature_prompts, FEATURE_PROMPTS

logger = logging.getLogger(__name__)


class PromptDataset:
    """
    Dataset for feature-based prompts.

    Stores prompts organized by feature. No randomness, no model awareness.
    Simply maps indices to (prompt, feature_id) pairs.
    """

    def __init__(
        self,
        feature_names: List[str],
        architecture: str = "sd15",
        prompts_per_feature: int = 3,
    ):
        """
        Initialize prompt dataset.

        Args:
            feature_names: List of feature IDs to include
            architecture: Model architecture for prompt formatting
            prompts_per_feature: Number of prompts per feature
        """
        self.feature_names = feature_names
        self.architecture = architecture
        self.prompts_per_feature = prompts_per_feature

        # Build prompt list
        self.prompts: List[str] = []
        self.features: List[str] = []

        for feature in feature_names:
            feature_prompts = get_feature_prompts(
                feature,
                architecture=architecture,
                num_prompts=prompts_per_feature
            )

            self.prompts.extend(feature_prompts)
            self.features.extend([feature] * len(feature_prompts))

        logger.info(
            f"PromptDataset initialized: {len(self.prompts)} prompts "
            f"({len(feature_names)} features × {prompts_per_feature} prompts)"
        )

    def __len__(self) -> int:
        """Get total number of prompts."""
        return len(self.prompts)

    def __getitem__(self, idx: int) -> Tuple[str, str]:
        """
        Get prompt and feature at index.

        Args:
            idx: Index

        Returns:
            Tuple of (prompt, feature_id)
        """
        return self.prompts[idx], self.features[idx]

    def get_prompt(self, idx: int) -> str:
        """Get prompt at index."""
        return self.prompts[idx]

    def get_feature(self, idx: int) -> str:
        """Get feature ID at index."""
        return self.features[idx]


class SemanticBatchSampler:
    """
    Sampler for balanced feature sampling and timestep generation.

    Ensures each batch has equal representation of all features.
    Pre-generates timesteps from a fixed distribution.
    """

    def __init__(
        self,
        dataset: PromptDataset,
        batch_size: int = 12,
        num_batches: int = 10,
        timestep_range: Tuple[int, int] = (0, 999),
        seed: int = 42,
    ):
        """
        Initialize batch sampler.

        Args:
            dataset: PromptDataset to sample from
            batch_size: Batch size (should be divisible by num features)
            num_batches: Total number of batches to generate
            timestep_range: Range of diffusion timesteps (min, max)
            seed: Random seed for reproducibility
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.timestep_range = timestep_range
        self.seed = seed

        # Validate batch size
        num_features = len(dataset.feature_names)
        if batch_size % num_features != 0:
            logger.warning(
                f"Batch size {batch_size} not divisible by {num_features} features. "
                f"Feature balance may be imperfect."
            )

        self.samples_per_feature = batch_size // num_features

        # Pre-generate timesteps
        self.timesteps = self._generate_timesteps()

        logger.info(
            f"SemanticBatchSampler initialized: "
            f"{num_batches} batches × {batch_size} samples, "
            f"{self.samples_per_feature} samples per feature"
        )

    def _generate_timesteps(self) -> torch.Tensor:
        """
        Pre-generate timesteps for all batches.

        Distribution:
        - 33% early steps (600-999)
        - 33% mid steps (300-599)
        - 33% late steps (0-299)

        Returns:
            Tensor of shape [num_batches, batch_size]
        """
        rng = np.random.RandomState(self.seed)
        total_samples = self.num_batches * self.batch_size

        # Split into thirds
        num_early = total_samples // 3
        num_mid = total_samples // 3
        num_late = total_samples - num_early - num_mid  # Remainder

        # Generate timesteps for each range
        t_early = rng.randint(600, 1000, size=num_early)
        t_mid = rng.randint(300, 600, size=num_mid)
        t_late = rng.randint(0, 300, size=num_late)

        # Concatenate and shuffle
        timesteps = np.concatenate([t_early, t_mid, t_late])
        rng.shuffle(timesteps)

        # Reshape to [num_batches, batch_size]
        timesteps = timesteps[:self.num_batches * self.batch_size]
        timesteps = timesteps.reshape(self.num_batches, self.batch_size)

        return torch.from_numpy(timesteps).long()

    def sample_indices(self, batch_idx: int) -> List[int]:
        """
        Sample balanced prompt indices for a batch.

        Uses round-robin sampling to ensure equal feature representation.

        Args:
            batch_idx: Batch index

        Returns:
            List of prompt indices
        """
        rng = np.random.RandomState(self.seed + batch_idx)

        # Group dataset indices by feature
        feature_indices: Dict[str, List[int]] = {
            feature: [] for feature in self.dataset.feature_names
        }

        for idx in range(len(self.dataset)):
            feature = self.dataset.get_feature(idx)
            feature_indices[feature].append(idx)

        # Sample from each feature
        sampled_indices = []
        for feature in self.dataset.feature_names:
            available = feature_indices[feature]
            if len(available) == 0:
                logger.warning(f"No prompts available for feature '{feature}'")
                continue

            # Sample with replacement if needed
            samples = rng.choice(
                available,
                size=self.samples_per_feature,
                replace=len(available) < self.samples_per_feature
            )
            sampled_indices.extend(samples.tolist())

        # Shuffle to avoid feature clustering
        rng.shuffle(sampled_indices)

        return sampled_indices[:self.batch_size]

    def sample_timesteps(self, batch_idx: int) -> torch.Tensor:
        """
        Get pre-generated timesteps for a batch.

        Args:
            batch_idx: Batch index

        Returns:
            Tensor of shape [batch_size]
        """
        return self.timesteps[batch_idx]


class HybridSemanticDataLoader:
    """
    Main data loader for semantic adapter training.

    Pre-generates latents and timesteps for determinism.
    Yields batches containing:
    - prompts: List[str]
    - features: List[str]
    - timesteps: Tensor[B]
    - latents: Tensor[B, 4, H, W]
    """

    def __init__(
        self,
        feature_names: List[str],
        architecture: str = "sd15",
        prompts_per_feature: int = 3,
        batch_size: int = 12,
        num_batches: int = 10,
        latent_height: int = 64,
        latent_width: int = 96,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
        seed: int = 42,
    ):
        """
        Initialize data loader.

        Args:
            feature_names: List of feature IDs
            architecture: Model architecture
            prompts_per_feature: Number of prompts per feature
            batch_size: Batch size
            num_batches: Number of batches per epoch
            latent_height: Latent height (64 for 512px)
            latent_width: Latent width (96 for 768px)
            device: Device for tensors
            dtype: Data type for tensors
            seed: Random seed
        """
        self.feature_names = feature_names
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.device = device
        self.dtype = dtype
        self.seed = seed

        # Track current epoch for metadata (can be updated by trainer)
        self.current_epoch = 0

        # Create dataset
        self.dataset = PromptDataset(
            feature_names=feature_names,
            architecture=architecture,
            prompts_per_feature=prompts_per_feature,
        )

        # Create sampler
        self.sampler = SemanticBatchSampler(
            dataset=self.dataset,
            batch_size=batch_size,
            num_batches=num_batches,
            seed=seed,
        )

        # Pre-generate latents
        self.latents = self._generate_latents(latent_height, latent_width)

        logger.info(
            f"HybridSemanticDataLoader initialized: "
            f"{num_batches} batches, "
            f"latent shape: {self.latents.shape}"
        )

    def _generate_latents(self, height: int, width: int) -> torch.Tensor:
        """
        Pre-generate latent noise for all batches.

        Args:
            height: Latent height
            width: Latent width

        Returns:
            Tensor of shape [num_batches, batch_size, 4, height, width]
        """
        torch.manual_seed(self.seed)

        shape = (self.num_batches, self.batch_size, 4, height, width)
        latents = torch.randn(shape, dtype=self.dtype, device=self.device)

        logger.info(f"Pre-generated {self.num_batches * self.batch_size} latent tensors")

        return latents

    def __len__(self) -> int:
        """Get number of batches."""
        return self.num_batches

    def set_epoch(self, epoch: int):
        """
        Set current epoch for metadata tracking.

        Args:
            epoch: Current epoch number
        """
        self.current_epoch = epoch

    def __iter__(self):
        """
        Iterate over batches.

        Yields:
            Dict with keys:
            - prompts: List[str]
            - features: List[str]
            - timesteps: Tensor[batch_size]
            - latents: Tensor[batch_size, 4, H, W]
            - meta: Dict with diagnostic metadata
        """
        for batch_idx in range(self.num_batches):
            # Sample prompt indices
            indices = self.sampler.sample_indices(batch_idx)

            # Fetch prompts and features
            prompts = [self.dataset.get_prompt(idx) for idx in indices]
            features = [self.dataset.get_feature(idx) for idx in indices]

            # Get timesteps
            timesteps = self.sampler.sample_timesteps(batch_idx)

            # Get latents
            latents = self.latents[batch_idx]

            # Compute feature distribution for diagnostics
            feature_counts = {}
            for feature in features:
                feature_counts[feature] = feature_counts.get(feature, 0) + 1

            # Create metadata for diagnostics and debugging
            metadata = {
                "epoch": self.current_epoch,
                "batch_idx": batch_idx,
                "prompt_indices": indices,
                "latent_ids": list(range(
                    batch_idx * self.batch_size,
                    (batch_idx + 1) * self.batch_size
                )),
                "num_samples": len(prompts),
                "feature_distribution": feature_counts,
                "seed": self.seed,
            }

            yield {
                "prompts": prompts,
                "features": features,
                "timesteps": timesteps,
                "latents": latents,
                "meta": metadata,
            }

    @staticmethod
    def format_batch_metadata(batch: Dict) -> str:
        """
        Format batch metadata for logging/debugging.

        Args:
            batch: Batch dict with 'meta' key

        Returns:
            Formatted string with metadata info
        """
        if "meta" not in batch:
            return "No metadata available"

        meta = batch["meta"]
        lines = [
            f"Batch Metadata:",
            f"  Epoch: {meta['epoch']}",
            f"  Batch: {meta['batch_idx']}",
            f"  Samples: {meta['num_samples']}",
            f"  Feature distribution: {meta['feature_distribution']}",
            f"  Latent IDs: {meta['latent_ids'][0]} - {meta['latent_ids'][-1]}",
            f"  Seed: {meta['seed']}",
        ]
        return "\n".join(lines)

    def get_statistics(self) -> Dict:
        """
        Get data loader statistics.

        Returns:
            Dict with loader configuration and statistics
        """
        return {
            "num_features": len(self.feature_names),
            "features": self.feature_names,
            "batch_size": self.batch_size,
            "num_batches": self.num_batches,
            "total_samples_per_epoch": self.batch_size * self.num_batches,
            "latent_shape": tuple(self.latents.shape[1:]),
            "device": str(self.device),
            "dtype": str(self.dtype),
            "seed": self.seed,
            "current_epoch": self.current_epoch,
        }

    def __repr__(self) -> str:
        return (
            f"HybridSemanticDataLoader("
            f"features={len(self.feature_names)}, "
            f"batches={self.num_batches}, "
            f"batch_size={self.batch_size})"
        )
