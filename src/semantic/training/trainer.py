"""
Main training loop for semantic adapters.

Trains all adapters jointly using a global objective that aggregates
losses across layers and timesteps.
"""

import logging
from typing import Dict, Optional
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..adapter.registry import AdapterRegistry
from ..adapter.hook_manager import ForwardHookManager
from .losses import SemanticAdapterLoss
from .curriculum import TrainingCurriculum
from .delta_provider import DeltaProvider, SyntheticDeltaProvider
from ...utils.progress import ThreadSafeProgressBar


class AdapterTrainer:
    """
    Main training loop for semantic adapters.

    Trains all adapters jointly using a global objective that
    aggregates losses across layers and timesteps.

    Note: This is a foundational implementation. Full training with UNet
    forward pass hooks will be implemented in Phase 3 (UNet Hook Integration).
    """

    def __init__(
        self,
        adapter_registry: AdapterRegistry,
        base_model: nn.Module,
        lora_stack: Dict,
        semantic_merger,  # SemanticMerger instance
        feature_prompts: list,
        device: torch.device,
        dtype: torch.dtype,
        use_hooks: bool = False,
        delta_provider: Optional[DeltaProvider] = None,
    ):
        """
        Initialize adapter trainer.

        Args:
            adapter_registry: Registry containing all adapters
            base_model: Base diffusion model (UNet)
            lora_stack: LoRA stack
            semantic_merger: SemanticMerger for teacher targets
            feature_prompts: List of prompts for training
            device: Compute device
            dtype: Compute dtype
            use_hooks: Whether to use forward hooks for training (experimental)
            delta_provider: Optional DeltaProvider (defaults to SyntheticDeltaProvider)
        """
        self.adapter_registry = adapter_registry
        self.base_model = base_model
        self.lora_stack = lora_stack
        self.semantic_merger = semantic_merger
        self.feature_prompts = feature_prompts
        self.device = device
        self.dtype = dtype
        self.use_hooks = use_hooks

        # Initialize delta provider (defaults to synthetic)
        if delta_provider is None:
            self.delta_provider = SyntheticDeltaProvider(
                adapter_registry=adapter_registry,
                device=device,
                dtype=dtype,
                delta_magnitude=0.1,
            )
            logging.info("Using SyntheticDeltaProvider (default)")
        else:
            self.delta_provider = delta_provider
            logging.info(f"Using custom DeltaProvider: {type(delta_provider).__name__}")

        # Freeze base model and LoRAs
        self.base_model.eval()
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Initialize training components
        self.curriculum = TrainingCurriculum()
        self.loss_fn = SemanticAdapterLoss()
        self.optimizer = self._create_optimizer()

        # Hook manager (created on demand)
        self.hook_manager: Optional[ForwardHookManager] = None
        if use_hooks:
            self._setup_hooks()

        # Training statistics
        self.training_stats = {
            "losses": [],
            "adapter_stats": [],
        }

        logging.info(
            f"Initialized AdapterTrainer with {len(adapter_registry)} adapters"
            f"{' (hooks enabled)' if use_hooks else ''}"
        )

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """
        Create optimizer for adapter parameters.

        Returns:
            AdamW optimizer
        """
        lr = self.curriculum.get_learning_rate()
        return torch.optim.AdamW(
            self.adapter_registry.parameters(),
            lr=lr,
            weight_decay=0.01,
        )

    def _update_optimizer_lr(self):
        """Update optimizer learning rate from curriculum."""
        lr = self.curriculum.get_learning_rate()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        logging.info(f"Updated learning rate to {lr}")

    def _setup_hooks(self):
        """Setup forward hooks for training."""
        if self.hook_manager is not None:
            logging.warning("Hooks already set up")
            return

        self.hook_manager = ForwardHookManager(
            model=self.base_model,
            adapter_registry=self.adapter_registry,
            lora_stack=self.lora_stack,
            device=self.device,
            dtype=self.dtype,
        )
        self.hook_manager.register_hooks()
        logging.info("Forward hooks registered for training")

    def _remove_hooks(self):
        """Remove forward hooks."""
        if self.hook_manager is not None:
            self.hook_manager.remove_hooks()
            self.hook_manager = None
            logging.info("Forward hooks removed")

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            dataloader: DataLoader providing (prompts, timesteps) pairs

        Returns:
            Dict of averaged loss components
        """
        epoch_losses = {
            "total": 0.0,
            "teacher": 0.0,
            "dominance": 0.0,
            "residual": 0.0,
            "consistency": 0.0,
        }

        num_batches = len(dataloader) if hasattr(dataloader, '__len__') else 1

        with ThreadSafeProgressBar(num_batches, desc="Training") as pbar:
            for batch_idx, batch_data in enumerate(dataloader):
                # Forward pass with adapters
                batch_loss = self._train_step(batch_data)

                # Accumulate losses
                for key in epoch_losses:
                    epoch_losses[key] += batch_loss[key].item()

                pbar.update(1)

        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= num_batches

        return epoch_losses

    def _compute_semantic_teacher(
            self,
            layer_deltas: Dict[str, torch.Tensor],
            layer_key: str,
            feature: str,
    ) -> torch.Tensor:
        """
        Compute semantic teacher target for a layer.
        """
        from .semantic_key_projection import get_semantic_importance

        weighted_sum = None
        total_weight = 0.0
        num_loras_with_maps = 0

        for lora_name, delta in layer_deltas.items():
            # Try to get semantic importance
            importance = 0.0

            if lora_name in self.adapter_registry.semantic_maps:
                lora_map = self.adapter_registry.semantic_maps[lora_name]

                enable_debug = not hasattr(self, '_projection_debug_logged')
                importance = get_semantic_importance(
                    semantic_map=lora_map,
                    layer_key=layer_key,
                    feature=feature,
                    default_importance=0.0,
                    debug=enable_debug
                )
                if enable_debug:
                    self._projection_debug_logged = True

                num_loras_with_maps += 1

            if weighted_sum is None:
                weighted_sum = importance * delta
            else:
                weighted_sum = weighted_sum + importance * delta

            total_weight += importance

        # === FIX: If no semantic maps matched, use uniform weights ===
        if total_weight == 0.0 and weighted_sum is not None:
            if not hasattr(self, '_uniform_weight_logged'):
                logger = logging.getLogger(__name__)
                logger.warning(
                    f"No semantic importance found for layer '{layer_key}', feature '{feature}'. "
                    f"Using uniform weights across {len(layer_deltas)} LoRAs."
                )
                self._uniform_weight_logged = True

            # Uniform average of all deltas
            return sum(layer_deltas.values()) / len(layer_deltas)

        if total_weight > 0 and weighted_sum is not None:
            return weighted_sum / total_weight
        elif weighted_sum is not None:
            return weighted_sum
        else:
            return torch.zeros_like(next(iter(layer_deltas.values())))

    def _train_step(self, batch_data) -> Dict[str, torch.Tensor]:
        """
        Single training step with synthetic delta generation.
        """
        self.optimizer.zero_grad()

        # Get current loss weights from curriculum
        loss_weights = self.curriculum.get_loss_weights()
        self.loss_fn.lambda_teacher = loss_weights["lambda_teacher"]
        self.loss_fn.lambda_dominance = loss_weights["lambda_dominance"]
        self.loss_fn.lambda_residual = loss_weights["lambda_residual"]
        self.loss_fn.lambda_consistency = loss_weights["lambda_consistency"]

        # Handle both real batch dict and dummy None batches
        if batch_data is None:
            batch_data = {
                "prompts": ["dummy prompt"],
                "features": ["hair"],
                "timesteps": torch.tensor([500], device=self.device),
                "latents": torch.randn(1, 4, 64, 96, device=self.device, dtype=self.dtype),
            }

        features = batch_data["features"]

        # Generate deltas using delta provider (Invariant 1)
        layer_deltas = self.delta_provider.get_deltas(batch_data)

        # Debug: Log adapter registry keys vs delta keys (first batch only)
        if not hasattr(self, '_debug_logged'):
            logger = logging.getLogger(__name__)
            registry_keys = list(self.adapter_registry.adapters.keys())[:5]
            delta_keys = list(layer_deltas.keys())[:5] if layer_deltas else []
            logger.info(f"Adapter registry sample keys: {registry_keys}")
            logger.info(f"Delta provider sample keys: {delta_keys}")
            self._debug_logged = True

        # Accumulate losses across all layers
        total_teacher_loss = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        total_dominance_loss = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        total_residual_loss = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        num_layers = 0
        num_nan_layers = 0

        # For each layer, compute teacher target and adapter output
        for layer_key, lora_deltas in layer_deltas.items():
            if len(lora_deltas) == 0:
                continue

            # === FIX 1: Move deltas to training device and dtype ===
            lora_deltas = {
                lora_name: delta.to(device=self.device, dtype=self.dtype)
                for lora_name, delta in lora_deltas.items()
            }

            # === FIX 2: Normalize deltas to prevent numerical overflow ===
            lora_deltas = self._normalize_deltas(lora_deltas)

            # Get adapter for this layer
            adapter = self.adapter_registry.adapters.get(layer_key)
            if adapter is None:
                continue

            # For each feature in the batch, compute teacher target
            feature = features[0] if len(features) > 0 else "hair"

            # Compute semantic teacher target
            teacher_target = self._compute_semantic_teacher(
                lora_deltas, layer_key, feature
            )

            # Create semantic vector for this feature
            num_features = len(self.adapter_registry.feature_names)

            # Get batch size from deltas
            first_delta = next(iter(lora_deltas.values()))
            batch_size = first_delta.shape[0]

            # Create batched semantic vector on training device
            semantic_vector = torch.zeros(
                batch_size, num_features,
                device=self.device, dtype=self.dtype
            )
            if feature in self.adapter_registry.feature_names:
                feature_idx = self.adapter_registry.feature_names.index(feature)
                semantic_vector[:, feature_idx] = 1.0

            # Call adapter forward pass
            try:
                adapter_output = adapter.forward(lora_deltas, semantic_vector)
            except Exception as e:
                logger = logging.getLogger(__name__)
                logger.warning(f"Adapter forward failed for {layer_key}: {e}, using fallback")
                adapter_output = sum(lora_deltas.values()) / len(lora_deltas)

            # === FIX 3: Check for NaN/Inf and skip problematic layers ===
            if self._has_invalid_values(adapter_output, teacher_target):
                num_nan_layers += 1
                if num_nan_layers <= 3:  # Log first 3 only
                    logger = logging.getLogger(__name__)
                    logger.warning(
                        f"NaN/Inf detected at {layer_key}, skipping. "
                        f"adapter: nan={torch.isnan(adapter_output).any()}, inf={torch.isinf(adapter_output).any()}, "
                        f"teacher: nan={torch.isnan(teacher_target).any()}, inf={torch.isinf(teacher_target).any()}"
                    )
                continue

            # Compute losses for this layer
            layer_teacher_loss = self.loss_fn.teacher_alignment_loss(
                adapter_output, teacher_target
            )
            layer_residual_loss = self.loss_fn.residual_economy_loss(
                adapter_output, teacher_target
            )

            # === FIX 4: Skip if loss itself is invalid ===
            if not torch.isfinite(layer_teacher_loss):
                num_nan_layers += 1
                continue

            total_teacher_loss = total_teacher_loss + layer_teacher_loss
            total_residual_loss = total_residual_loss + layer_residual_loss
            num_layers += 1

        # Log NaN statistics once
        if num_nan_layers > 0 and not hasattr(self, '_nan_logged'):
            logger = logging.getLogger(__name__)
            logger.warning(f"Skipped {num_nan_layers} layers due to NaN/Inf values")
            self._nan_logged = True

        # Debug: Log how many layers were actually trained
        if not hasattr(self, '_layers_logged'):
            logger = logging.getLogger(__name__)
            logger.info(f"Trained {num_layers} layers (out of {len(layer_deltas)} delta layers)")
            self._layers_logged = True

        # Average across layers
        if num_layers > 0:
            avg_teacher_loss = total_teacher_loss / num_layers
            avg_residual_loss = total_residual_loss / num_layers
        else:
            logger = logging.getLogger(__name__)
            logger.error("No valid layers for training! All layers had NaN/Inf.")
            avg_teacher_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            avg_residual_loss = torch.tensor(0.0, device=self.device)

        # Dominance and consistency losses not yet implemented
        dominance_loss = torch.tensor(0.0, device=self.device)
        consistency_loss = torch.tensor(0.0, device=self.device)

        # Compute total weighted loss
        total_loss = (
                self.loss_fn.lambda_teacher * avg_teacher_loss +
                self.loss_fn.lambda_dominance * dominance_loss +
                self.loss_fn.lambda_residual * avg_residual_loss +
                self.loss_fn.lambda_consistency * consistency_loss
        )

        losses = {
            "total": total_loss,
            "teacher": avg_teacher_loss,
            "dominance": dominance_loss,
            "residual": avg_residual_loss,
            "consistency": consistency_loss,
        }

        # Backpropagation (only if loss is valid)
        if total_loss.requires_grad and torch.isfinite(total_loss):
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.adapter_registry.parameters(),
                max_norm=1.0
            )
            self.optimizer.step()

        return losses

    def _normalize_deltas(
            self,
            lora_deltas: Dict[str, torch.Tensor],
            max_norm: float = 10.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Normalize deltas to prevent numerical overflow.

        Args:
            lora_deltas: Dict of LoRA name to delta tensor
            max_norm: Maximum L2 norm per sample

        Returns:
            Normalized deltas
        """
        normalized = {}
        for lora_name, delta in lora_deltas.items():
            # Compute per-sample norm (flatten spatial dims)
            flat = delta.view(delta.shape[0], -1)  # [batch, features]
            norms = flat.norm(dim=1, keepdim=True).clamp(min=1e-8)  # [batch, 1]

            # Scale down if norm exceeds max_norm
            scale = torch.clamp(max_norm / norms, max=1.0)

            # Reshape scale for broadcasting
            for _ in range(delta.ndim - 2):
                scale = scale.unsqueeze(-1)

            normalized[lora_name] = delta * scale

        return normalized

    def _has_invalid_values(self, *tensors) -> bool:
        """Check if any tensor contains NaN or Inf values."""
        for t in tensors:
            if torch.isnan(t).any() or torch.isinf(t).any():
                return True
        return False

    def train(
        self,
        dataloader: DataLoader = None,
        num_epochs: Optional[int] = None,
    ) -> Dict:
        """
        Full training loop with curriculum.

        Args:
            dataloader: Training data (optional for testing)
            num_epochs: Override total epochs (default: sum of curriculum stages)

        Returns:
            Training statistics
        """
        if num_epochs is None:
            num_epochs = self.curriculum.get_total_epochs()

        # If no dataloader provided, create dummy one for testing
        if dataloader is None:
            logging.warning("No dataloader provided, using dummy data for testing")
            dataloader = [None] * 10  # 10 dummy batches per epoch

        for epoch in range(num_epochs):
            progress = self.curriculum.get_progress()
            stage = self.curriculum.get_current_stage()

            logging.info(
                f"\nEpoch {epoch + 1}/{num_epochs} - "
                f"Stage: {stage.name} "
                f"({progress['epoch_in_stage']}/{progress['stage_epochs']})"
            )

            # Update dataloader epoch for metadata tracking
            if hasattr(dataloader, 'set_epoch'):
                dataloader.set_epoch(epoch)

            # Train one epoch
            epoch_losses = self.train_epoch(dataloader)

            # Log losses
            logging.info(
                f"Losses: "
                f"total={epoch_losses['total']:.6f}, "
                f"teacher={epoch_losses['teacher']:.6f}, "
                f"dominance={epoch_losses['dominance']:.6f}, "
                f"residual={epoch_losses['residual']:.6f}, "
                f"consistency={epoch_losses['consistency']:.6f}"
            )
            self.training_stats["losses"].append(epoch_losses)

            # Get adapter statistics
            adapter_stats = self.adapter_registry.get_statistics()
            self.training_stats["adapter_stats"].append(adapter_stats)

            # Step curriculum
            stage_changed = self.curriculum.step_epoch()
            if stage_changed:
                new_stage = self.curriculum.get_current_stage()
                logging.info(f"Advanced to stage: {new_stage.name}")
                self._update_optimizer_lr()

        logging.info(f"Training completed after {num_epochs} epochs")
        return self.training_stats

    def save_checkpoint(self, path: str):
        """
        Save training checkpoint.

        Args:
            path: Path to save checkpoint
        """
        checkpoint = {
            "adapter_state_dict": self.adapter_registry.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "curriculum_state": {
                "stage_idx": self.curriculum.current_stage_idx,
                "epoch": self.curriculum.current_epoch,
            },
            "training_stats": self.training_stats,
        }

        # Ensure parent directory exists
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        torch.save(checkpoint, path)
        logging.info(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: str):
        """
        Load training checkpoint.

        Args:
            path: Path to load checkpoint from
        """
        checkpoint = torch.load(path, map_location=self.device)

        self.adapter_registry.load_state_dict(checkpoint["adapter_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.curriculum.current_stage_idx = checkpoint["curriculum_state"]["stage_idx"]
        self.curriculum.current_epoch = checkpoint["curriculum_state"]["epoch"]
        self.training_stats = checkpoint["training_stats"]

        logging.info(
            f"Loaded checkpoint from {path} "
            f"(epoch {self.curriculum.current_epoch})"
        )

    def __repr__(self) -> str:
        progress = self.curriculum.get_progress()
        return (
            f"AdapterTrainer("
            f"adapters={len(self.adapter_registry)}, "
            f"stage={progress['stage_name']}, "
            f"epoch={progress['current_epoch']}/{progress['total_epochs']})"
        )
