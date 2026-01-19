"""
Loss functions for training semantic adapters.

Provides composite loss with multiple components:
1. Teacher alignment: Match static semantic merge
2. Feature dominance: Ensure correct features dominate
3. Residual economy: Penalize unnecessary complexity
4. Output consistency: Maintain global image quality
"""

from typing import Dict
import torch
import torch.nn.functional as F


class SemanticAdapterLoss:
    """
    Composite loss for training semantic adapters.

    Components:
    1. Teacher alignment: Match static semantic merge output
    2. Feature dominance: Penalize deviation from requested source per feature
    3. Residual economy: Penalize adapter outputs larger than teacher
    4. Output consistency: Maintain final denoised output quality
    """

    def __init__(
        self,
        lambda_teacher: float = 1.0,
        lambda_dominance: float = 0.5,
        lambda_residual: float = 0.1,
        lambda_consistency: float = 0.3,
    ):
        """
        Initialize composite loss.

        Args:
            lambda_teacher: Weight for teacher alignment loss
            lambda_dominance: Weight for feature dominance loss
            lambda_residual: Weight for residual economy loss
            lambda_consistency: Weight for output consistency loss
        """
        self.lambda_teacher = lambda_teacher
        self.lambda_dominance = lambda_dominance
        self.lambda_residual = lambda_residual
        self.lambda_consistency = lambda_consistency

    def teacher_alignment_loss(
        self,
        adapter_delta: torch.Tensor,
        semantic_merge_delta: torch.Tensor,
    ) -> torch.Tensor:
        """
        Teacher alignment loss: ||Δ̂_adapter - Δ_semantic_merge||²

        Ensures adapter learns at least as well as static semantic merge.

        Args:
            adapter_delta: Output from adapter
            semantic_merge_delta: Output from static semantic merge (teacher)

        Returns:
            MSE loss (scalar)
        """
        # Ensure both tensors are on the same device (defensive check)
        if adapter_delta.device != semantic_merge_delta.device:
            semantic_merge_delta = semantic_merge_delta.to(adapter_delta.device)
        return F.mse_loss(adapter_delta, semantic_merge_delta)

    def feature_dominance_loss(
        self,
        adapter_delta: torch.Tensor,
        source_deltas: Dict[str, torch.Tensor],
        feature_requests: Dict[str, str],
        feature_weights: Dict[str, float],
    ) -> torch.Tensor:
        """
        Feature dominance loss: Σ_f w_f · ||Δ̂ - Δ_source(f)||²

        Penalizes deviation from requested source for each feature.

        Args:
            adapter_delta: Output from adapter
            source_deltas: Dict mapping lora_name -> delta tensor
            feature_requests: Dict mapping feature -> source_lora
            feature_weights: Dict mapping feature -> weight

        Returns:
            Weighted MSE loss (scalar)
        """
        if not feature_requests:
            return torch.tensor(0.0, device=adapter_delta.device)

        loss = torch.tensor(0.0, device=adapter_delta.device)
        for feature, source_lora in feature_requests.items():
            if source_lora in source_deltas:
                weight = feature_weights.get(feature, 1.0)
                target_delta = source_deltas[source_lora]
                # Ensure device consistency
                if target_delta.device != adapter_delta.device:
                    target_delta = target_delta.to(adapter_delta.device)
                loss += weight * F.mse_loss(adapter_delta, target_delta)

        return loss / len(feature_requests)

    def residual_economy_loss(
        self,
        adapter_delta: torch.Tensor,
        semantic_merge_delta: torch.Tensor,
    ) -> torch.Tensor:
        """
        Residual economy loss: ReLU(||Δ̂||² / ||Δ_semantic||² - 1)²

        Penalizes adapter outputs larger than the teacher merge.
        Only applies penalty if adapter magnitude exceeds teacher.

        Args:
            adapter_delta: Output from adapter
            semantic_merge_delta: Output from teacher

        Returns:
            Loss (scalar)
        """
        # Ensure both tensors are on the same device (defensive check)
        if adapter_delta.device != semantic_merge_delta.device:
            semantic_merge_delta = semantic_merge_delta.to(adapter_delta.device)

        adapter_norm = torch.norm(adapter_delta)
        teacher_norm = torch.norm(semantic_merge_delta)

        # Check for invalid values
        if not torch.isfinite(adapter_norm) or not torch.isfinite(teacher_norm):
            return torch.tensor(0.0, device=adapter_delta.device)

        # Avoid division by zero - return zero loss if teacher is too small
        if teacher_norm < 1e-6:
            return torch.tensor(0.0, device=adapter_delta.device)

        ratio = adapter_norm / teacher_norm

        # Penalize only if adapter is larger than teacher
        excess = F.relu(ratio - 1.0)
        return excess ** 2

    def output_consistency_loss(
        self,
        predicted_noise: torch.Tensor,
        target_noise: torch.Tensor,
    ) -> torch.Tensor:
        """
        Output consistency loss: ||ε̂ - ε_target||²

        Ensures final denoised output matches expected quality.
        This operates at the diffusion output level (noise prediction).

        Args:
            predicted_noise: Predicted noise from model with adapters
            target_noise: Target noise (ground truth or reference)

        Returns:
            MSE loss (scalar)
        """
        # Ensure device consistency
        if predicted_noise.device != target_noise.device:
            target_noise = target_noise.to(predicted_noise.device)
        return F.mse_loss(predicted_noise, target_noise)

    def __call__(
        self,
        adapter_delta: torch.Tensor,
        semantic_merge_delta: torch.Tensor,
        source_deltas: Dict[str, torch.Tensor],
        feature_requests: Dict[str, str],
        feature_weights: Dict[str, float],
        predicted_noise: torch.Tensor,
        target_noise: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all loss components.

        Args:
            adapter_delta: Output from adapter
            semantic_merge_delta: Output from teacher merge
            source_deltas: Dict mapping lora_name -> delta tensor
            feature_requests: Dict mapping feature -> source_lora
            feature_weights: Dict mapping feature -> weight
            predicted_noise: Predicted noise from model
            target_noise: Target noise

        Returns:
            Dict with keys: total, teacher, dominance, residual, consistency
        """
        l_teacher = self.teacher_alignment_loss(adapter_delta, semantic_merge_delta)
        l_dominance = self.feature_dominance_loss(
            adapter_delta, source_deltas, feature_requests, feature_weights
        )
        l_residual = self.residual_economy_loss(adapter_delta, semantic_merge_delta)
        l_consistency = self.output_consistency_loss(predicted_noise, target_noise)

        total = (
            self.lambda_teacher * l_teacher +
            self.lambda_dominance * l_dominance +
            self.lambda_residual * l_residual +
            self.lambda_consistency * l_consistency
        )

        return {
            "total": total,
            "teacher": l_teacher,
            "dominance": l_dominance,
            "residual": l_residual,
            "consistency": l_consistency,
        }

    def __repr__(self) -> str:
        return (
            f"SemanticAdapterLoss("
            f"λ_teacher={self.lambda_teacher}, "
            f"λ_dominance={self.lambda_dominance}, "
            f"λ_residual={self.lambda_residual}, "
            f"λ_consistency={self.lambda_consistency})"
        )
