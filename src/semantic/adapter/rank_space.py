"""
Rank-space adapter for attention and MLP layers.

Operates in the low-rank subspace between A (down-projection) and B (up-projection)
matrices, which is more efficient than output-space adaptation for layers with
natural low-rank structure.
"""

from typing import Dict, List
import torch
import torch.nn as nn

from .base import BaseAdapter


class RankSpaceAdapter(BaseAdapter):
    """
    Rank-space adapter for attention and MLP layers.

    Operates in the low-rank subspace between A and B matrices:
        z_lora_i = A_i @ x  (down-projection)
        g = gate_network(semantic_vector)
        z_mixed = Σ (g_i ⊙ z_lora_i)
        output = B_shared @ z_mixed

    This is more efficient than output-space adaptation for
    layers with natural low-rank structure (attention, MLP).
    """

    def __init__(
        self,
        layer_key: str,
        semantic_dim: int,
        feature_names: List[str],
        rank: int,
        num_loras: int,
        use_residual_mlp: bool = False,
    ):
        """
        Initialize rank-space adapter.

        Args:
            layer_key: Layer identifier
            semantic_dim: Dimensionality of semantic feature vector
            feature_names: List of feature names
            rank: LoRA rank dimension
            num_loras: Number of LoRAs to combine
            use_residual_mlp: Whether to use residual MLP for fine-tuning
        """
        super().__init__(layer_key, semantic_dim, feature_names)

        self.rank = rank
        self.num_loras = num_loras
        self.use_residual_mlp = use_residual_mlp

        # Gate network: semantic_vector -> [rank * num_loras]
        # Maps semantic importance to per-element gates in rank space
        self.gate_network = nn.Sequential(
            nn.Linear(semantic_dim, rank * num_loras // 2),
            nn.SiLU(),
            nn.Linear(rank * num_loras // 2, rank * num_loras),
            nn.Sigmoid(),
        )

        # Optional residual MLP for fine-tuning the mixed representation
        if use_residual_mlp:
            self.residual_mlp = nn.Sequential(
                nn.Linear(rank * num_loras, rank // 2),
                nn.SiLU(),
                nn.Linear(rank // 2, rank),
            )
        else:
            self.residual_mlp = None

        # Initialize weights for numerical stability
        self._init_weights()

    def _init_weights(self):
        """
        Initialize adapter weights for numerical stability.

        Uses Xavier uniform for linear layers with small gains.
        """
        # Initialize gate network
        for module in self.gate_network.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

        # Initialize residual MLP (if present)
        if self.residual_mlp is not None:
            for module in self.residual_mlp.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight, gain=0.01)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0.0)

    def forward(
        self,
        lora_down_outputs: Dict[str, torch.Tensor],
        b_shared: torch.Tensor,
        semantic_vector: torch.Tensor,
    ) -> torch.Tensor:
        """
        Combine multiple LoRA down-projection outputs using semantic guidance.

        Args:
            lora_down_outputs: Dict[lora_name -> z_i] where z_i = A_i @ x
                               Each z_i has shape [batch, seq_len, rank] or [batch, rank]
            b_shared: Shared up-projection matrix [out_dim, rank]
            semantic_vector: Feature importance [batch, semantic_dim]

        Returns:
            Combined output: B_shared @ z_mixed with shape [batch, seq_len, out_dim]
                             or [batch, out_dim]
        """
        # Get shapes from first down-projection output
        first_output = next(iter(lora_down_outputs.values()))
        batch_size = first_output.shape[0]

        # Handle both 2D [batch, rank] and 3D [batch, seq_len, rank] tensors
        if first_output.ndim == 2:
            # 2D case: [batch, rank]
            has_seq_dim = False
            seq_len = 1
        elif first_output.ndim == 3:
            # 3D case: [batch, seq_len, rank]
            has_seq_dim = True
            seq_len = first_output.shape[1]
        else:
            raise ValueError(
                f"Expected 2D or 3D tensors, got shape {first_output.shape}"
            )

        # Stack all down-projection outputs
        # Result: [batch, seq_len, num_loras * rank] or [batch, num_loras * rank]
        z_list = list(lora_down_outputs.values())
        if has_seq_dim:
            z_stacked = torch.cat(z_list, dim=-1)  # [batch, seq_len, num_loras * rank]
        else:
            z_stacked = torch.cat(z_list, dim=-1)  # [batch, num_loras * rank]

        # Compute gates from semantic vector
        # semantic_vector: [batch, semantic_dim]
        # gates: [batch, num_loras * rank]
        gates = self.gate_network(semantic_vector)

        # Broadcast gates over sequence dimension if needed
        if has_seq_dim:
            # Expand gates: [batch, 1, num_loras * rank]
            gates = gates.unsqueeze(1)

        # Apply gates element-wise
        z_gated = gates * z_stacked

        # Reshape and sum over LoRAs
        # [batch, seq_len, num_loras, rank] -> [batch, seq_len, rank]
        # or [batch, num_loras, rank] -> [batch, rank]
        if has_seq_dim:
            z_gated = z_gated.view(batch_size, seq_len, self.num_loras, self.rank)
            z_mixed = z_gated.sum(dim=2)  # [batch, seq_len, rank]
        else:
            z_gated = z_gated.view(batch_size, self.num_loras, self.rank)
            z_mixed = z_gated.sum(dim=1)  # [batch, rank]

        # Optional residual
        if self.residual_mlp is not None:
            z_residual = self.residual_mlp(z_stacked)
            z_mixed = z_mixed + z_residual

        # Apply shared up-projection
        # b_shared: [out_dim, rank]
        # z_mixed: [batch, seq_len, rank] or [batch, rank]
        # output: [batch, seq_len, out_dim] or [batch, out_dim]
        output = torch.matmul(z_mixed, b_shared.t())

        return output

    def get_statistics(self) -> Dict[str, float]:
        """
        Return adapter statistics for monitoring.

        Returns:
            Dict with:
            - gate_sparsity: Fraction of near-zero gate weights
            - gate_mean: Mean gate weight
            - gate_std: Standard deviation of gate weights
            - residual_norm: L2 norm of residual MLP (if present)
        """
        with torch.no_grad():
            # Get gate network weights
            gate_weights = self.gate_network[0].weight

            # Compute sparsity (fraction of weights < 0.01)
            gate_sparsity = (gate_weights.abs() < 0.01).float().mean().item()

            stats = {
                "gate_sparsity": gate_sparsity,
                "gate_mean": gate_weights.mean().item(),
                "gate_std": gate_weights.std().item(),
            }

            # Add residual MLP statistics if present
            if self.residual_mlp is not None:
                residual_weights = self.residual_mlp[0].weight
                residual_norm = torch.norm(residual_weights).item()
                stats["residual_norm"] = residual_norm

            return stats

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"layer_key={self.layer_key}, "
            f"rank={self.rank}, "
            f"num_loras={self.num_loras}, "
            f"use_residual_mlp={self.use_residual_mlp})"
        )
