"""
Output-space adapter for conv and non-decomposable layers.

Operates on full LoRA deltas rather than in rank space, providing a safe
fallback for layers without natural low-rank structure.
"""

from typing import Dict, List
import torch
import torch.nn as nn

from .base import BaseAdapter


class OutputSpaceAdapter(BaseAdapter):
    """
    Output-space adapter for conv and non-decomposable layers.

    Operates on final LoRA deltas:
        Δ_i = B_i @ A_i @ x  (full LoRA output)
        g = gate_network(semantic_vector)
        Δ_mixed = Σ (g_i ⊙ Δ_i) + residual([Δ_1, ..., Δ_n, semantic_vector])

    Fallback for layers without natural rank structure (conv layers, unknown layers).
    """

    def __init__(
        self,
        layer_key: str,
        semantic_dim: int,
        feature_names: List[str],
        output_dim: int,
        num_loras: int,
    ):
        """
        Initialize output-space adapter.

        Args:
            layer_key: Layer identifier
            semantic_dim: Dimensionality of semantic feature vector
            feature_names: List of feature names
            output_dim: Output dimension of the layer
            num_loras: Number of LoRAs to combine
        """
        super().__init__(layer_key, semantic_dim, feature_names)

        self.output_dim = output_dim
        self.num_loras = num_loras

        # Gate network: semantic_vector -> [num_loras]
        # Produces scalar or channel-wise weights per LoRA
        self.gate_network = nn.Sequential(
            nn.Linear(semantic_dim, num_loras * 2),
            nn.SiLU(),
            nn.Linear(num_loras * 2, num_loras),
            nn.Sigmoid(),
        )

        # Residual MLP: concatenated normalized deltas + semantic -> output_dim
        # This allows learning non-linear combinations beyond weighted sum
        concat_dim = output_dim * num_loras + semantic_dim
        self.residual_mlp = nn.Sequential(
            nn.LayerNorm(concat_dim, eps=1e-5),  # Added eps for numerical stability
            nn.Linear(concat_dim, output_dim // 4),
            nn.SiLU(),
            nn.Linear(output_dim // 4, output_dim),
        )

        # Initialize weights for numerical stability
        self._init_weights()

    def _init_weights(self):
        """
        Initialize adapter weights for numerical stability.

        Uses Xavier uniform for linear layers and small constant bias.
        Gate network is initialized to produce near-uniform weights initially.

        IMPORTANT: Weights are initialized in float32 for numerical stability,
        even if the adapter will later be converted to another dtype.
        """
        # Ensure initialization happens in float32 for stability
        original_dtype = next(self.gate_network.parameters()).dtype
        if original_dtype not in [torch.float32, torch.float16, torch.bfloat16]:
            # Convert to float32 for initialization
            self.gate_network = self.gate_network.to(dtype=torch.float32)
            self.residual_mlp = self.residual_mlp.to(dtype=torch.float32)

        # Initialize gate network
        for module in self.gate_network.modules():
            if isinstance(module, nn.Linear):
                # Xavier uniform initialization
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    # Small positive bias so sigmoid starts around 0.5
                    nn.init.constant_(module.bias, 0.0)

        # Initialize residual MLP
        for module in self.residual_mlp.modules():
            if isinstance(module, nn.Linear):
                # Small initialization for residual path
                # This ensures residual starts near zero (identity-like behavior)
                nn.init.xavier_uniform_(module.weight, gain=0.01)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

        # Convert back to original dtype if needed
        if original_dtype not in [torch.float32, torch.float16, torch.bfloat16]:
            self.gate_network = self.gate_network.to(dtype=original_dtype)
            self.residual_mlp = self.residual_mlp.to(dtype=original_dtype)

    def forward(
        self,
        lora_deltas: Dict[str, torch.Tensor],
        semantic_vector: torch.Tensor,
    ) -> torch.Tensor:
        """
        Combine multiple LoRA deltas using semantic guidance.

        Args:
            lora_deltas: Dict[lora_name -> Δ_i] where Δ_i = full LoRA output
                         Shape: [batch, seq_len, channels] for attention (channels-last)
                                [batch, channels, *spatial_dims] for conv (channels-first)
            semantic_vector: Feature importance [batch, semantic_dim]

        Returns:
            Combined delta with same shape as input deltas
        """
        # Input validation: check for NaN/Inf in inputs
        for lora_name, delta in lora_deltas.items():
            if torch.isnan(delta).any():
                raise ValueError(
                    f"NaN detected in input delta for {lora_name} in layer {self.layer_key}"
                )
            if torch.isinf(delta).any():
                raise ValueError(
                    f"Inf detected in input delta for {lora_name} in layer {self.layer_key}"
                )

        if torch.isnan(semantic_vector).any():
            raise ValueError(f"NaN detected in semantic_vector for layer {self.layer_key}")
        if torch.isinf(semantic_vector).any():
            raise ValueError(f"Inf detected in semantic_vector for layer {self.layer_key}")

        # Ensure computation happens in a compatible dtype (float32/float16/bfloat16)
        # Float8 dtypes don't support all operations, so we upcast to float16
        compute_dtype = torch.float32
        for delta in lora_deltas.values():
            if delta.dtype in [torch.float32, torch.float16, torch.bfloat16]:
                compute_dtype = delta.dtype
                break

        # If using float8, upcast to float16 for computation
        if compute_dtype not in [torch.float32, torch.float16, torch.bfloat16]:
            compute_dtype = torch.float16

        # Convert inputs to compute dtype
        lora_deltas_compute = {
            name: delta.to(dtype=compute_dtype)
            for name, delta in lora_deltas.items()
        }
        semantic_vector_compute = semantic_vector.to(dtype=compute_dtype)

        # Stack deltas along new dimension
        delta_list = list(lora_deltas_compute.values())
        deltas_stacked = torch.stack(delta_list, dim=1)  # Stack at dim 1
        # Result: [batch, num_loras, seq_len, channels] for attention
        #         [batch, num_loras, channels, H, W] for conv2d

        # Detect tensor format based on rank and channel position
        # Attention layers: [batch, num_loras, seq_len, channels] - 4D, channels last
        # Conv2D layers: [batch, num_loras, channels, H, W] - 5D, channels early
        # Linear layers: [batch, num_loras, channels] - 3D
        batch_size = deltas_stacked.shape[0]
        num_loras = deltas_stacked.shape[1]

        if deltas_stacked.ndim == 3:
            # Linear layer (2D input): [batch, num_loras, channels]
            channels = deltas_stacked.shape[2]
            spatial_dims = ()
            deltas_permuted = deltas_stacked  # Already channels-last
        elif deltas_stacked.ndim == 4:
            # Attention layer (3D input): [batch, num_loras, seq_len, channels]
            # Treat seq_len as a 1D spatial dimension (channels-last format)
            seq_len = deltas_stacked.shape[2]
            channels = deltas_stacked.shape[3]
            spatial_dims = (seq_len,)
            deltas_permuted = deltas_stacked  # Already [batch, num_loras, seq_len, channels]
        elif deltas_stacked.ndim == 5:
            # Conv2D layer (4D input): [batch, num_loras, channels, H, W]
            # Convert from channels-first to channels-last
            channels = deltas_stacked.shape[2]
            spatial_dims = deltas_stacked.shape[3:]
            deltas_permuted = deltas_stacked.permute(0, 1, 3, 4, 2)  # [batch, num_loras, H, W, channels]
        else:
            raise NotImplementedError(f"Unsupported tensor rank: {deltas_stacked.ndim}")

        # Rearrange to [batch, *spatial_dims, num_loras, channels] for computation
        if len(spatial_dims) == 0:
            # Linear layer: [batch, num_loras, channels] -> [batch, num_loras, channels]
            # Already in correct format (no spatial dims)
            deltas_final = deltas_permuted
        elif len(spatial_dims) == 1:
            # Attention or 1D: [batch, num_loras, seq_len, channels] -> [batch, seq_len, num_loras, channels]
            deltas_final = deltas_permuted.permute(0, 2, 1, 3)
        elif len(spatial_dims) == 2:
            # 2D conv: [batch, num_loras, H, W, channels] -> [batch, H, W, num_loras, channels]
            deltas_final = deltas_permuted.permute(0, 2, 3, 1, 4)
        else:
            raise NotImplementedError(f"Spatial dims > 2 not yet supported: {spatial_dims}")

        # Compute gates: [batch, num_loras]
        # Ensure gate network operates in compute_dtype for numerical stability
        gate_network_compute = self.gate_network
        if next(self.gate_network.parameters()).dtype != compute_dtype:
            # Temporarily convert network to compute dtype
            # This handles cases where adapter was initialized in float8 but needs float16/32 for computation
            gate_network_compute = self.gate_network.to(dtype=compute_dtype)

        gates = gate_network_compute(semantic_vector_compute)

        # Check for NaN in gates (can happen if network weights exploded)
        if torch.isnan(gates).any():
            raise RuntimeError(
                f"NaN detected in gate network output for layer {self.layer_key}. "
                f"This may indicate numerical instability. "
                f"Gate stats: min={gates.min().item():.4f}, max={gates.max().item():.4f}"
            )

        # Reshape gates for broadcasting
        # Target shape: [batch, 1, ..., 1, num_loras, 1]
        gate_shape = [batch_size] + [1] * len(spatial_dims) + [num_loras, 1]
        gates_reshaped = gates.view(*gate_shape)

        # Weighted sum: [batch, *spatial_dims, channels]
        delta_mixed = (gates_reshaped * deltas_final).sum(dim=-2)

        # VRAM optimization: Skip residual MLP for large sequences (attention layers)
        # Residual MLP processes every position, which is huge for seq_len=6144
        # For now, only use gated mixing (still learnable via gate_network)
        total_spatial_size = 1
        for dim_size in spatial_dims:
            total_spatial_size *= dim_size

        if total_spatial_size > 1000:
            # Large sequence (attention layer): skip residual to save VRAM
            # Just use gated weighted sum
            output_channels_last = delta_mixed
        else:
            # Small or no sequence (linear/small conv): apply full residual MLP
            # Prepare input for residual MLP
            # Flatten: [batch, *spatial_dims, num_loras * channels]
            flat_shape = [batch_size] + list(spatial_dims) + [num_loras * channels]
            deltas_flat = deltas_final.reshape(*flat_shape)

            # Expand semantic vector to match spatial dimensions
            # semantic_vector_compute: [batch, semantic_dim]
            semantic_expanded = semantic_vector_compute.view(batch_size, *([1] * len(spatial_dims)), self.semantic_dim)

            # Broadcast to match spatial dimensions
            if len(spatial_dims) > 0:
                semantic_expanded = semantic_expanded.expand(batch_size, *spatial_dims, self.semantic_dim)

            # Concatenate: [batch, *spatial_dims, num_loras * channels + semantic_dim]
            concat_input = torch.cat([deltas_flat, semantic_expanded], dim=-1)

            # Apply residual MLP
            residual = self.residual_mlp(concat_input)

            # Add residual to weighted sum
            output_channels_last = delta_mixed + residual

        # Return in same format as input
        if len(spatial_dims) == 0:
            # Linear: [batch, channels] - already correct
            output = output_channels_last
        elif len(spatial_dims) == 1:
            # Attention: [batch, seq_len, channels] - already correct (channels-last)
            output = output_channels_last
        elif len(spatial_dims) == 2:
            # Conv2D: [batch, H, W, channels] -> [batch, channels, H, W] (channels-first)
            output = output_channels_last.permute(0, 3, 1, 2)

        # Final NaN check with detailed diagnostics
        if torch.isnan(output).any() or torch.isinf(output).any():
            import logging
            logger = logging.getLogger(__name__)
            logger.error(
                f"NaN/Inf in adapter output for layer {self.layer_key}:\n"
                f"  Output: nan={torch.isnan(output).any()}, inf={torch.isinf(output).any()}\n"
                f"  Output stats: min={output.min().item():.4f}, max={output.max().item():.4f}, mean={output.mean().item():.4f}\n"
                f"  Gates: min={gates.min().item():.4f}, max={gates.max().item():.4f}, mean={gates.mean().item():.4f}\n"
                f"  Delta_mixed: nan={torch.isnan(delta_mixed).any()}, inf={torch.isinf(delta_mixed).any()}\n"
                f"  Input deltas shapes: {[d.shape for d in lora_deltas.values()]}\n"
                f"  Spatial dims: {spatial_dims}, total size: {total_spatial_size}"
            )
            # Don't raise exception here - let trainer handle it
            # This provides better debugging info

        # Convert back to original dtype if we upcast
        original_dtype = next(iter(lora_deltas.values())).dtype
        if output.dtype != original_dtype:
            output = output.to(dtype=original_dtype)

        return output

    def get_statistics(self) -> Dict[str, float]:
        """
        Return adapter statistics for monitoring.

        Returns:
            Dict with:
            - gate_entropy: Entropy of gate distribution (higher = more uniform)
            - residual_magnitude: L2 norm of residual MLP weights
        """
        with torch.no_grad():
            # Get gate network weights
            gate_weights = self.gate_network[0].weight

            # Compute entropy of softmax distribution
            # Higher entropy = more uniform gating
            gate_probs = torch.softmax(gate_weights, dim=-1)
            gate_log_probs = torch.log_softmax(gate_weights, dim=-1)
            gate_entropy = -(gate_probs * gate_log_probs).sum().item()

            # Get residual MLP weight magnitude
            residual_weights = self.residual_mlp[-1].weight
            residual_magnitude = torch.norm(residual_weights).item()

            return {
                "gate_entropy": gate_entropy,
                "residual_magnitude": residual_magnitude,
            }

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"layer_key={self.layer_key}, "
            f"output_dim={self.output_dim}, "
            f"num_loras={self.num_loras})"
        )
