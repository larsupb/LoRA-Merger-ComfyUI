"""
Centralized type definitions for LoRA Power-Merger.

This module consolidates all type aliases and type definitions used throughout the codebase.
All modules should import types from this module for consistency.
"""

from typing import Dict, Tuple, List, Set, Any, Optional, Protocol, TypedDict, Union
import torch
from comfy.weight_adapter import LoRAAdapter


# ============================================================================
# Core LoRA Tensor Types
# ============================================================================

# A LoRA is represented as a tuple of (up_tensor, down_tensor, alpha_value)
# This is the fundamental decomposed representation used in merging operations
LORA_TENSORS = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]

# Dictionary mapping LoRA names to their tensor tuples for a single layer
LORA_TENSOR_DICT = Dict[str, LORA_TENSORS]

# Dictionary mapping layer keys to LORA_TENSOR_DICT
# This represents all layers for all LoRAs being merged
LORA_TENSORS_BY_LAYER = Dict[str, LORA_TENSOR_DICT]


# ============================================================================
# ComfyUI LoRA Types
# ============================================================================

# ComfyUI's internal LoRA representation: layer key -> LoRAAdapter
# The LoRAAdapter contains weights tuple: (up, down, alpha, mid, dora_scale, reshape)
LORA_KEY_DICT = Dict[str, LoRAAdapter]

# Stack of LoRAs: LoRA name -> layer dictionary
# This is the primary data structure for managing multiple LoRAs
LORA_STACK = Dict[str, LORA_KEY_DICT]

# Raw LoRA state dicts: Maps LoRA name to original raw state dict
# Used to preserve CLIP weights when saving merged LoRAs
LORA_RAW_DICT = Dict[str, Dict[str, torch.Tensor]]

# Weight/strength metadata for each LoRA
# Maps LoRA name to dict with 'strength_model' and 'strength_clip' keys
LORA_WEIGHTS = Dict[str, Dict[str, float]]


# ============================================================================
# Merge Method Types
# ============================================================================

class MergeMethodDict(TypedDict, total=False):
    """Type definition for merge method configuration dictionaries."""
    name: str  # Required
    settings: Dict[str, Any]  # Optional method-specific settings


class MergeMethod(Protocol):
    """Protocol for merge method implementations."""

    def __call__(
        self,
        tensors: Dict[str, torch.Tensor],
        gather_tensors: Dict[str, torch.Tensor],
        weight_info: Dict[str, float],
        tensor_parameters: Dict[str, Dict[str, Any]],
        method_args: Dict[str, Any]
    ) -> torch.Tensor:
        """Execute the merge operation."""
        ...


class MergeContextDict(TypedDict, total=False):
    """
    Merge context containing all parameters needed for LoRA merging.
    Used to pass merge configuration between nodes without duplicating inputs.
    """
    method: MergeMethodDict  # Merge method configuration
    components: LORA_TENSORS_BY_LAYER  # Decomposed LoRA components
    strengths: LORA_WEIGHTS  # Per-LoRA strength values
    lambda_: float  # Lambda scaling factor
    device: str  # Device for computation
    dtype: str  # Data type for computation


# Type alias for merge context
MergeContext = MergeContextDict


# ============================================================================
# Node Input/Output Types
# ============================================================================

class LoRABundleDict(TypedDict, total=False):
    """
    Single LoRA bundle with metadata.
    Used for wrapping individual LoRAs with strength information.

    Fields:
        lora_raw: Raw state dict (complete LoRA file) - used to preserve original CLIP weights
        lora: ComfyUI format (model weights only, after layer filtering)
        strength_model: Strength multiplier for model weights
        strength_clip: Strength multiplier for CLIP weights (optional)
        name: Pretty name of the LoRA
    """
    lora_raw: Dict[str, torch.Tensor]  # Raw state dict (includes both model and CLIP weights)
    lora: LORA_KEY_DICT  # ComfyUI format (model weights, may be filtered)
    strength_model: float
    strength_clip: float
    name: str


# Layer filter options for selective merging
LayerFilterType = str  # "full" | "attn-mlp" | "attn-only"

# Device specification
DeviceType = Union[torch.device, str]  # e.g., "cpu", "cuda", torch.device("cuda:0")

# Dtype specification
DtypeType = Union[torch.dtype, str]  # e.g., "float32", torch.float32


# ============================================================================
# Architecture-Specific Types
# ============================================================================

class BlockNameInfo(TypedDict, total=False):
    """
    Information extracted from layer key parsing.
    Used for Stable Diffusion architecture.
    """
    block_type: str  # "input_blocks" | "middle_block" | "output_blocks"
    block_idx: str
    inner_idx: Optional[str]
    component: str  # "attn1" | "attn2" | "ff" | "proj_in" | "proj_out"
    main_block: str
    sub_block: str
    transformer_idx: Optional[str]


class DiTBlockNameInfo(TypedDict, total=False):
    """
    Information extracted from layer key parsing.
    Used for DiT (Diffusion Transformer) architecture.
    """
    layer_idx: str
    component: str  # "attention" | "norm1" | "norm2" | "mlp" | etc.
    main_block: str  # e.g., "layers_group.2"
    sub_block: str  # e.g., "layers.13"
    group_idx: int
    group_start: int
    group_end: int


# Union of all block info types
BlockNameInfoUnion = Union[BlockNameInfo, DiTBlockNameInfo, None]


# ============================================================================
# Tensor Parameter Types
# ============================================================================

class TensorParameterDict(TypedDict, total=False):
    """Parameters for individual tensors in merge operations."""
    weight: float  # Strength multiplier
    density: Optional[float]  # For DARE/sparse methods
    epsilon: Optional[float]  # For numerical stability


class WeightInfoDict(TypedDict):
    """Weight information for merge operations."""
    strength_model: float
    strength_clip: float


# ============================================================================
# SVD/Decomposition Types
# ============================================================================

class SVDConfig(TypedDict, total=False):
    """Configuration for SVD operations."""
    rank: Optional[int]  # Target rank (None = auto)
    sv_ratio: Optional[float]  # Singular value ratio threshold
    sv_cumulative: Optional[float]  # Cumulative energy threshold
    sv_fro: Optional[float]  # Frobenius norm retention threshold
    distribution: str  # "symmetric" | "asymmetric"


class SVDStats(TypedDict):
    """Statistics from SVD operation."""
    original_rank: int
    new_rank: int
    sv_retention: float  # Percentage of singular values retained
    fro_retention: float  # Frobenius norm retention
    energy_retention: float  # Energy (cumulative SV) retention


class DecompositionResult(TypedDict):
    """Result from tensor decomposition operation."""
    up: torch.Tensor
    down: torch.Tensor
    alpha: torch.Tensor
    stats: Optional[SVDStats]


# ============================================================================
# Progress and Status Types
# ============================================================================

class ProgressInfo(TypedDict):
    """Progress information for long-running operations."""
    current: int
    total: int
    description: str


# ============================================================================
# Validation Types
# ============================================================================

class ValidationError(TypedDict):
    """Validation error information."""
    code: str
    message: str
    location: Optional[str]


class ValidationResult(TypedDict):
    """Result of validation operation."""
    valid: bool
    errors: List[ValidationError]
    warnings: List[str]


# ============================================================================
# Layer Filtering Types
# ============================================================================

# Set of layer component names to include in filtering
LayerComponentSet = Set[str]


# ============================================================================
# Constants for Architecture Detection
# ============================================================================

# Standard LoRA tensor indices in ComfyUI's (up, down, alpha, mid, dora_scale, reshape) tuple
SD_LORA_UP_IDX = 0
SD_LORA_DOWN_IDX = 1
SD_LORA_ALPHA_IDX = 2

# Minimum singular value threshold for SVD operations
MIN_SINGULAR_VALUE = 1e-6


# ============================================================================
# Type Guards and Validators
# ============================================================================

def is_lora_tensors(obj: Any) -> bool:
    """Type guard for LORA_TENSORS."""
    return (
        isinstance(obj, tuple) and
        len(obj) == 3 and
        all(isinstance(t, torch.Tensor) for t in obj[:2]) and
        (isinstance(obj[2], torch.Tensor) or isinstance(obj[2], (int, float)))
    )


def is_lora_stack(obj: Any) -> bool:
    """Type guard for LORA_STACK."""
    return (
        isinstance(obj, dict) and
        all(isinstance(k, str) and isinstance(v, dict) for k, v in obj.items())
    )


def validate_lora_tensors(tensors: LORA_TENSORS) -> None:
    """
    Validate LORA_TENSORS structure.

    Raises:
        ValueError: If tensors are invalid
    """
    if not is_lora_tensors(tensors):
        raise ValueError(f"Invalid LORA_TENSORS structure: {type(tensors)}")

    up, down, alpha = tensors

    if up.ndim not in (2, 4):
        raise ValueError(f"Invalid up tensor dimensions: {up.shape}")

    if down.ndim not in (2, 4):
        raise ValueError(f"Invalid down tensor dimensions: {down.shape}")


def validate_lora_stack(stack: LORA_STACK) -> None:
    """
    Validate LORA_STACK structure.

    Raises:
        ValueError: If stack is invalid
    """
    if not is_lora_stack(stack):
        raise ValueError(f"Invalid LORA_STACK structure: {type(stack)}")

    if len(stack) == 0:
        raise ValueError("LORA_STACK cannot be empty")


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # Core types
    'LORA_TENSORS',
    'LORA_TENSOR_DICT',
    'LORA_TENSORS_BY_LAYER',
    'LORA_KEY_DICT',
    'LORA_STACK',
    'LORA_RAW_DICT',
    'LORA_WEIGHTS',

    # Merge types
    'MergeMethodDict',
    'MergeMethod',
    'TensorParameterDict',
    'WeightInfoDict',

    # Node types
    'LoRABundleDict',
    'LayerFilterType',
    'DeviceType',
    'DtypeType',

    # Architecture types
    'BlockNameInfo',
    'DiTBlockNameInfo',
    'BlockNameInfoUnion',

    # SVD types
    'SVDConfig',
    'SVDStats',
    'DecompositionResult',

    # Validation types
    'ValidationError',
    'ValidationResult',

    # Other types
    'LayerComponentSet',
    'ProgressInfo',

    # Constants
    'SD_LORA_UP_IDX',
    'SD_LORA_DOWN_IDX',
    'SD_LORA_ALPHA_IDX',
    'MIN_SINGULAR_VALUE',

    # Validators
    'is_lora_tensors',
    'is_lora_stack',
    'validate_lora_tensors',
    'validate_lora_stack',
]
