"""
Configuration constants for LoRA Power-Merger.

Centralizes all magic numbers and configuration values used throughout the codebase.
"""

# ============================================================================
# SVD/Decomposition Constants
# ============================================================================

# Minimum singular value threshold for SVD operations
MIN_SINGULAR_VALUE = 1e-6

# Default SVD parameters
DEFAULT_SVD_RANK = 16
DEFAULT_SVD_DISTRIBUTION = "symmetric"  # or "asymmetric"

# Dynamic rank selection defaults
DEFAULT_SV_RATIO = 100.0
DEFAULT_SV_CUMULATIVE = 0.95
DEFAULT_SV_FRO = 0.99


# ============================================================================
# Merge Operation Constants
# ============================================================================

# Maximum number of worker threads for parallel processing
MAX_MERGE_WORKERS = 8

# Default lambda scaling factor
DEFAULT_LAMBDA = 1.0

# Default normalization setting
DEFAULT_NORMALIZE = True


# ============================================================================
# Validation Constants
# ============================================================================

# Minimum number of LoRAs required for merge
MIN_LORAS_FOR_MERGE = 2

# Minimum key overlap ratio to avoid warnings
MIN_KEY_OVERLAP_RATIO = 0.5

# Typical strength value range (for warnings)
TYPICAL_STRENGTH_MIN = 0.0
TYPICAL_STRENGTH_MAX = 1.0


# ============================================================================
# Device and Memory Constants
# ============================================================================

# Supported device types
SUPPORTED_DEVICES = ["cpu", "cuda", "mps", "auto"]

# Supported dtype strings
SUPPORTED_DTYPES = [
    "float16", "float32", "float64",
    "bfloat16",
    "int8", "int16", "int32", "int64",
]

# Default device for computation
DEFAULT_DEVICE = "cpu"

# Default dtype for computation
DEFAULT_DTYPE = "float32"


# ============================================================================
# Progress Bar Constants
# ============================================================================

# Update frequency for progress bars (in seconds)
PROGRESS_UPDATE_INTERVAL = 0.1

# Default progress bar format
PROGRESS_BAR_FORMAT = "{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}"


# ============================================================================
# Layer Filter Constants
# ============================================================================

# Architecture-agnostic layer filter sets (works for SD, DiT, Flux, and Wan)
# Note: 'attn' is added as a general pattern to catch Flux keys like 'img_attn_proj'
ATTENTION_LAYERS = {"attn", "attn1", "attn2", "attention", "self_attn", "cross_attn"}
MLP_LAYERS = {"ff", "mlp", "feed_forward", "ffn"}
ATTENTION_MLP_LAYERS = {"attn", "attn1", "attn2", "attention", "self_attn", "cross_attn", "ff", "mlp", "feed_forward", "ffn"}

# Legacy architecture-specific constants (deprecated, kept for backward compatibility)
SD_ATTENTION_LAYERS = {"attn1", "attn2"}
SD_MLP_LAYERS = {"ff"}
SD_ATTENTION_MLP_LAYERS = {"attn1", "attn2", "ff"}
SD_PROJECTION_LAYERS = {"proj_in", "proj_out"}
DIT_ATTENTION_LAYERS = {"attention"}
DIT_MLP_LAYERS = {"mlp", "feed_forward"}
WAN_ATTENTION_LAYERS = {"self_attn", "cross_attn"}
WAN_MLP_LAYERS = {"ffn"}


# ============================================================================
# File I/O Constants
# ============================================================================

# Supported LoRA file extensions
LORA_FILE_EXTENSIONS = [".safetensors", ".pt", ".pth", ".ckpt"]

# Default LoRA save format
DEFAULT_LORA_SAVE_FORMAT = "safetensors"


# ============================================================================
# Caching Constants
# ============================================================================

# Maximum cache size for decomposition results (number of entries)
MAX_DECOMPOSITION_CACHE_SIZE = 100

# Cache TTL in seconds (time-to-live)
CACHE_TTL_SECONDS = 3600  # 1 hour


# ============================================================================
# Logging Constants
# ============================================================================

# Default logging level
DEFAULT_LOG_LEVEL = "INFO"

# Log format
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


# ============================================================================
# Export All Constants
# ============================================================================

__all__ = [
    # SVD/Decomposition
    'MIN_SINGULAR_VALUE',
    'DEFAULT_SVD_RANK',
    'DEFAULT_SVD_DISTRIBUTION',
    'DEFAULT_SV_RATIO',
    'DEFAULT_SV_CUMULATIVE',
    'DEFAULT_SV_FRO',

    # Merge Operations
    'MAX_MERGE_WORKERS',
    'DEFAULT_LAMBDA',
    'DEFAULT_NORMALIZE',

    # Validation
    'MIN_LORAS_FOR_MERGE',
    'MIN_KEY_OVERLAP_RATIO',
    'TYPICAL_STRENGTH_MIN',
    'TYPICAL_STRENGTH_MAX',

    # Device and Memory
    'SUPPORTED_DEVICES',
    'SUPPORTED_DTYPES',
    'DEFAULT_DEVICE',
    'DEFAULT_DTYPE',

    # Progress
    'PROGRESS_UPDATE_INTERVAL',
    'PROGRESS_BAR_FORMAT',

    # Layer Filters
    'ATTENTION_LAYERS',
    'MLP_LAYERS',
    'ATTENTION_MLP_LAYERS',
    # Legacy (deprecated)
    'SD_ATTENTION_LAYERS',
    'SD_MLP_LAYERS',
    'SD_ATTENTION_MLP_LAYERS',
    'SD_PROJECTION_LAYERS',
    'DIT_ATTENTION_LAYERS',
    'DIT_MLP_LAYERS',
    'WAN_ATTENTION_LAYERS',
    'WAN_MLP_LAYERS',

    # File I/O
    'LORA_FILE_EXTENSIONS',
    'DEFAULT_LORA_SAVE_FORMAT',

    # Caching
    'MAX_DECOMPOSITION_CACHE_SIZE',
    'CACHE_TTL_SECONDS',

    # Logging
    'DEFAULT_LOG_LEVEL',
    'LOG_FORMAT',
]
