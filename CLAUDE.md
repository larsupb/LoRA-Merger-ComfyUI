# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LoRA Power-Merger is a ComfyUI custom node extension for merging LoRA (Low-Rank Adaptation) models using various advanced techniques. It's a fork of laksjdjf's LoRA Merger with enhanced functionality including Mergekit integration, DARE merge, SVD support, and XY plot capabilities.

## Key Dependencies

- **ComfyUI**: This is a custom node for ComfyUI, so all nodes integrate with ComfyUI's node system
- **Mergekit**: Core merging algorithms from `arcee-ai/mergekit` (installed via git+https://github.com/arcee-ai/mergekit.git)
- **PyTorch**: All tensor operations use PyTorch
- **lxml**: Required dependency

Install dependencies:
```bash
pip install -r requirements.txt
```

## Architecture Overview

### Node Registration System
All nodes are registered in `__init__.py` via two dictionaries:
- `NODE_CLASS_MAPPINGS`: Maps node IDs to Python classes
- `NODE_DISPLAY_NAME_MAPPINGS`: Maps node IDs to display names in ComfyUI UI

### Core Components

**LoRA Data Flow:**
1. **Loading**: LoRAs are loaded from disk as raw tensors
2. **Decomposition**: LoRAs are decomposed into (up, down, alpha) tuple components via `LoraDecompose`
3. **Stacking**: Multiple LoRAs are organized into dictionaries with strength metadata via `LoraStack` or `LoraStackFromDir`
4. **Merging**: Components are merged using various algorithms in `LoraMergerMergekit`
5. **Application**: Merged LoRAs are applied to models via `LoraApply` or saved via `LoraSave`

### Module Structure (Refactored)

The codebase is organized into focused, single-responsibility modules:

**Core Modules:**
- `src/types.py` - Centralized type definitions, validators, and type guards (332 lines)
- `src/merge/` - Merge operations (4 files, 872 lines)
  - `utils.py` - Helper functions (tensor weighting, gathering)
  - `algorithms.py` - All 8 merge algorithm implementations
  - `dispatcher.py` - Registry-based method routing
  - `base_node.py` - Base classes for merge method nodes (reduces boilerplate by ~60%)
- `src/device/` - Device and dtype management
  - `manager.py` - DeviceManager class for unified device/dtype handling
- `src/decomposition/` - Tensor decomposition (3 files, 770 lines)
  - `base.py` - TensorDecomposer abstract base class
  - `svd.py` - SVD, randomized SVD, QR implementations with error handling
- `src/validation/` - Input validation (2 files, 550 lines)
  - `validators.py` - LoRAStackValidator, TensorShapeValidator, MergeParameterValidator
- `src/utils/` - Utilities (3 files, 350 lines)
  - `config.py` - All configuration constants and magic numbers
  - `layer_filter.py` - LayerFilter class for selective merging
  - `progress.py` - ThreadSafeProgressBar wrapper

**Architecture Modules:**
- `src/architectures/` - LoRA architecture-specific code
  - `general_architecture.py` - Re-exports types for backward compatibility
  - `sd_lora.py` - Stable Diffusion LoRA handling (up/down/alpha extraction, key analysis, block detection)
  - `dit_lora.py` - DiT (Diffusion Transformer) architecture support with automatic layer grouping

**Semantic Analysis Module (NEW):**
- `src/analysis/` - Semantic feature attribution for LoRAs (9 files, ~3,500 lines)
  - `gradient_analyzer.py` - GradientSemanticAnalyzer for LoRA feature analysis
  - `sd_loader.py` - Load SD1.5/SDXL UNet models for gradient analysis (diffusers-based)
  - `cache.py` - Semantic importance map caching system
  - `feature_prompts.py` - Feature isolation prompt templates
  - `feature_timesteps.py` - Timestep configuration for gradient analysis
  - `semantic_merger.py` - Semantic LoRA merging with importance maps
  - `merge_statistics.py` - Merge statistics collection and reporting
  - `lora_loader.py` - LoRA file I/O with key normalization
  - `key_utils.py` - Key normalization utilities for diffusers/kohya formats
  - Infrastructure for text-guided, feature-aware LoRA merging

**Supported LoRA Architectures:**
- **Stable Diffusion (SD1.5/SDXL)**: UNet architecture with `down_blocks`, `up_blocks`, `mid_block`. Uses underscore-separated naming like `lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q.lora_down.weight`. Attention layers: `attn1`/`attn2`, MLP layers: `ff`.
- **DiT (Diffusion Transformer)**: Flat transformer with dot-separated naming like `diffusion_model.layers.13.attention.qkv.weight`. Attention layers: `attention`, MLP layers: `mlp`/`feed_forward`.
- **zImage**: Pure DiT architecture with dot-separated naming like `diffusion_model.layers.0.attention.to_k.lora_A.weight`. Attention layers: `attention`, MLP layers: `feed_forward` (w1, w2, w3). Also includes `adaLN_modulation` layers (adaptive layer norm).
- **Flux**: Modern architecture with `double_blocks` and `single_blocks`. Uses underscore-separated naming like `lora_unet_double_blocks_0_img_attn_proj.alpha`. Attention layers: `img_attn_*`/`txt_attn_*` (matches generic `attn` pattern), MLP layers: `img_mlp_*`/`txt_mlp_*`.
- **Wan 2.2**: Transformer with dot-separated naming like `diffusion_model.blocks.0.cross_attn.q.lora_A.weight`. Attention layers: `self_attn`/`cross_attn`, MLP layers: `ffn`.
- **Qwen Image Edit**: Transformer blocks with dot-separated naming like `transformer_blocks.0.attn.to_k.alpha`. Attention layers: `.attn.` (to_k, to_q, to_v, add_k_proj, etc.), MLP layers: `img_mlp`, `txt_mlp`.

### Type System (src/types.py)

**ComfyUI Node Interface Types:**
- `"LoRABundle"`: Single LoRA with metadata wrapper (dict with lora_raw/lora/strength_model/name)
- `"LoRAStack"`: Multiple LoRAs with metadata (Dict[LoRA name -> {"patches": Dict[layer key -> LoRAAdapter], "file_path": str}])
- `"LoRATensors"`: Decomposed LoRA tensors ready for merging (Dict[layer key -> Dict[LoRA name -> (up, down, alpha)]])
- `"LoRAWeights"`: Per-LoRA strength values (Dict[LoRA name -> {"strength_model": float}])
- `"MergeMethod"`: Merge algorithm configuration (Dict with "name" and "settings" keys)

**Python Type Aliases:**
- `LORA_TENSORS`: Tuple of (up_tensor, down_tensor, alpha_value) - the core LoRA tensor representation
- `LORA_TENSOR_DICT`: Dict mapping LoRA names to LORA_TENSORS
- `LORA_TENSORS_BY_LAYER`: Dict mapping layer keys to LORA_TENSOR_DICT
- `LoRAStackEntry`: TypedDict with "patches" (LORA_KEY_DICT) and "file_path" (str) - metadata wrapper for each LoRA
- `LORA_STACK`: Dict mapping LoRA names to LoRAStackEntry (includes both patches and file path metadata)
- `LORA_WEIGHTS`: Dict mapping LoRA names to strength_model/strength_clip values

**Type Guards and Validators:**
- `is_lora_tensors()`, `is_lora_stack()` - Runtime type checking
- `validate_lora_tensors()`, `validate_lora_stack()` - Runtime validation with structured errors

### Merging Methods (nodes_merge_methods.py)

**Base Class Pattern:**
Merge method nodes now extend `BaseMergeMethodNode` or `BaseTaskArithmeticNode` to reduce boilerplate:

```python
class LinearMergeMethod(BaseMergeMethodNode):
    METHOD_NAME = "linear"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"normalize": ("BOOLEAN", {"default": True})}}

    def get_settings(self, normalize: bool = True):
        return {"normalize": normalize}
```

Each method returns a method definition dict:
```python
{
    "name": "method_name",
    "settings": { /* method-specific parameters */ }
}
```

**Method Categories:**
- **Task Arithmetic**: task_arithmetic, ties, dare, della, breadcrumbs
- **Spherical Interpolation**: slerp, nuslerp, karcher
- **Specialized**: linear, sce, nearswap

**All merge implementations are in `src/merge/algorithms.py`** with comprehensive documentation.

### Main Merger (lora_mergekit_merge.py)

**Refactored from 790 → 482 lines (-39%)**

**LoraMergerMergekit workflow:**
1. Receives decomposed components and strengths
2. **Validates input** using `src/validation/validators.py`:
   - Component existence and structure
   - Tensor shape compatibility
   - Strength presence for all LoRAs
3. Processes each layer key in parallel (ThreadPoolExecutor with max_workers=8)
4. For each key:
   - Extracts up/down/alpha tuples for all LoRAs
   - Scales by strength (strength_model for UNet layers, strength_clip for CLIP)
   - Dispatches to merge method via `src/merge/dispatcher.py`
   - Applies lambda scaling to results
5. Returns merged LoRA as LoRAAdapter state dict

**Thread Safety:** Uses ThreadPoolExecutor for parallel processing of keys. Work is distributed across available devices (CPU + specified device).

**Method Dispatch:** Now uses registry pattern via `get_merge_method()` instead of if-elif chain.

### CLIP Layer Merging

**Overview:**
The merger now intelligently handles CLIP (text encoder) layers separately from UNet layers, using different merge strategies for each.

**Design Philosophy:**
- **UNet layers**: Use advanced merge methods (TIES, DARE, SLERP, etc.) for sophisticated model merging
- **CLIP layers**: Use simple weighted average for predictable, stable text encoder merging
- **Separation**: Automatic detection and routing based on layer key patterns

**CLIP Detection (`src/utils/layer_filter.py`):**
```python
from src.utils import is_clip_layer

# Detect CLIP vs UNet layers
is_clip_layer("lora_te_text_model_encoder_layers_0_self_attn_q_proj")  # True
is_clip_layer("lora_unet_down_blocks_0_attentions_0_attn1_to_q")  # False
```

**CLIP Key Patterns:**
ComfyUI Internal Formats:
- `clip_l.*` - CLIP-L (OpenAI CLIP Large, used in SD1.5)
- `clip_g.*` - CLIP-G (OpenCLIP bigG, used in SDXL)
- `clip_h.*` - CLIP-H (OpenCLIP Huge)

LoRA File Formats:
- `lora_te*` - SD1.5 text encoder
- `text_encoder*` - Generic text encoder
- `lora_te1_text_model*` - SDXL text encoder 1
- `lora_te2_text_model*` - SDXL text encoder 2
- `text_model*` - Generic text model
- `transformer.text_model*` - Transformer text model

**Merge Behavior:**
1. **Layer Separation**: During merge, keys are split into CLIP and UNet groups
2. **Weight Selection**:
   - CLIP layers use `strength_clip` from LoRAWeights
   - UNet layers use `strength_model` from LoRAWeights
3. **Merge Method**:
   - CLIP: `simple_weighted_average()` (normalized linear interpolation)
   - UNet: Selected merge method (TIES/DARE/SLERP/etc.)
4. **Output**: Both merged CLIP and UNet layers are combined in the final LoRAAdapter

**Simple Weighted Average (`src/merge/utils.py`):**
```python
from src.merge import simple_weighted_average

# Merge CLIP tensors
tensors = {"lora1": tensor1, "lora2": tensor2}
weights = {"lora1": 0.3, "lora2": 0.7}
merged = simple_weighted_average(tensors, weights, normalize=True)
# Result: 0.3 * tensor1 + 0.7 * tensor2
```

**Application:**
The `LoraApply` node now supports optional CLIP input/output:
```python
# In ComfyUI workflow
model, clip = LoraApply(model, clip, merged_lora)
# CLIP layers are applied using strength_clip
# UNet layers are applied using strength_model
```

**Benefits:**
- **Consistency**: CLIP merging respects `strength_clip` parameter
- **Stability**: Simple averaging avoids complex merge artifacts in text understanding
- **Separation of Concerns**: Advanced methods for visual generation, simple methods for text encoding
- **Backward Compatible**: CLIP input is optional; works with existing workflows

### Layer Filtering (src/utils/layer_filter.py)

**Architecture Detection:**
The layer filter system now includes automatic architecture detection. When a LoRA is loaded, the system analyzes the layer keys and logs the detected architecture.

```python
from src.utils import detect_lora_architecture

# Detect architecture
arch_name, metadata = detect_lora_architecture(lora_patches)
print(f"Detected {arch_name} architecture")
# Output: "Detected Wan 2.2 architecture"
```

**Supported Architecture Detection:**
- **Wan 2.2**: Identifies `diffusion_model.blocks.N` with `self_attn`/`cross_attn`/`ffn`
- **Flux**: Identifies `double_blocks`/`single_blocks` with `img_attn`/`txt_attn`/`img_mlp`/`txt_mlp`
- **Qwen Image Edit**: Identifies `transformer_blocks.N.attn` with `img_mlp`/`txt_mlp`
- **zImage**: Identifies `diffusion_model.layers.N` with `adaLN_modulation`
- **DiT**: Identifies `diffusion_model.layers.N` with `attention`/`feed_forward`
- **SD1.5 vs SDXL**: Distinguishes based on `lora_unet` patterns and key count

**LayerFilter Class:**
```python
from src.utils import LayerFilter

# Create filter
filter = LayerFilter("attn-only")  # Preset
filter = LayerFilter({"attn1", "attn2"})  # Custom

# Apply to patches (automatically detects architecture)
filtered_patches = filter.apply(lora_patches)
# Log output: "Detected Wan 2.2 architecture (400 keys)"
```

**Available Presets (Architecture-Agnostic):**
- `"full"`: All layers (no filter)
- `"attn-only"`: Only attention layers
  - SD: `attn1`, `attn2`
  - DiT: `attention`
  - zImage: `attention`
  - Flux: `img_attn_*`, `txt_attn_*` (matched by generic `attn`)
  - Wan: `self_attn`, `cross_attn`
  - Qwen: `.attn.` (to_k, to_q, to_v, add_k_proj, etc.)
- `"attn-mlp"`: Attention + MLP layers (combines both above)
- `"mlp-only"`: Only MLP/feedforward layers
  - SD: `ff`
  - DiT: `mlp`, `feed_forward`
  - zImage: `feed_forward`
  - Flux: `img_mlp_*`, `txt_mlp_*`
  - Wan: `ffn`
  - Qwen: `img_mlp`, `txt_mlp`

**Layer Filter Implementation:**
Uses **word-boundary aware matching** to avoid false positives. The regex pattern `(?:^|[._])pattern(?:[._]|$)` ensures patterns only match when surrounded by dots or underscores. For example, `'ff'` will match `.ff.` or `_ff_` but NOT `diffusion_model` (where "ff" appears as a substring). This ensures precise filtering across all six supported architectures (SD, DiT, zImage, Flux, Wan, Qwen) while avoiding accidental matches.

**ComfyUI Compatibility:**
The layer filter and architecture detection functions handle both string keys and tuple keys (ComfyUI's internal format). Keys are automatically converted to strings for pattern matching, ensuring compatibility with ComfyUI's node system.

### Semantic LoRA Merging (nodes_semantic_merge.py, src/analysis/)

**NEW: Text-Guided Feature-Aware Merging**

The semantic merge system enables merging LoRAs by selectively combining specific visual features: "take the hair from Character 1 and the clothing from Character 2."

**How It Works:**

1. **Semantic Analysis** (`PM LoRA Semantic Analyzer (Heuristic)`):
   - Takes a **LoRAStack** as input (analyzes all LoRAs in one pass)
   - Creates **semantic importance maps** for each LoRA in the stack
   - Maps features (hair, eyes, clothing, accessories, etc.) to layer importance scores
   - Current implementation uses depth-based heuristics:
     - Early layers (0-33%): body, composition, pose
     - Mid layers (33-66%): face, hair, eyes
     - Late layers (66-100%): clothing, accessories, details
   - Results are cached per-LoRA for fast reuse
   - Outputs: `SemanticMaps` dict mapping LoRA names to their feature importance maps

2. **Merge Specification** (`PM Semantic Merge Spec`):
   - Parse natural language feature assignments:
     ```
     hair from character1, eyes from character1, clothing from character2
     ```
   - Optional per-feature weights:
     ```
     hair from lora1 1.2, clothing from lora2 0.8
     ```

3. **Semantic Merger** (`PM Semantic Merger`):
   - Combines LoRAs layer-by-layer using feature importance scores
   - Weights each LoRA's contribution based on requested features
   - Formula: `merged_weight = Σ (importance[feature] × lora[source] × weight)`

**Available Features:**
- `hair` - Hairstyle, hair color (peaks at 50% depth)
- `eyes` - Eye shape, color (peaks at 50% depth)
- `face` - Facial structure (peaks at 40% depth)
- `clothing` - Outfit, style (peaks at 70% depth)
- `accessories` - Jewelry, decorations (peaks at 80% depth)
- `body` - Body proportions (peaks at 20% depth)

**Example Workflow:**
```
[LoRA Loader] ──┐
[LoRA Loader] ──┼→ [PM LoRA Power Stacker] ──┬→ [PM Semantic Analyzer] ─┐
                                              │                           │
                                              └───────────────────────────┤
                                                                          ├→ [PM Semantic Merger] → [Apply/Save]
[PM Semantic Merge Spec] ─────────────────────────────────────────────────┘
  "hair from lora1, clothing from lora2"
```

**Architecture Note:** The analyzer takes the entire LoRAStack as input and analyzes all LoRAs in a single pass, outputting semantic maps for all of them. This design matches the existing LoRA decomposition workflow pattern.

**Caching:**
- Semantic maps cached in `cache/semantic_maps/`
- Cache key: LoRA file path + tensor checksums + analysis config
- Average cache hit saves ~10-30 seconds per LoRA

**Future Enhancements:**
The `src/analysis/` module includes infrastructure for:
- **Gradient-based attribution**: Integrated Gradients for precise feature importance
- **Attention flow tracking**: Cross-attention analysis for spatial feature detection
- **CLIP-guided refinement**: Iterative optimization based on text-image similarity

See [SEMANTIC_MERGE.md](SEMANTIC_MERGE.md) for detailed documentation and [SEMANTIC_MERGE_QUICKSTART.md](SEMANTIC_MERGE_QUICKSTART.md) for a 5-minute tutorial.

### SVD and Tensor Decomposition (src/decomposition/)

**Decomposition Hierarchy:**
The refactored architecture provides a class-based decomposition system with error handling:

```python
from src.decomposition import (
    SVDDecomposer,
    RandomizedSVDDecomposer,
    EnergyBasedRandomizedSVDDecomposer,
    QRDecomposer,
)

# Use decomposer
decomposer = SVDDecomposer(device=device)
up, down, stats = decomposer.decompose(
    weight=tensor,
    target_rank=32,
    dynamic_method="sv_ratio",
    dynamic_param=0.9,
    scale=1.0
)
```

**TensorDecomposer Base Class Features:**
- **Error handling**: Try-catch with CPU fallback for GPU failures
- **Rank selection strategies**: ratio, cumulative, Frobenius norm
- **Statistics calculation**: Singular value retention, Frobenius norm retention
- **Shape handling**: Automatic reshape for 2D/3D/4D tensors (conv and linear layers)
- **Zero matrix detection**: Graceful handling of degenerate cases

**Available Decomposers:**
- `SVDDecomposer` - Standard full SVD
- `RandomizedSVDDecomposer` - Fast approximate SVD for large tensors
- `EnergyBasedRandomizedSVDDecomposer` - Adaptive SVD with energy threshold
- `QRDecomposer` - QR factorization (faster, no singular values)

**Legacy Functions (utility.py):**
For backward compatibility, `perform_lora_svd()` and `adjust_tensor_dims()` remain:
- **Dynamic rank selection**: `sv_ratio`, `sv_cumulative`, `sv_fro`, or fixed rank
- **Singular value distribution modes**:
  - Symmetric: `√S` split between up and down (used by `adjust_tensor_dims`)
  - Asymmetric: All `S` in up matrix (used by LoRA Resizer node)
- **Statistical reporting**: Frobenius norm retention, singular value retention
- **Conv and linear layers**: Automatically handles 2D, 3D, and 4D tensors
- **Interrupt handling**: Can be canceled during long operations

### Caching Strategy
`LoraDecompose` implements hash-based caching:
- Computes hash of LoRA names and tensor sum
- Skips expensive decomposition if inputs haven't changed
- Saves significant time when re-running with same LoRAs

## Development Patterns

### Adding a New Merge Method

**Step 1: Create the node class** (extends base class for reduced boilerplate):

```python
# In nodes_merge_methods.py
from src.merge.base_node import BaseMergeMethodNode

class MyNewMergeMethod(BaseMergeMethodNode):
    METHOD_NAME = "my_new_method"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "my_param": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0}),
            }
        }

    def get_settings(self, my_param: float = 1.0):
        return {"my_param": my_param}
```

**Step 2: Implement the merge algorithm:**

```python
# In src/merge/algorithms.py
from .utils import apply_weights_to_tensors

def my_new_merge(tensors, gather_tensors, weight_info, tensor_parameters, method_args):
    """
    My new merge algorithm.

    Args:
        tensors: Dict[str, torch.Tensor] - Tensors to merge
        gather_tensors: Callable - Gather tensors by reference
        weight_info: Dict - Weight metadata
        tensor_parameters: Dict - Tensor-specific parameters (weights)
        method_args: Dict - Method settings from node

    Returns:
        torch.Tensor - Merged result
    """
    weighted_tensors = apply_weights_to_tensors(tensors, tensor_parameters)
    # ... your merge logic
    return merged_tensor
```

**Step 3: Register in algorithm registry:**

```python
# In src/merge/dispatcher.py, add to MERGE_ALGORITHMS:
MERGE_ALGORITHMS = {
    # ... existing methods
    "my_new_method": my_new_merge,
}
```

**Step 4: Register node in ComfyUI:**

```python
# In __init__.py
NODE_CLASS_MAPPINGS = {
    # ... existing nodes
    "PM MyNewMerge": MyNewMergeMethod,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # ... existing mappings
    "PM MyNewMerge": "PM My New Merge",
}
```

That's it! The registry pattern automatically routes method calls.

### ComfyUI Node Conventions
- All nodes have `INPUT_TYPES()` classmethod
- `RETURN_TYPES` tuple defines output types (inherited from base class for merge methods)
- `FUNCTION` string names the method to call (inherited from base class)
- `CATEGORY` organizes nodes in UI (use "LoRA PowerMerge" prefix)
- `DESCRIPTION` provides tooltip documentation

### Device and Dtype Management

**New approach (preferred):**
```python
from src.device import DeviceManager

device, dtype = DeviceManager.parse(device_str, dtype_str)
```

**Legacy approach (still supported):**
```python
from src.utility import map_device

device, dtype = map_device(device_str, dtype_str)
```

### Progress Bars

**Thread-safe progress tracking:**
```python
from src.utils.progress import ThreadSafeProgressBar

with ThreadSafeProgressBar(total_steps, desc="Processing") as pbar:
    for item in items:
        # ... do work
        pbar.update(1)
```

**Legacy ComfyUI progress bar:**
```python
pbar = comfy.utils.ProgressBar(total_steps)
# ... in loop:
pbar.update(1)
```

### Input Validation

Use validators for robust error handling:

```python
from src.validation.validators import (
    validate_lora_stack_structure,
    validate_tensor_shapes_compatible,
    validate_merge_parameters,
)

# Validate LoRA stack
result = validate_lora_stack_structure(lora_stack, min_loras=2)
if result["errors"]:
    raise ValueError(f"Invalid stack: {result['errors']}")

# Validate tensor compatibility
result = validate_tensor_shapes_compatible(components)
for warning in result["warnings"]:
    logging.warning(warning)
```

## Important Notes

### Mergekit Integration
- Mergekit methods expect task vectors (deltas from base model)
- For LoRA merging, we create dummy zero base tensors since LoRAs are already deltas
- GTA (Generalized Task Arithmetic) methods require special handling for mode selection (ties vs linear variants)

### Memory Management
- Large tensor operations offload to CPU after processing
- `torch.cuda.empty_cache()` called after major operations
- ThreadPoolExecutor distributes work across CPU and GPU to balance load

### XY Plot Nodes (Deprecated)
XY plot nodes are commented out in `__init__.py` but remain in `deprecated/lora_merge_xy.py`. They required the efficiency-nodes-comfyui extension.

### Experimental Features
The `experimental/` directory contains work-in-progress features:
- LoRA Analyzer
- Attention logging and plotting
- Checkpoint (full model) merging
- LoRA pruning

## File Structure Reference

```
.
├── __init__.py                    # Node registration
├── lora_mergekit_merge.py        # Main merger + stacking nodes (refactored: 482 lines)
├── nodes_merge_methods.py        # Merge method node definitions (using base classes)
├── lora_apply.py                 # Apply merged LoRA to model
├── lora_save.py                  # Save LoRA to disk
├── lora_load.py                  # Load LoRA with LBW support
├── lora_resize.py                # Resize LoRA rank via SVD
├── lora_block_sampler.py         # Block-wise LoRA sampling
├── lora_stack_sampler.py         # Stack-based LoRA sampling
├── nodes_lora_modifier.py        # LoRA modification utilities
├── utility.py                    # Helper functions (legacy SVD, device mapping, etc.)
├── mergekit_utils.py             # Mergekit integration utilities
├── src/
│   ├── types.py                  # [NEW] Centralized type system (332 lines)
│   ├── merge/                    # [NEW] Merge operations (872 lines)
│   │   ├── __init__.py
│   │   ├── utils.py              # Helper functions (tensor weighting, gathering)
│   │   ├── algorithms.py         # All 8 merge algorithm implementations
│   │   ├── dispatcher.py         # Registry-based method routing
│   │   └── base_node.py          # Base classes for merge method nodes
│   ├── device/                   # [NEW] Device management (181 lines)
│   │   ├── __init__.py
│   │   └── manager.py            # DeviceManager class
│   ├── decomposition/            # [NEW] Tensor decomposition (770 lines)
│   │   ├── __init__.py
│   │   ├── base.py               # TensorDecomposer abstract base class
│   │   └── svd.py                # SVD/Randomized SVD/QR implementations
│   ├── validation/               # [NEW] Input validation (550 lines)
│   │   ├── __init__.py
│   │   └── validators.py         # Stack, shape, parameter validators
│   ├── utils/                    # [NEW] Utilities (350 lines)
│   │   ├── __init__.py
│   │   ├── config.py             # All configuration constants
│   │   ├── layer_filter.py       # LayerFilter class for selective merging
│   │   └── progress.py           # ThreadSafeProgressBar wrapper
│   └── architectures/            # Architecture-specific LoRA handling
│       ├── __init__.py
│       ├── general_architecture.py  # Type re-exports for compatibility
│       ├── sd_lora.py            # Stable Diffusion LoRA handling
│       └── dit_lora.py           # DiT architecture support
├── tests/                        # [NEW] Comprehensive test suite (1,058 lines)
│   ├── __init__.py
│   ├── test_types.py             # Type system tests (241 lines, 15+ tests)
│   ├── test_algorithms.py        # Algorithm tests (217 lines)
│   ├── test_decomposition.py     # Decomposition tests (280 lines, 20+ tests)
│   └── test_validation.py        # Validation tests (320 lines, 25+ tests)
├── deprecated/                   # Old XY plot nodes
├── experimental/                 # WIP features (LoRA Analyzer, checkpoint merge, etc.)
└── js/                           # Frontend ComfyUI widgets
```

**Key Improvements:**
- **17 new focused modules** (from 2-3 large files)
- **3,700+ lines** of new/refactored code
- **60+ unit tests** across 4 test files
- **100% type hint coverage** in refactored modules
- **Zero code duplication** in merge algorithms

## Testing and Validation

**Automated Test Suite:**
Run unit tests with pytest:
```bash
# From repository root
pytest tests/

# With coverage
pytest --cov=src tests/

# Specific test file
pytest tests/test_algorithms.py -v
```

**Test Coverage:**
- `test_types.py`: Type guards, validators, fixtures (15+ tests)
- `test_algorithms.py`: All merge algorithms with mocked tensors
- `test_decomposition.py`: SVD, randomized SVD, QR, error handling (20+ tests)
- `test_validation.py`: Stack, shape, parameter validation (25+ tests)

**Manual Testing Workflow:**
1. Load LoRAs using PM LoRA Loader or PM LoRA Stacker nodes
2. Connect to PM LoRA Stack Decompose
3. Select merge method node (e.g., PM TIES)
4. Connect to PM LoRA Merger (Mergekit)
5. Apply or save the result
6. Verify output quality visually in image generation

## Migration Guide (for Custom Extensions)

**If you have custom code extending this project**, you may need to update imports after the refactoring:

### Import Path Changes

**Type Definitions:**
```python
# Old
from .architectures.sd_lora import LORA_TENSORS, LORA_STACK
from .architectures.general_architecture import LORA_WEIGHTS

# New (preferred)
from .types import LORA_TENSORS, LORA_STACK, LORA_WEIGHTS

# Old imports still work for backward compatibility
```

**Device Management:**
```python
# Old
from .utility import map_device
device, dtype = map_device(device_str, dtype_str)

# New (preferred)
from .device import DeviceManager
device, dtype = DeviceManager.parse(device_str, dtype_str)

# Old function still works
```

**Merge Algorithms (if accessing directly):**
```python
# Old
# Algorithms were in lora_mergekit_merge.py

# New
from src.merge.algorithms import (
    linear_merge,
    slerp_merge,
    ties_merge,
    # ... etc
)
from src.merge.dispatcher import get_merge_method
merge_fn = get_merge_method("ties")
```

**Validation:**
```python
# New validators available
from src.validation.validators import (
    validate_lora_stack_structure,
    validate_tensor_shapes_compatible,
    validate_merge_parameters,
)
```

**Layer Filtering:**
```python
# Old
from .utility import parse_layer_filter, apply_layer_filter
filter_set = parse_layer_filter("attn-only")
filtered = apply_layer_filter(patches, filter_set)

# New (preferred)
from src.utils.layer_filter import LayerFilter
filter = LayerFilter("attn-only")
filtered = filter.apply(patches)

# Old functions still work
```

### Node Class Changes

**For custom merge method nodes:**
```python
# Old approach (still works)
class MyMergeMethod:
    RETURN_TYPES = ("MergeMethod",)
    FUNCTION = "get_method"
    CATEGORY = "LoRA PowerMerge"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {...}}

    def get_method(self, **kwargs):
        return ({"name": "my_method", "settings": {...}},)

# New approach (recommended, reduces boilerplate)
from src.merge.base_node import BaseMergeMethodNode

class MyMergeMethod(BaseMergeMethodNode):
    METHOD_NAME = "my_method"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {...}}

    def get_settings(self, **kwargs):
        return {...}  # Just settings, name is automatic
```

**No changes required for ComfyUI workflows** - all node interfaces remain the same.

## Common Workflows

**Basic Two-LoRA Merge:**
```
LoRA Loader → LoRA Stacker → LoRA Decompose → Merger → Apply/Save
               ↑
           LoRA Loader (lora2)
```

**Batch Directory Merge:**
```
LoRA Stacker (from Directory) → LoRA Decompose → Merger → Apply/Save
```

**Custom Method Merge:**
```
LoRA components → Method Node (TIES/DARE/etc.) → Merger → Output
                                                    ↑
                                               LoRA Weights
```

## Code Quality Metrics

The refactoring achieved significant improvements in code quality:

- **Reduced complexity**: Main merge file reduced from 790 → 482 lines (-39%)
- **Eliminated duplication**: 5x weighted tensor pattern consolidated into single utility
- **Type safety**: 100% type hint coverage in refactored modules
- **Test coverage**: 60+ unit tests across 4 test files
- **Modularity**: 17 focused modules (from 2-3 large files)
- **Error handling**: Comprehensive validation and CPU fallback strategies
- **Maintainability**: Clear separation of concerns, centralized configuration
