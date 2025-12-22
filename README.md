# LoRA Power-Merger ComfyUI

Advanced LoRA merging for ComfyUI with Mergekit integration, supporting 8+ merge algorithms including TIES, DARE, SLERP, and more. Features modular architecture, SVD decomposition, selective layer filtering, and comprehensive validation.

![pm-merge-methods.png](assets/pm-merge-methods.png)

This is an enhanced fork of laksjdjf's [LoRA Merger](https://github.com/laksjdjf/LoRA-Merger-ComfyUI) with extensive refactoring and new features. Core merging algorithms from [Mergekit](https://github.com/arcee-ai/mergekit) by Arcee AI.

## Features

- **8+ Merge Algorithms**: Task Arithmetic, TIES, DARE, DELLA, Breadcrumbs, SLERP, and more
- **Mergekit Integration**: Production-grade merge methods from Arcee AI's Mergekit library
- **SVD Support**: Full, randomized, and energy-based SVD decomposition with dynamic rank selection
- **Modular Architecture**: Clean separation of concerns with focused, single-responsibility modules
- **DiT Architecture Support**: Automatic layer grouping for Diffusion Transformer models
- **Selective Layer Merging**: Filter by attention layers, MLP layers, or custom patterns
- **Comprehensive Validation**: Runtime type checking and structured error reporting
- **Thread-Safe Processing**: Parallel processing with device-aware workload distribution

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/YourUsername/LoRA-Merger-ComfyUI.git
cd LoRA-Merger-ComfyUI
pip install -r requirements.txt
```

**Requirements:**
- PyTorch
- Mergekit (`git+https://github.com/arcee-ai/mergekit.git`)
- lxml

## Quick Start

### Basic Two-LoRA Merge

![pm-merge_methods.png](assets/pm-merge_methods.png)

1. Stack LoRAs with **PM LoRA Stacker** or **PM LoRA Power Stacker**
2. Decompose using **PM LoRA Stack Decompose**
3. Choose a merge method (e.g., **PM TIES**, **PM DARE**)
4. Merge with **PM LoRA Merger (Mergekit)**
5. Apply with **PM LoRA Apply** or save with **PM LoRA Save**

## Node Reference

### Core Workflow Nodes

#### PM LoRA Stacker
Combine multiple LoRAs into a stack for merging. Dynamically adds connection points as you connect LoRAs.

![pm-lora_stacker.png](assets/pm-lora_stacker.png)

**Inputs:**
- `lora_1`, `lora_2`, ... `lora_N`: LoRABundle inputs (unlimited)

**Output:**
- `LoRAStack`: Dictionary mapping LoRA names to their patch dictionaries

#### PM LoRA Stacker (Directory)
Load all LoRAs from a directory automatically.

![pm-stack_drom_dir.png](assets/pm-stack_drom_dir.png)

**Parameters:**
- `directory`: Path to folder containing LoRA files
- `layer_filter`: Preset filters ("full", "attn-only", "attn-mlp", "mlp-only", "dit-attn", "dit-mlp") or custom
- `sort_by`: "alphabetical" or "modification_time"
- `limit`: Limit number of LoRAs to load (default: 0 for all)


#### PM LoRA Stack Decompose
Decompose LoRA stack into (up, down, alpha) tensor components for merging.

![pm-lora_decomposer.png](assets/pm-lora_decomposer.png)

**Features:**
- **Hash-based caching**: Skips expensive decomposition if inputs unchanged
- **Architecture detection**: Automatically identifies SD vs DiT LoRAs
- **Layer filtering**: Apply preset or custom layer filters

**Parameters:**
- `key_dicts`: Input LoRAStack
- `decomposition_method`: Choose from Standard SVD, Randomized SVD, or Energy-Based Randomized SVD
- `svd_rank`: Target rank for decomposition (0 for full rank)'
- `device`: Processing device ("cpu", "cuda")

**Outputs:**
- `components`: LoRATensors (decomposed tensors by layer)

#### PM LoRA Merger (Mergekit)
Main merging node using Mergekit algorithms. Processes layers in parallel with thread-safe progress tracking.

![pm-lora_merger.png](assets/pm-lora_merger.png)

**Parameters:**
- `merge_method`: MergeMethod configuration from method nodes
- `components`: Decomposed LoRATensors from decompose node
- `strengths`: LoRAWeights from decompose node
- `_lambda`: Final scaling factor (default: 1.0)
- `device`: Processing device ("cpu", "cuda")
- `dtype`: Computation precision ("float32", "float16", "bfloat16")

**Features:**
- **Parallel processing**: ThreadPoolExecutor with max_workers=8
- **Comprehensive validation**: Input structure, tensor shapes, strength presence
- **Smart strength application**: Uses strength_model for UNet, strength_clip for CLIP layers
- **Device-aware distribution**: Balances CPU and GPU workload

**Output:**
- Merged LoRA as LoRAAdapter state dictionary

#### PM LoRA Apply
Apply merged LoRA to a model.

![pm-lora_apply.png](assets/pm-lora_apply.png)

**Inputs:**
- `model`: ComfyUI model to patch
- `lora`: Merged LoRA from merger

**Outputs:**
- `model`: Patched model

#### PM LoRA Save
Save merged LoRA to disk in standard format. This will also save the original clip weights if present.

![pm-save_lora.png](assets/pm-save_lora.png)

**Parameters:**
- `lora`: Merged LoRA to save
- `filename`: Output filename (without extension)

### Merge Method Nodes

Each method node configures algorithm-specific parameters. Connect to the `merge_method` input of **PM LoRA Merger**.

#### PM Linear
Simple weighted linear combination.

**Parameters:**
- `normalize` (bool): Normalize by number of LoRAs (default: True)

#### PM TIES
Task Arithmetic with Interference Elimination and Sign consensus.

**Parameters:**
- `density` (float): Fraction of values to keep (0.0-1.0, default: 0.9)
- `normalize` (bool): Normalize merged result (default: True)

**Reference:** [TIES-Merging Paper](https://arxiv.org/abs/2306.01708)

#### PM DARE
Drop And REscale for efficient model merging.

**Parameters:**
- `density` (float): Probability of keeping each parameter (default: 0.9)
- `normalize` (bool): Normalize after rescaling (default: True)

**Reference:** [DARE Paper](https://arxiv.org/abs/2311.03099)

Depth-Enhanced Low-rank adaptation with Layer-wise Averaging.

**Parameters:**
- `density` (float): Layer density parameter (default: 0.9)
- `epsilon` (float): Small value for numerical stability (default: 1e-8)
- `lambda_factor` (float): Scaling factor (default: 1.0)

#### PM Breadcrumbs
Breadcrumb-based merging strategy.

**Parameters:**
- `density` (float): Path density (default: 0.9)
- `tie_method` ("sum" or "mean"): How to combine tied parameters

#### PM SLERP
Spherical Linear Interpolation for smooth model interpolation.

**Parameters:**
- `t` (float): Interpolation factor (0.0-1.0, default: 0.5)

**Note:** SLERP requires exactly 2 LoRAs. For multiple LoRAs, use **PM NuSLERP** or **PM Karcher**.

#### PM NuSLERP
Normalized Spherical Linear Interpolation for multiple models.

**Parameters:**
- `normalize` (bool): Normalize result to unit sphere (default: True)

#### PM Karcher
Karcher mean on the manifold (generalized SLERP for N models).

**Parameters:**
- `max_iterations` (int): Maximum optimization iterations (default: 100)
- `tolerance` (float): Convergence threshold (default: 1e-6)

#### PM Task Arithmetic
Standard task vector arithmetic (delta merging).

**Parameters:**
- `normalize` (bool): Normalize by number of models (default: False)

#### PM SCE (Selective Consensus Ensemble)
Selective consensus with threshold-based parameter selection.

**Parameters:**
- `threshold` (float): Consensus threshold (default: 0.5)

#### PM NearSwap
Nearest neighbor parameter swapping.

**Parameters:**
- `distance_metric` ("cosine" or "euclidean"): Distance measure

#### PM Arcee Fusion
Arcee's proprietary fusion method for high-quality merges.

**Parameters:**
- Various advanced parameters (see node UI)

### Utility Nodes

#### PM LoRA Resizer
Adjust LoRA rank using SVD decomposition.

**Parameters:**
- `lora`: Input LoRA
- `rank_mode`: Rank selection strategy
  - `target_rank` specifies exact rank
  - `sv_ratio`: Keep singular values above ratio threshold
  - `sv_cumulative`: Keep top N% of cumulative energy
  - `sv_fro`: Frobenius norm-based truncation
- `target_rank` (int): Target rank for fixed mode
- `dynamic_param` (float): Parameter for dynamic modes (ratio/cumulative/fro)
- `device`: Processing device
- `dtype`: Computation precision

**Features:**
- **Dynamic rank selection**: Automatically choose optimal rank based on singular value distribution
- **Statistical reporting**: Shows Frobenius norm retention and singular value retention
- **Conv and linear support**: Handles 2D, 3D, and 4D tensors
- **Decomposer selection**: Choose from Standard SVD, Randomized SVD, or QR factorization

**Output:**
- Resized LoRA with adjusted rank

#### PM LoRA Block Sampler
Sample different block configurations for layer-wise experiments.

#### PM LoRA Stack Sampler
Sample subsets of LoRAs from a stack.

#### PM Parameter Sweep Sampler
Systematically sweep through parameter combinations for merge optimization.

**Features:**
- Cartesian product of parameter ranges
- Support for strength, density, rank variations
- Export results for analysis

### Power Stacker Node

#### PM LoRA Power Stacker
Advanced stacking with per-LoRA configuration and dynamic input management.


**Features:**
- **Dynamic LoRA inputs**: Add unlimited LoRAs with individual strength controls
- **Per-LoRA strengths**: Separate strength_model and strength_clip for each LoRA
- **Architecture detection**: Automatically identifies SD vs DiT LoRAs
- **Layer filtering**: Built-in preset and custom layer filters

## SVD and Decomposition

The project includes a comprehensive tensor decomposition system with multiple strategies:

### Decomposition Methods

#### Standard SVD
Full singular value decomposition for exact low-rank approximation.

**Use case:** High accuracy, small to medium tensors

#### Randomized SVD
Fast approximate SVD using randomized linear algebra.

**Use case:** Large tensors where speed is critical

#### Energy-Based Randomized SVD
Adaptive SVD that automatically selects rank based on energy threshold.

**Use case:** Automatic rank selection with quality guarantees

### Error Handling

All decomposers include:
- **GPU failure fallback**: Automatically retries on CPU if GPU decomposition fails
- **Zero matrix detection**: Gracefully handles degenerate cases
- **Shape validation**: Automatic reshape for 2D/3D/4D tensors (conv and linear layers)
- **Numerical stability**: Epsilon regularization for near-singular matrices

## Layer Filtering

Selective merging allows targeting specific layer types:

### Preset Filters

- `"full"`: All layers (no filter)
- `"attn-only"`: Only attention layers (attn1, attn2)
- `"attn-mlp"`: Attention + feed-forward (attn1, attn2, ff)
- `"mlp-only"`: Only feed-forward layers
- `"dit-attn"`: DiT attention layers
- `"dit-mlp"`: DiT MLP layers

### Custom Filters

Specify layer keys as comma-separated string or set:
```
"attn1, attn2, ff.net.0"
```

## Architecture Support

### Stable Diffusion LoRAs
Automatic detection and handling of:
- **UNet blocks**: Input, middle, output blocks
- **Attention layers**: attn1 (self-attention), attn2 (cross-attention)
- **Feed-forward**: ff.net layers
- **CLIP text encoder**: text_model layers

### DiT (Diffusion Transformer) LoRAs
Automatic layer grouping for:
- **Joint blocks**: Unified transformer blocks
- **Attention**: Multi-head attention layers
- **MLP**: Feed-forward networks
- **Positional encoding**: Learned position embeddings

The system automatically detects architecture and applies appropriate decomposition strategies.


## License

MIT License

Copyright (c) 2024 LoRA Power-Merger Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Credits

- Original LoRA Merger by [laksjdjf](https://github.com/laksjdjf/LoRA-Merger-ComfyUI)
- Mergekit by [Arcee AI](https://github.com/arcee-ai/mergekit)
- TIES-Merging: [Paper](https://arxiv.org/abs/2306.01708)
- DARE: [Paper](https://arxiv.org/abs/2311.03099)
- ComfyUI by [comfyanonymous](https://github.com/comfyanonymous/ComfyUI)
