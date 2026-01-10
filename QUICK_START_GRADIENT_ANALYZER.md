# Quick Start: Gradient-Based Semantic Analyzer

## TL;DR - Get Started in 30 Seconds

1. Add `PM LoRA Semantic Analyzer (Gradient)` node
2. Click `checkpoint_name` dropdown → select your model
3. Connect CLIP from any CLIP loader
4. Connect lora_stack from LoRA stacker
5. Run!

## The Easiest Way (Dropdown Method)

```
┌─────────────────────────────┐
│ CLIP Loader                 │
│   └─ clip ─────────────┐    │
└─────────────────────────┘    │
                               │
┌─────────────────────────────┤
│ PM LoRA Power Stacker       │
│   └─ lora_stack ────────┐   │
└─────────────────────────┘   │
                              │
┌─────────────────────────────┴──────────────────────┐
│ PM LoRA Semantic Analyzer (Gradient)               │
│                                                     │
│  checkpoint_name: [▼ checkpoints/flux_dev.safeten…]│ ← Click here!
│  clip: ←──────────────────────────────────────────│
│  lora_stack: ←─────────────────────────────────────│
│  features: hair,eyes,face,clothing,accessories     │
│  use_cache: ✓                                      │
│                                                     │
│ Output: semantic_maps ──────────────────────────→  │
└────────────────────────────────────────────────────┘
```

## What the Dropdown Shows

When you click `checkpoint_name`, you'll see all your models:

```
┌──────────────────────────────────────────┐
│ checkpoint_name                          │
├──────────────────────────────────────────┤
│                                          │ ← Empty (for MODEL auto-extract)
│ checkpoints/flux_dev.safetensors         │
│ checkpoints/flux_schnell.safetensors     │
│ checkpoints/sdxl_base_1.0.safetensors    │
│ diffusion_models/zimage_v1.safetensors   │
│ diffusion_models/wan_2.2.safetensors     │
└──────────────────────────────────────────┘
```

**Models come from:**
- `ComfyUI/models/checkpoints/` → Shows as `checkpoints/...`
- `ComfyUI/models/diffusion_models/` → Shows as `diffusion_models/...`

## Complete Workflow Example

### Setup

```
1. [CLIPLoader]
      └─ clip

2. [PM LoRA Loader] ─┐
                     │
3. [PM LoRA Loader] ─┼─→ [PM LoRA Power Stacker]
                     │        └─ lora_stack
4. [PM LoRA Loader] ─┘

5. [PM LoRA Semantic Analyzer (Gradient)]
      ├─ checkpoint_name: Select from dropdown
      ├─ clip: ← Connect from step 1
      ├─ lora_stack: ← Connect from step 2
      └─ features: "hair,eyes,clothing"

6. [PM Semantic Merge Spec]
      └─ specification: "hair from lora1, clothing from lora2"

7. [PM Semantic Merger]
      ├─ lora_stack: ← Connect from step 2
      ├─ semantic_maps: ← Connect from step 5
      └─ merge_spec: ← Connect from step 6

8. [PM LoRA Apply] or [PM LoRA Save]
      └─ lora_bundle: ← Connect from step 7
```

## Dropdown vs MODEL Auto-Extract

### Use Dropdown When:
- Quick testing
- Single node setup
- Don't need MODEL elsewhere

**Pros:** Fastest setup, no extra nodes

### Use MODEL Auto-Extract When:
- Complex workflow
- Need MODEL for other nodes (Apply LoRA, etc.)

**Setup:**
```
[CheckpointLoaderSimple]
   ├─ model ───→ (to other nodes)
   └─ clip ──────────────┐
                         │
[PM LoRA Semantic Analyzer (Gradient)]
   ├─ checkpoint_name: select from dropdown
   └─ clip: ←───────────┘
```

## Common Settings

### Features
```
# Predefined features (optimized depth curves):
"hair,eyes,face,clothing,accessories,body"

# Custom features (use default depth):
"uniform,weapon,tattoo,jewelry"

# Mix both:
"hair,eyes,uniform,weapon"
```

### Architecture Hint
```
"auto"    ← Recommended (auto-detects)
"sd15"    ← Force SD1.5
"sdxl"    ← Force SDXL
"flux"    ← Force Flux
"zimage"  ← Force zImage (currently only fully supported)
```

### Device & Dtype
```
device: "auto"     ← Uses CUDA if available
dtype:  "auto"     ← Uses model's native dtype

# Or specify:
device: "cuda"
dtype:  "float16"  ← Faster, uses less VRAM
```

### Num Samples
```
1  ← Fastest, less stable
3  ← Recommended balance
5  ← More stable, slower
10 ← Best quality, slowest
```

## Troubleshooting

### ❌ Error: "checkpoint_path is required"

**Cause:** No checkpoint selected in dropdown

**Fix:**
1. Select a checkpoint from the `checkpoint_name` dropdown

### ❌ Dropdown is empty

**Cause:** No checkpoints in standard folders

**Fix:**
1. Move your checkpoint to:
   - `ComfyUI/models/checkpoints/`, OR
   - `ComfyUI/models/diffusion_models/`
2. Refresh ComfyUI (Ctrl+R)

### ❌ Error: "Could not detect model type"

**Cause:** Selected file is not a valid checkpoint/diffusion model

**Fix:**
- Make sure the file is a `.safetensors`, `.ckpt`, or `.pt` model file
- Verify the file isn't corrupted

### ❌ Analysis is very slow

**Causes:**
1. High `num_samples` setting
2. CPU instead of CUDA
3. Large checkpoint

**Fixes:**
- Reduce `num_samples` to 1-3
- Set `device` to "cuda" if you have GPU
- Use `use_cache` (enabled by default) - second run will be instant!

## Performance Tips

### First Run (Uncached)
```
zImage (1.5B params)
├─ num_samples: 1  → ~10 seconds
├─ num_samples: 3  → ~30 seconds
└─ num_samples: 10 → ~90 seconds
```

### Second Run (Cached)
```
Any architecture
└─ num_samples: any → <1 second (loads from cache)
```

**Cache location:** `cache/semantic_maps/`

### Speed Up Analysis

1. **Enable caching** (default: ON)
   - First analysis: slow
   - Subsequent: instant

2. **Reduce num_samples**
   - `num_samples: 1` is often good enough
   - Increase only if results are unstable

3. **Use GPU**
   - `device: "cuda"` (if available)
   - `dtype: "float16"` for faster processing

4. **Clear cache if needed**
   ```bash
   rm -rf cache/semantic_maps/*
   ```

## What Gets Cached

Cached when:
- Same LoRA (file path + tensor checksums match)
- Same features (e.g., "hair,eyes,clothing")
- Same analysis method (heuristic vs gradient)
- Same architecture hint

Cache invalidated when:
- LoRA file changes
- Different features requested
- Different num_samples (for gradient method)

## Tips & Tricks

### Tip 1: Test with Heuristic First
```
Use "PM LoRA Semantic Analyzer (Heuristic)" first:
├─ Very fast (no checkpoint needed!)
├─ Good baseline results
└─ If results look good, try gradient for refinement
```

### Tip 2: Start with Common Features
```
Begin with: "hair,eyes,clothing"
├─ Most tested features
├─ Clear visual results
└─ Add more features as needed
```

### Tip 3: Use Descriptive Feature Names
```
Good: "uniform,hat,weapon"
Bad:  "style1,thing1,other"
     └─ Generic names get generic depth curves
```

### Tip 4: Cache Your Favorite LoRAs
```
Run analysis once on your common LoRAs:
├─ Takes 10-30 seconds each
├─ Cached forever
└─ Future merges are instant!
```

## Next Steps

After getting semantic maps:

1. **Create Merge Spec**
   - Use `PM Semantic Merge Spec` node
   - Specify: "hair from lora1, clothing from lora2"

2. **Merge LoRAs**
   - Use `PM Semantic Merger` node
   - Outputs merged LoRABundle

3. **Use Merged LoRA**
   - Apply with `PM LoRA Apply`
   - Or save with `PM LoRA Save`

## Full Documentation

- **Dropdown feature:** `DROPDOWN_CHECKPOINT_SELECTOR.md`
- **Custom loaders:** `CHECKPOINT_LOADER_WITH_PATH.md`
- **Semantic merging:** `SEMANTIC_MERGE.md`
- **Quick tutorial:** `SEMANTIC_MERGE_QUICKSTART.md`

## Questions?

Check the documentation files or open an issue on GitHub!
