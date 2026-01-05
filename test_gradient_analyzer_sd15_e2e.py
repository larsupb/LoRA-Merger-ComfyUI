#!/usr/bin/env python3
"""
End-to-End Test for PM LoRA Semantic Analyzer (Gradient) - SD1.5 Version.

This test replicates the COMPLETE workflow from PM-Gradient-Analyzer-SD1.5.json:
1. Load Checkpoint (SD1.5 model with UNet, CLIP, VAE)
2. Stack LoRAs
3. Run gradient-based semantic analysis
4. Create merge specification
5. Merge LoRAs using semantic maps
6. Apply merged LoRA to model
7. Generate image using KSampler
8. Save output image
"""

import sys
import os
import time
from pathlib import Path
from typing import Tuple, Dict, Any

# Set PyTorch memory management for better VRAM handling
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Add ComfyUI to path
COMFYUI_PATH = Path("/home/lars/SD/Apps/ComfyUI")
sys.path.insert(0, str(COMFYUI_PATH))

# Add custom nodes to path
CUSTOM_NODES_PATH = COMFYUI_PATH / "custom_nodes" / "LoRA-Merger-ComfyUI"
sys.path.insert(0, str(CUSTOM_NODES_PATH))

print("=" * 80)
print("PM LoRA Semantic Analyzer (Gradient) - SD1.5 END-TO-END TEST")
print("=" * 80)
print(f"ComfyUI path: {COMFYUI_PATH}")
print(f"Custom nodes path: {CUSTOM_NODES_PATH}")
print()

# Import ComfyUI modules
try:
    import torch
    import folder_paths
    import comfy.sd
    import comfy.samplers
    import comfy.sample
    import nodes

    print("✓ Imported ComfyUI core modules")
except ImportError as e:
    print(f"✗ Failed to import ComfyUI modules: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Import custom nodes
try:
    from src.lora_power_stacker import LoraPowerStacker
    from src.lora_apply import LoraApply
    from src.nodes_semantic_analysis import PMLoRASemanticAnalyzerGradient
    from src.nodes_semantic_merge import PMSemanticMergeSpec, PMSemanticMerger

    print("✓ Imported custom nodes")
except ImportError as e:
    print(f"✗ Failed to import custom nodes: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test configuration from workflow (PM-Gradient-Analyzer-SD1.5.json)
CONFIG = {
    # Checkpoint (from node 135)
    "checkpoint": "SD1.5/analogMadness_v70.safetensors",

    # LoRAs (from node 105)
    "loras": [
        {"on": True, "lora": "SD1.5/Characters/Sarah F/SarahF-LoRA-SD15-v52-AM-D128-A64.safetensors", "strength": 1.0},
        {"on": True, "lora": "SD1.5/Characters/Tessa Fowler/TessaF-LoRA-SD15-v4-AM-D128-A64.safetensors", "strength": 1.0},
    ],
    "layer_filter": "attn-mlp",

    # Gradient analysis (from node 134)
    "checkpoint_name": "checkpoints/SD1.5/analogMadness_v70.safetensors",
    "features": "hair,earrings,face,breasts",
    "architecture_hint": "sd15",
    "num_samples": 10,  # From workflow
    "use_cache": False,  # From workflow
    "device": "auto",
    "dtype": "float16",
    "offload_layers": False,
    "gpu_memory_gb": 0.0,  # 0 = auto-detect 80% of VRAM

    # Merge spec (from node 113)
    "merge_spec": "face from TessaF-LoRA-SD15-v4-AM-D128-A64, hair from TessaF-LoRA-SD15-v4-AM-D128-A64, earrings from TessaF-LoRA-SD15-v4-AM-D128-A64, breasts from SarahF-LoRA-SD15-v52-AM-D128-A64",

    # Prompts (from nodes 6 and 7)
    # Note: Workflow has empty positive prompt, using generic SD1.5 prompt instead
    "positive_prompt": """A photorealistic portrait of a woman, nude, standing, breasts, professional photography,
high detail, sharp focus, natural lighting, 8k uhd, masterpiece, best quality""",
    "negative_prompt": "text, watermark",

    # Image generation (from node 24 - PixelResolutionCalculator)
    # Megapixels=1, aspect_ratio=2:3, divided_by=8 results in 1024x1024
    "width": 512,
    "height": 768,

    # Sampling (from node 117)
    "seed": 214379613299061,  # From workflow (will be randomized in test)
    "steps": 20,
    "cfg": 5.0,
    "sampler_name": "euler_ancestral",
    "scheduler": "karras",
    "denoise": 1.0,

    # Output
    "output_image": "test_output_gradient_analyzer_sd15.png",
}


def test_step(step_num: int, total_steps: int, description: str):
    """Print test step header."""
    print(f"\n{'=' * 80}")
    print(f"STEP {step_num}/{total_steps}: {description}")
    print(f"{'=' * 80}")


def main():
    """Run complete end-to-end test."""
    total_steps = 9
    current_step = 0
    start_time = time.time()

    try:
        # Step 1: Load Checkpoint (SD1.5 model with UNet, CLIP, VAE)
        current_step += 1
        test_step(current_step, total_steps, "Load SD1.5 Checkpoint")

        checkpoint_loader = nodes.CheckpointLoaderSimple()
        model, clip, vae = checkpoint_loader.load_checkpoint(CONFIG["checkpoint"])
        print(f"✓ SD1.5 checkpoint loaded: {CONFIG['checkpoint']}")
        print(f"  Includes: UNet (~2.6B params), CLIP, VAE")

        # Step 2: Stack LoRAs
        current_step += 1
        test_step(current_step, total_steps, "Stack LoRAs")

        stacker = LoraPowerStacker()

        # Format LoRAs as keyword arguments (node expects LORA_1, LORA_2, etc.)
        lora_kwargs = {
            "model": model,
            "clip": clip,
            "layer_filter": CONFIG["layer_filter"],
        }

        # Unload model from GPU to save memory before gradient analysis
        # Use ModelPatcher's unpatch_model to move model to CPU
        import comfy.model_management
        model.unpatch_model(device_to=torch.device('cpu'))
        comfy.model_management.soft_empty_cache()

        for idx, lora_config in enumerate(CONFIG["loras"], start=1):
            lora_kwargs[f"LORA_{idx}"] = lora_config

        lora_stack, lora_weights = stacker.stack_loras_widget(**lora_kwargs)

        print(f"✓ Stacked {len(CONFIG['loras'])} LoRAs")
        print(f"  LoRA stack keys: {len(lora_stack)} LoRAs")
        for lora_name in lora_stack.keys():
            num_layers = len(lora_stack[lora_name])
            print(f"    - {lora_name}: {num_layers} layers")

        # Step 3: Run Gradient Analysis
        current_step += 1
        test_step(current_step, total_steps, "Gradient-Based Semantic Analysis (SD1.5)")

        analyzer = PMLoRASemanticAnalyzerGradient()
        print(f"  Checkpoint: {CONFIG['checkpoint_name']}")
        print(f"  Features: {CONFIG['features']}")
        print(f"  Architecture: {CONFIG['architecture_hint']}")
        print(f"  Num samples: {CONFIG['num_samples']}")
        print(f"  Use cache: {CONFIG['use_cache']}")
        print(f"  Device: {CONFIG['device']}")
        print(f"  Dtype: {CONFIG['dtype']} (FP8 for memory efficiency)")
        print(f"  Layer offloading: {CONFIG['offload_layers']}")
        print(f"  GPU memory limit: {CONFIG['gpu_memory_gb']} GB (0=auto-detect)")
        print()
        print("  ⚠ This may take 2-10 minutes on first run (downloads SD1.5 config + analysis)")
        print("  ⚠ Subsequent runs will be much faster if cache is enabled")
        print("  ⚠ With 8GB VRAM, offloading is enabled for better memory management")
        print()

        analysis_start = time.time()
        semantic_maps, analysis_info = analyzer.analyze(
            clip=clip,
            lora_stack=lora_stack,
            checkpoint_name=CONFIG["checkpoint_name"],
            features=CONFIG["features"],
            use_cache=CONFIG["use_cache"],
            architecture_hint=CONFIG["architecture_hint"],
            device=CONFIG["device"],
            dtype=CONFIG["dtype"],
            num_samples=CONFIG["num_samples"],
            offload_layers=CONFIG["offload_layers"],
            gpu_memory_gb=CONFIG["gpu_memory_gb"],
        )
        analysis_time = time.time() - analysis_start

        print(f"✓ Gradient analysis complete in {analysis_time:.1f}s")
        print(f"  Analysis info:\n{analysis_info}")
        print(f"  Semantic maps: {len(semantic_maps)} LoRAs")

        # Step 4: Create Merge Specification
        current_step += 1
        test_step(current_step, total_steps, "Create Merge Specification")

        spec_node = PMSemanticMergeSpec()
        merge_spec, parsed_spec = spec_node.create_spec(CONFIG["merge_spec"])
        print(f"✓ Merge specification created")
        print(f"  Specification: '{CONFIG['merge_spec']}'")
        print(f"  Parsed:\n{parsed_spec}")

        # Step 5: Merge LoRAs
        current_step += 1
        test_step(current_step, total_steps, "Semantic LoRA Merging")

        merger = PMSemanticMerger()
        (merged_lora,) = merger.merge_semantic(
            model=model,
            lora_stack=lora_stack,
            semantic_maps=semantic_maps,
            merge_spec=merge_spec,
            lambda_value=1.0,
        )
        print(f"✓ LoRAs merged semantically")
        print(f"  Merged LoRA name: {merged_lora.get('name', 'unknown')}")
        print(f"  Merged layers: {len(merged_lora.get('lora', {}))}")

        # Step 6: Apply Merged LoRA
        current_step += 1
        test_step(current_step, total_steps, "Apply Merged LoRA to Model")

        lora_apply = LoraApply()
        model, clip = lora_apply.apply_merged_lora(model, clip, merged_lora)
        print(f"✓ Merged LoRA applied to model")

        # Step 7: Encode Prompts
        current_step += 1
        test_step(current_step, total_steps, "Encode Prompts")

        clip_text_encode = nodes.CLIPTextEncode()
        (positive_cond,) = clip_text_encode.encode(clip, CONFIG["positive_prompt"])
        (negative_cond,) = clip_text_encode.encode(clip, CONFIG["negative_prompt"])
        print(f"✓ Prompts encoded")
        print(f"  Positive: {len(CONFIG['positive_prompt'])} chars")
        print(f"  Negative: {len(CONFIG['negative_prompt'])} chars")

        # Step 8: Generate Image with KSampler
        current_step += 1
        test_step(current_step, total_steps, "Generate Image (KSampler)")

        # Create empty latent
        empty_latent = nodes.EmptyLatentImage()
        (latent,) = empty_latent.generate(
            width=CONFIG["width"],
            height=CONFIG["height"],
            batch_size=1
        )
        print(f"✓ Empty latent created: {CONFIG['width']}x{CONFIG['height']}")

        # Run KSampler
        print(f"  Running KSampler...")
        print(f"    Seed: randomize (fixed in workflow: {CONFIG['seed']})")
        print(f"    Steps: {CONFIG['steps']}")
        print(f"    CFG: {CONFIG['cfg']}")
        print(f"    Sampler: {CONFIG['sampler_name']}")
        print(f"    Scheduler: {CONFIG['scheduler']}")
        print()

        ksampler = nodes.KSampler()
        sampling_start = time.time()
        (sampled_latent,) = ksampler.sample(
            model=model,
            seed=CONFIG["seed"],
            steps=CONFIG["steps"],
            cfg=CONFIG["cfg"],
            sampler_name=CONFIG["sampler_name"],
            scheduler=CONFIG["scheduler"],
            positive=positive_cond,
            negative=negative_cond,
            latent_image=latent,
            denoise=CONFIG["denoise"],
        )
        sampling_time = time.time() - sampling_start
        print(f"✓ Sampling complete in {sampling_time:.1f}s")

        # Step 9: Decode and Save Image
        current_step += 1
        test_step(current_step, total_steps, "Decode and Save Image")

        # Decode latent
        vae_decode = nodes.VAEDecode()
        (image,) = vae_decode.decode(vae, sampled_latent)
        print(f"✓ Latent decoded to image")

        # Save image
        save_image = nodes.SaveImage()
        results = save_image.save_images(
            images=image,
            filename_prefix=CONFIG["output_image"].replace(".png", "")
        )

        # Find saved image path
        output_dir = folder_paths.get_output_directory()
        saved_files = results.get("ui", {}).get("images", [])
        if saved_files:
            saved_path = Path(output_dir) / saved_files[0]["subfolder"] / saved_files[0]["filename"]
            print(f"✓ Image saved: {saved_path}")
        else:
            print(f"✓ Image saved to output directory")

        # Print summary
        total_time = time.time() - start_time
        print(f"\n{'=' * 80}")
        print("TEST SUMMARY")
        print(f"{'=' * 80}")
        print(f"✓ All {total_steps} steps completed successfully!")
        print()
        print(f"Architecture: SD1.5 (UNet-based, 2.6B params)")
        print(f"Checkpoint: {CONFIG['checkpoint']}")
        print(f"LoRAs: {len(CONFIG['loras'])}")
        print(f"Features analyzed: {CONFIG['features']}")
        print()
        print(f"Timing breakdown:")
        print(f"  Gradient analysis: {analysis_time:.1f}s")
        print(f"  Image sampling: {sampling_time:.1f}s")
        print(f"  Total time: {total_time:.1f}s")
        print()
        print(f"Output:")
        if saved_files:
            print(f"  Image: {saved_path}")
        print(f"  Size: {CONFIG['width']}x{CONFIG['height']}")
        print(f"  Steps: {CONFIG['steps']}")
        print(f"  Sampler: {CONFIG['sampler_name']} ({CONFIG['scheduler']})")
        print()
        print("🎉 SD1.5 END-TO-END TEST PASSED!")

        return 0

    except KeyboardInterrupt:
        print("\n\n⚠ Test interrupted by user")
        return 130

    except Exception as e:
        print(f"\n\n✗ TEST FAILED at step {current_step}/{total_steps}")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except Exception as e:
        print(f"\n\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
