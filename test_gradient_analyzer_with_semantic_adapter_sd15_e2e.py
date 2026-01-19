#!/usr/bin/env python3
"""
End-to-End Test for Semantic Adapters with Gradient Analysis - SD1.5 Version.

This test replicates the workflow from test_gradient_analyzer_sd15_e2e.py but TRAINS
semantic adapters instead of merging LoRAs, then uses the adapters for inference:

1. Load Checkpoint (SD1.5 model with UNet, CLIP, VAE)
2. Stack LoRAs
3. Run gradient-based semantic analysis
4. Create merge specification
5. TRAIN SEMANTIC ADAPTERS (replaces merge step)
6. Apply adapters to model using hooks
7. Generate image using KSampler with adapters active
8. Save output image

This validates the complete semantic adapter workflow: analysis → training → inference.
"""

import sys
import os
import time
import logging
from pathlib import Path
from typing import Tuple, Dict, Any

# Configure basic info-level logging for the test run
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("pm_semantic_adapter_sd15_e2e")

# Set PyTorch memory management for better VRAM handling
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Add ComfyUI to path
COMFYUI_PATH = Path("/home/lars/SD/Apps/ComfyUI")
sys.path.insert(0, str(COMFYUI_PATH))

# Add custom nodes to path
CUSTOM_NODES_PATH = COMFYUI_PATH / "custom_nodes" / "LoRA-Merger-ComfyUI"
sys.path.insert(0, str(CUSTOM_NODES_PATH))

print("=" * 80)
print("SEMANTIC ADAPTERS WITH GRADIENT ANALYSIS - SD1.5 END-TO-END TEST")
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

from src.lora_power_stacker import LoraPowerStacker
from src.lora_apply import LoraApply
from src.nodes_semantic_analysis import PMLoRASemanticAnalyzerGradient
from src.nodes_semantic_merge import PMSemanticMergeSpec

# Import semantic adapter modules
from src.semantic.adapter.registry import AdapterRegistry
from src.semantic.training.trainer import AdapterTrainer
from src.semantic.training.data_loader import HybridSemanticDataLoader
from src.semantic.inference.pipeline import AdapterInferencePipeline
from src.semantic.semantic_merger import SemanticMerger, MergeSpec

# Test configuration
CONFIG = {
    # Checkpoint
    "checkpoint": "SD1.5/analogMadness_v70.safetensors",

    # LoRAs
    "loras": [
        {"on": True, "lora": "SD1.5/Characters/Sarah F/SarahF-LoRA-SD15-v52-AM-D128-A64.safetensors", "strength": 1.0},
        {"on": True, "lora": "SD1.5/Characters/Tessa Fowler/TessaF-LoRA-SD15-v4-AM-D128-A64.safetensors", "strength": 1.0},
    ],
    "layer_filter": "attn-mlp",

    # Gradient analysis
    "checkpoint_name": "checkpoints/SD1.5/analogMadness_v70.safetensors",
    "features": "hair,face,breasts",
    "architecture_hint": "sd15",
    "num_samples": 1,
    "use_cache": False,
    "device": "auto",
    "dtype": "float16",
    "offload_layers": False,
    "gpu_memory_gb": 0.0,

    # Merge spec (used for adapter training)
    "merge_spec": "face from SarahF-LoRA-SD15-v52-AM-D128-A64 2 exclusive, hair from SarahF-LoRA-SD15-v52-AM-D128-A64 exclusive, breasts from TessaF-LoRA-SD15-v4-AM-D128-A64 2 exclusive",

    # Adapter training
    "num_training_epochs": 10,  # Quick training for testing (normally 150)
    "use_hooks_for_training": False,  # Set to True to use hook-based training
    "use_real_unet_deltas": True,  # PHASE 3: Adapter fix applied (OutputSpaceAdapter for hooked Linear layers)

    # Prompts
    "positive_prompt": """A photorealistic portrait of a woman, nude, standing, professional photography,
high detail, sharp focus, natural lighting, 8k uhd, masterpiece, best quality""",
    "negative_prompt": "text, watermark",

    # Image generation
    "width": 512,
    "height": 768,

    # Sampling
    "seed": 214379613299061,
    "steps": 20,
    "cfg": 5.0,
    "sampler_name": "euler_ancestral",
    "scheduler": "karras",
    "denoise": 1.0,

    # Output
    "output_image": "test_semantic_adapter_image.png",
}


def test_step(step_num: int, total_steps: int, description: str):
    """Print test step header."""
    print(f"\n{'=' * 80}")
    print(f"STEP {step_num}/{total_steps}: {description}")
    print(f"{'=' * 80}")


def main():
    """Run complete end-to-end test with semantic adapters."""
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

        # Store original model for later restoration
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if CONFIG["dtype"] == "float16" else torch.float32

        # Step 2: Stack LoRAs
        current_step += 1
        test_step(current_step, total_steps, "Stack LoRAs")

        stacker = LoraPowerStacker()

        # Format LoRAs as keyword arguments
        lora_kwargs = {
            "model": model,
            "clip": clip,
            "layer_filter": CONFIG["layer_filter"],
        }

        # Unload model from GPU to save memory before gradient analysis
        import comfy.model_management
        model.unpatch_model(device_to=torch.device('cpu'))
        comfy.model_management.soft_empty_cache()

        for idx, lora_config in enumerate(CONFIG["loras"], start=1):
            lora_kwargs[f"LORA_{idx}"] = lora_config

        lora_stack, lora_weights = stacker.stack_loras_widget(**lora_kwargs)

        print(f"✓ Stacked {len(CONFIG['loras'])} LoRAs")
        print(f"  LoRA stack keys: {len(lora_stack)} LoRAs")
        for lora_name in lora_stack.keys():
            num_layers = len(lora_stack[lora_name]["patches"]) if isinstance(lora_stack[lora_name], dict) else len(lora_stack[lora_name])
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
        print(f"  Dtype: {CONFIG['dtype']}")
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
        )
        analysis_time = time.time() - analysis_start

        print(f"✓ Gradient analysis complete in {analysis_time:.1f}s")
        print(f"  Analysis info:\n{analysis_info}")
        print(f"  Semantic maps: {len(semantic_maps)} LoRAs")

        # Extract feature names from semantic maps
        feature_names = []
        if semantic_maps:
            first_lora = list(semantic_maps.values())[0]
            feature_names = list(first_lora.keys())
            print(f"  Features detected: {feature_names}")

        # Step 4: Create Merge Specification
        current_step += 1
        test_step(current_step, total_steps, "Create Merge Specification")

        spec_node = PMSemanticMergeSpec()
        _, parsed_spec = spec_node.create_spec(CONFIG["merge_spec"])
        print(f"✓ Merge specification created")
        print(f"  Specification: '{CONFIG['merge_spec']}'")
        print(f"  Parsed:\n{parsed_spec}")

        # Parse specification for adapter training
        merge_spec = MergeSpec.from_text(CONFIG["merge_spec"])

        # Clear VRAM before Phase 3 (gradient analysis loaded pipelines 3 times)
        print(f"\n  Clearing VRAM before adapter training...")
        import gc
        torch.cuda.empty_cache()
        gc.collect()
        print(f"✓ VRAM cleared")

        # Step 5: Train Semantic Adapters (replaces merge step)
        current_step += 1
        test_step(current_step, total_steps, "Train Semantic Adapters")

        # Create adapter registry
        print(f"  Creating adapter registry...")
        registry = AdapterRegistry(
            lora_stack=lora_stack,
            semantic_maps=semantic_maps,
            feature_names=feature_names,
            device=device,
            dtype=dtype,
        )

        all_adapters = registry.get_all_adapters()
        print(f"✓ Adapter registry created: {len(all_adapters)} adapters")

        # Count adapter types
        from src.semantic.adapter.rank_space import RankSpaceAdapter
        from src.semantic.adapter.output_space import OutputSpaceAdapter
        rank_space = sum(1 for a in all_adapters.values() if isinstance(a, RankSpaceAdapter))
        output_space = sum(1 for a in all_adapters.values() if isinstance(a, OutputSpaceAdapter))
        print(f"  - {rank_space} rank-space adapters (attention/MLP)")
        print(f"  - {output_space} output-space adapters (conv)")

        # Create semantic merger for training targets
        semantic_merger = SemanticMerger(device=device, dtype=dtype)

        # Create delta provider for Phase 3 (if enabled)
        delta_provider = None
        if CONFIG["use_real_unet_deltas"]:
            print(f"\n  Creating UNetDeltaProvider (Phase 3)...")
            from src.semantic.training.delta_provider import create_delta_provider

            # Construct full checkpoint path
            checkpoint_full_path = str(COMFYUI_PATH / "models" / CONFIG["checkpoint_name"])
            print(f"  Checkpoint path: {checkpoint_full_path}")

            delta_provider = create_delta_provider(
                provider_type="unet",
                adapter_registry=registry,
                device=device,
                dtype=dtype,
                checkpoint_path=checkpoint_full_path,
            )
            print(f"✓ UNetDeltaProvider created")
            print(f"  LoRAs loaded: {len(delta_provider.lora_names)}")
            print(f"  Hooks registered: {len(delta_provider.hooks)}")
        else:
            print(f"\n  Using SyntheticDeltaProvider (default)")

        # Create trainer
        print(f"\n  Creating adapter trainer...")
        print(f"  Training epochs: {CONFIG['num_training_epochs']}")
        print(f"  Use hooks: {CONFIG['use_hooks_for_training']}")

        trainer = AdapterTrainer(
            adapter_registry=registry,
            base_model=model.model,  # Extract PyTorch model from ComfyUI wrapper
            lora_stack=lora_stack,
            semantic_merger=semantic_merger,
            feature_prompts=feature_names,  # Use feature names as prompts
            device=device,
            dtype=dtype,
            use_hooks=CONFIG["use_hooks_for_training"],
            delta_provider=delta_provider,  # Pass UNetDeltaProvider if Phase 3
        )

        # Create real dataloader for training
        print(f"\n  Creating data loader...")
        architecture_hint = CONFIG.get("architecture_hint", "sd15")
        dataloader = HybridSemanticDataLoader(
            feature_names=feature_names,
            architecture=architecture_hint,
            prompts_per_feature=3,  # 3 prompts per feature
            batch_size=3,  # Reduced from 12 to 3 for Phase 3 VRAM savings (1 per feature)
            num_batches=10,  # 10 batches per epoch
            latent_height=64,  # 512 / 8
            latent_width=96,  # 768 / 8
            device=device,
            dtype=dtype,
            seed=42,
        )
        print(f"✓ Data loader created: {len(dataloader)} batches")

        # Train adapters
        print(f"\n  Training adapters...")
        training_start = time.time()
        training_stats = trainer.train(
            dataloader=dataloader,
            num_epochs=CONFIG["num_training_epochs"],
        )
        training_time = time.time() - training_start

        print(f"✓ Adapter training complete in {training_time:.1f}s")
        print(f"  Epochs: {CONFIG['num_training_epochs']}")
        print(f"  Final losses:")
        if training_stats["losses"]:
            final_loss = training_stats["losses"][-1]
            print(f"    Total: {final_loss['total']:.6f}")
            print(f"    Teacher: {final_loss['teacher']:.6f}")
            print(f"    Dominance: {final_loss['dominance']:.6f}")
            print(f"    Residual: {final_loss['residual']:.6f}")

        # Step 6: Setup Adapter Inference Pipeline
        current_step += 1
        test_step(current_step, total_steps, "Setup Adapter Inference Pipeline")

        # Create inference pipeline
        inference_pipeline = AdapterInferencePipeline(
            base_model=model.model,
            adapter_registry=registry,
            lora_stack=lora_stack,
            device=device,
            dtype=dtype,
        )

        print(f"✓ Inference pipeline created")
        print(f"  Adapters: {len(registry)}")
        print(f"  Device: {device}")
        print(f"  Dtype: {dtype}")

        # Create semantic vector from merge specification
        # Map features to their importance based on merge spec
        semantic_vector_dict = {}
        for feature_name in feature_names:
            # Check if feature is in merge spec
            if feature_name in merge_spec.feature_assignments:
                assignment = merge_spec.feature_assignments[feature_name]
                weight = assignment.get("weight", 1.0)
                semantic_vector_dict[feature_name] = weight
            else:
                semantic_vector_dict[feature_name] = 0.5  # Default weight

        # Convert to tensor [batch_size, num_features]
        batch_size = 1
        semantic_vector = torch.tensor(
            [[semantic_vector_dict.get(f, 0.5) for f in feature_names]],
            device=device,
            dtype=dtype
        )
        print(f"  Semantic vector: {semantic_vector.tolist()}")

        # Step 7: Encode Prompts
        current_step += 1
        test_step(current_step, total_steps, "Encode Prompts")

        clip_text_encode = nodes.CLIPTextEncode()
        (positive_cond,) = clip_text_encode.encode(clip, CONFIG["positive_prompt"])
        (negative_cond,) = clip_text_encode.encode(clip, CONFIG["negative_prompt"])
        print(f"✓ Prompts encoded")
        print(f"  Positive: {len(CONFIG['positive_prompt'])} chars")
        print(f"  Negative: {len(CONFIG['negative_prompt'])} chars")

        # Step 8: Generate Image with Adapters Active
        current_step += 1
        test_step(current_step, total_steps, "Generate Image with Semantic Adapters")

        # Create empty latent
        empty_latent = nodes.EmptyLatentImage()
        (latent,) = empty_latent.generate(
            width=CONFIG["width"],
            height=CONFIG["height"],
            batch_size=1
        )
        print(f"✓ Empty latent created: {CONFIG['width']}x{CONFIG['height']}")

        # Setup hooks for adapter injection
        print(f"\n  Setting up adapter hooks...")
        inference_pipeline.setup_hooks()
        inference_pipeline.set_semantic_vector(semantic_vector)
        print(f"✓ Adapter hooks registered")

        # Run KSampler with adapters active
        print(f"\n  Running KSampler with adapters...")
        print(f"    Seed: randomize (fixed in config: {CONFIG['seed']})")
        print(f"    Steps: {CONFIG['steps']}")
        print(f"    CFG: {CONFIG['cfg']}")
        print(f"    Sampler: {CONFIG['sampler_name']}")
        print(f"    Scheduler: {CONFIG['scheduler']}")
        print()

        ksampler = nodes.KSampler()
        sampling_start = time.time()

        try:
            # Sample with adapters active
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

        finally:
            # Always remove hooks
            inference_pipeline.remove_hooks()
            print(f"✓ Adapter hooks removed")

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
        print(f"Adapter training:")
        print(f"  Epochs: {CONFIG['num_training_epochs']}")
        print(f"  Adapters: {len(all_adapters)} ({rank_space} rank-space, {output_space} output-space)")
        print(f"  Training time: {training_time:.1f}s")
        print()
        print(f"Timing breakdown:")
        print(f"  Gradient analysis: {analysis_time:.1f}s")
        print(f"  Adapter training: {training_time:.1f}s")
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
        print("🎉 SEMANTIC ADAPTER END-TO-END TEST PASSED!")

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
