"""
Load zImage model for gradient analysis using diffusers.

This module loads zImage transformer architecture from HuggingFace
(config only) and initializes with custom .safetensors weights.

Uses automatic CPU offloading for memory efficiency.
"""

import torch
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


def load_zimage_for_gradients(
    checkpoint_path: str,
    device: torch.device,
    dtype: torch.dtype,
    offload_layers: bool = False,
    gpu_memory_gb: Optional[float] = None,
) -> torch.nn.Module:
    """
    Load zImage transformer with custom weights.

    **Optimized approach:**
    1. Load zImage config from HuggingFace (config only, ~few KB)
    2. Initialize model architecture with random weights
    3. Load actual weights from checkpoint_path
    4. Enable automatic CPU offloading if requested
    5. Enable gradients

    This downloads only the config (~few KB) instead of full model weights (~2GB).

    Args:
        checkpoint_path: Path to .safetensors file containing model weights
        device: cuda/cpu
        dtype: torch.float32, torch.float16, torch.bfloat16, torch.float8_e4m3fn, or torch.float8_e5m2
        offload_layers: If True, enable automatic CPU offloading to reduce VRAM usage
        gpu_memory_gb: Ignored (kept for API compatibility). CPU offloading is automatic.

    Returns:
        zImage transformer ready for gradient computation

    Raises:
        ImportError: If diffusers is not installed
        FileNotFoundError: If checkpoint_path doesn't exist
        RuntimeError: If model loading fails
    """
    try:
        from diffusers.models import ZImageTransformer2DModel
    except ImportError as e:
        raise ImportError(
            "diffusers library required for gradient analysis. "
            "Install with: pip install diffusers>=0.30.0"
        ) from e

    import safetensors.torch
    import os

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    logger.info("Loading zImage architecture from HuggingFace...")
    logger.info("(First run will download config only, ~few KB, then cached)")

    try:
        # Check if dtype supports gradients
        # fp8 doesn't support CUDA gradient operations
        gradient_compatible_dtype = dtype
        if dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
            logger.warning(
                f"⚠️  fp8 dtype ({dtype}) doesn't support gradient computation!\n"
                f"   Converting to fp16 for gradient analysis..."
            )
            gradient_compatible_dtype = torch.float16

        # Load ONLY the config (no weights downloaded!)
        config = ZImageTransformer2DModel.load_config(
            "Tongyi-MAI/Z-Image-Turbo",
            subfolder="transformer",
        )

        # Initialize model in the CORRECT dtype to avoid FP32 bloat
        logger.info(f"Initializing model architecture in {gradient_compatible_dtype} on CPU...")

        with torch.device("cpu"):
            transformer = ZImageTransformer2DModel.from_config(config, torch_dtype=gradient_compatible_dtype)

        logger.info(f"Loading custom weights from {checkpoint_path}...")

        # Load weights to CPU
        custom_weights = safetensors.torch.load_file(
            checkpoint_path,
            device="cpu"
        )

        # Preprocess keys (remove ComfyUI prefixes)
        processed_weights = _preprocess_state_dict_keys(custom_weights)

        # Load weights into transformer
        missing_keys, unexpected_keys = transformer.load_state_dict(processed_weights, strict=False)

        # Free checkpoint memory
        del custom_weights, processed_weights
        import gc
        gc.collect()

        if missing_keys:
            logger.warning(f"Missing keys when loading weights: {len(missing_keys)} keys")
            logger.debug(f"Missing keys: {missing_keys[:10]}...")

        if unexpected_keys:
            logger.info(f"Unexpected keys in checkpoint: {len(unexpected_keys)} keys (normal for ComfyUI checkpoints)")

        # Setup device placement with automatic CPU offloading
        if offload_layers:
            logger.info("Setting up automatic CPU offloading...")
            try:
                from accelerate import cpu_offload
                # Use accelerate's automatic CPU offloading
                # This is simpler than manual device_map and works well for zImage
                cpu_offload(transformer, execution_device=device)
                logger.info("✓ CPU offloading enabled")
            except Exception as e:
                logger.warning(f"CPU offloading failed: {e}")
                logger.warning("Falling back to full GPU loading...")
                transformer.to(device)
        else:
            logger.info(f"Moving transformer to {device}...")
            transformer.to(device)
            logger.info(f"✓ Transformer on {device}")

        # Enable gradient computation
        logger.info("Enabling gradients...")
        transformer.train()
        transformer.requires_grad_(True)

        logger.info("✓ zImage model loaded successfully")
        logger.info(f"  Device: {device}, dtype: {dtype}")
        logger.info(f"  CPU offloading: {'enabled' if offload_layers else 'disabled'}")
        logger.info(f"  Parameters: {sum(p.numel() for p in transformer.parameters()):,}")
        logger.info(f"  Gradient enabled: {next(transformer.parameters()).requires_grad}")

        return transformer

    except Exception as e:
        raise RuntimeError(f"Failed to load zImage model: {e}") from e


def _preprocess_state_dict_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Remove ComfyUI prefixes from state dict keys.

    ComfyUI checkpoints may have prefixes like:
    - "model."
    - "diffusion_model."
    - "model.diffusion_model."

    Args:
        state_dict: Raw state dict from checkpoint

    Returns:
        State dict with cleaned keys
    """
    processed = {}

    for key, value in state_dict.items():
        clean_key = key

        # Remove common prefixes
        for prefix in ["model.diffusion_model.", "diffusion_model.", "model."]:
            if clean_key.startswith(prefix):
                clean_key = clean_key[len(prefix):]
                break

        processed[clean_key] = value

    return processed


def apply_lora_simple(
    model: torch.nn.Module,
    lora_patches: Dict[str, Any],
    checkpoint_path: Optional[str] = None,
) -> torch.nn.Module:
    """
    Apply LoRA patches to zImage model.

    Merges LoRA delta directly into model weights using simple key matching.

    Args:
        model: zImage transformer model
        lora_patches: ComfyUI LoRA patches {layer_key: LoRAAdapter}
        checkpoint_path: Optional, not used (kept for API compatibility)

    Returns:
        Model with LoRA patches merged
    """
    logger.info(f"Applying {len(lora_patches)} LoRA patches...")

    state_dict = model.state_dict()
    model_keys = set(state_dict.keys())
    applied = 0
    skipped = 0

    for lora_key, adapter in lora_patches.items():
        if not hasattr(adapter, 'weights'):
            skipped += 1
            continue

        up, down, alpha, *_ = adapter.weights

        # Convert ComfyUI key to string if it's a tuple
        lora_key_str = lora_key[0] if isinstance(lora_key, tuple) else str(lora_key)

        # Find matching diffusers key
        diffusers_key = _find_matching_key(lora_key_str, model_keys)

        if diffusers_key:
            # Compute LoRA delta: scale * (up @ down)
            rank = down.shape[0]
            scale = (alpha / rank) if alpha is not None else 1.0
            delta = scale * (up @ down)

            # Merge into original weight
            original = state_dict[diffusers_key]
            merged = original + delta.to(device=original.device, dtype=original.dtype)
            state_dict[diffusers_key] = merged
            applied += 1
        else:
            skipped += 1
            logger.debug(f"No match found for LoRA key: {lora_key_str}")

    # Reload model with merged weights
    model.load_state_dict(state_dict)

    logger.info(f"✓ Applied {applied}/{len(lora_patches)} LoRA patches ({skipped} skipped)")

    if applied == 0:
        logger.warning(
            "⚠️  WARNING: No LoRA patches were applied!\n"
            "\n"
            "This may happen if:\n"
            "1. LoRA architecture doesn't match the model\n"
            "2. LoRA patches are in an unexpected format\n"
            "\n"
            "Gradient analysis will analyze the BASE MODEL only, not the LoRA's contribution.\n"
            "\n"
            "RECOMMENDATION: Use 'PM LoRA Semantic Analyzer (Heuristic)' instead.\n"
        )
    elif applied < len(lora_patches) * 0.5:
        logger.warning(
            f"Only {applied}/{len(lora_patches)} LoRA patches were applied ({applied/len(lora_patches)*100:.1f}%). "
            f"Results may be inaccurate. Consider using the heuristic analyzer instead."
        )

    return model


def _find_matching_key(lora_key: str, model_keys: set) -> Optional[str]:
    """
    Find matching model key for a LoRA key (simple fuzzy matching).

    zImage uses dot-separated naming, similar to diffusers.

    Args:
        lora_key: ComfyUI LoRA key (e.g., "diffusion_model.layers.5.attn.to_q.lora_up")
        model_keys: Set of model parameter keys

    Returns:
        Matching model key or None
    """
    # Remove LoRA suffixes
    clean = lora_key.replace('.lora_up.weight', '').replace('.lora_down.weight', '')
    clean = clean.replace('.lora_up', '').replace('.lora_down', '')
    clean = clean.replace('.lora_A.weight', '').replace('.lora_B.weight', '')
    clean = clean.replace('.alpha', '')

    # Remove common prefixes
    clean = clean.replace('diffusion_model.', '').replace('model.', '')

    # Try exact match with .weight suffix
    if clean + '.weight' in model_keys:
        return clean + '.weight'

    # Try exact match without .weight
    if clean in model_keys:
        return clean

    # Fuzzy match: find key containing our cleaned key
    clean_base = clean.replace('.weight', '')
    for key in model_keys:
        key_base = key.replace('.weight', '')
        if clean_base in key_base or key_base in clean_base:
            return key

    return None
