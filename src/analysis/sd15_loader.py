"""
Load SD1.5 UNet for gradient analysis using diffusers from_single_file().

Similar to SDXL loader but for Stable Diffusion 1.5 models.
"""

import torch
import logging
from typing import Optional

try:
    from diffusers import StableDiffusionPipeline
except ImportError as e:
    raise ImportError(
        "diffusers library required for gradient analysis. "
        "Install with: pip install diffusers>=0.30.0"
    ) from e

logger = logging.getLogger(__name__)


def load_sd15_for_gradients(
    checkpoint_path: str,
    device: torch.device,
    dtype: torch.dtype,
    offload_layers: bool = False,
    gpu_memory_gb: Optional[float] = None,
) -> StableDiffusionPipeline:
    """
    Load SD1.5 UNet from checkpoint using diffusers' from_single_file().

    Same approach as SDXL but using StableDiffusionPipeline instead.

    Args:
        checkpoint_path: Path to .safetensors checkpoint file
        device: cuda/cpu (used if offload_layers=False)
        dtype: torch.float32, torch.float16, torch.bfloat16
        offload_layers: If True, enable automatic CPU offloading
        gpu_memory_gb: Ignored (kept for API compatibility)

    Returns:
        SD1.5 UNet ready for gradient computation

    Raises:
        ImportError: If diffusers is not installed
        FileNotFoundError: If checkpoint_path doesn't exist
        RuntimeError: If model loading fails
    """
    import os

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    logger.info(f"Loading SD1.5 UNet from checkpoint: {checkpoint_path}")
    logger.info(f"Using from_single_file() - no HuggingFace download needed!")

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

        # Load checkpoint directly
        logger.info(f"Loading in {gradient_compatible_dtype} precision...")
        pipe = StableDiffusionPipeline.from_single_file(
            checkpoint_path,
            torch_dtype=gradient_compatible_dtype,
            use_safetensors=True,
            # Don't load safety checker (new API)
            safety_checker=None,
            feature_extractor=None,
            # Allow downloading tiny config files but not model weights
        )

        logger.info("✓ Pipeline loaded successfully")

        # Setup device placement
        if offload_layers:
            logger.info("Setting up automatic CPU offloading...")
            pipe.enable_model_cpu_offload()
            logger.info("✓ CPU offloading enabled")
        else:
            logger.info(f"Moving pipeline to {device}...")
            pipe.to(device)
            logger.info(f"✓ Pipeline on {device}")

        # Extract UNet
        unet = pipe.unet

        # Enable gradient computation
        logger.info("Enabling gradients for UNet...")
        unet.train()
        unet.requires_grad_(True)

        logger.info("✓ SD1.5 UNet loaded successfully")
        logger.info(f"  Device: {device}, dtype: {dtype}")
        logger.info(f"  CPU offloading: {'enabled' if offload_layers else 'disabled'}")
        logger.info(f"  Parameters: {sum(p.numel() for p in unet.parameters()):,}")
        logger.info(f"  Gradient enabled: {next(unet.parameters()).requires_grad}")

        return pipe

    except Exception as e:
        logger.error(f"Failed to load SD1.5 UNet: {e}", exc_info=True)
        raise RuntimeError(f"Failed to load SD1.5 UNet: {e}") from e
