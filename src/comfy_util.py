import comfy
import comfy.model_management

def load_as_comfy_lora(lora: dict, model, clip=None):
    """
    Load LoRA from raw state dict into ComfyUI patch format.

    Extracts both UNet and CLIP layers from the raw LoRA state dict using
    key mappings from the provided model and clip objects.

    Args:
        lora: LoRA dictionary containing 'lora_raw' key with raw state dict
        model: MODEL to extract UNet key mappings from
        clip: CLIP model to extract CLIP key mappings from (optional but recommended)

    Returns:
        Patch dictionary with UNet layers (and CLIP layers if clip provided)

    Note:
        Without CLIP, only UNet layers will be loaded and CLIP layers will be
        skipped with "lora key not loaded" warnings. Always provide clip when
        working with LoRAs that contain CLIP weights.
    """
    if 'lora_raw' not in lora or lora['lora_raw'] is None:
        raise ValueError("LoRA data is missing. Please provide a valid LoRA dictionary with 'lora_raw' key.")

    # Build key_map for both UNet and CLIP
    key_map = {}
    if model is not None:
        key_map = comfy.lora.model_lora_keys_unet(model.model, key_map)
    if clip is not None:
        key_map = comfy.lora.model_lora_keys_clip(clip.cond_stage_model, key_map)

    patch_dict = comfy.lora.load_lora(lora['lora_raw'], key_map)
    return patch_dict
