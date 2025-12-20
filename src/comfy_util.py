import comfy
import comfy.model_management

def load_as_comfy_lora(lora: dict, model, clip):
    if 'lora_raw' not in lora or lora['lora_raw'] is None:
        raise ValueError("LoRA data is missing. Please provide a valid LoRA dictionary with 'lora_raw' key.")
    key_map = {}
    if model is not None:
        key_map = comfy.lora.model_lora_keys_unet(model.model, key_map)
    if clip is not None:
        key_map = comfy.lora.model_lora_keys_clip(clip.cond_stage_model, key_map)
    patch_dict = comfy.lora.load_lora(lora['lora_raw'], key_map)
    return patch_dict
