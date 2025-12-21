from typing import Dict

from comfy.weight_adapter import LoRAAdapter

LORA_KEY_DICT = Dict[str, LoRAAdapter]    # Key -> (lora type, (up, down, alpha, None, None, None))
LORA_STACK = Dict[str, LORA_KEY_DICT]                # LoRA name -> LoRA key dict
LORA_WEIGHTS = Dict[str, Dict[str, float]]