from typing import Literal, get_args

import torch

import comfy
from .peft_utils import task_arithmetic, ties, dare_linear, dare_ties, magnitude_prune, concat
from .sd_lora import analyse_keys, calc_up_down_alphas, scale_alphas
from .utility import find_network_dim, to_dtype, adjust_tensor_dims

MODES = Literal["add", "concat", "ties", "dare_linear", "dare_ties", "magnitude_prune"]


class LoraMerger:
    """
       Class for merging LoRA models using various methods.

       Attributes:
           loaded_lora: A placeholder for the loaded LoRA model.
    """

    def __init__(self):
        self.loaded_lora = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "lora1": ("LoRA",),
                "mode": (get_args(MODES),),
                "density": ("FLOAT", {
                    "default": 1.0,
                    "min": 0,
                    "max": 1,
                    "step": 0.01,
                }),
                "epsilon": ("FLOAT", {
                    "default": 0.01,
                    "min": 0,
                    "max": 1,
                    "step": 0.01,
                }),
                "device": (["cuda", "cpu"],),
                "dtype": (["float32", "float16", "bfloat16"],),
            },
        }

    RETURN_TYPES = ("LoRA",)
    FUNCTION = "lora_merge"
    CATEGORY = "LoRA PowerMerge"

    @torch.no_grad()
    def lora_merge(self, lora1,
                   mode: MODES = None,
                   density=None, device=None, dtype=None, **kwargs):
        """
            Merge multiple LoRA models using the specified mode.

            This method merges the given LoRA models according to the specified mode, such as 'add', 'concat', 'ties',
            'dare_linear', 'dare_ties', or 'magnitude_prune'. The merging process considers the up and down
            projection matrices and their respective alpha values.

            Args:
                lora1 (dict): The first LoRA model to merge.
                mode (str, optional): The merging mode to use. Options include 'add', 'concat', 'ties', 'dare_linear',
                    'dare_ties', and 'magnitude_prune'. Default is None.
                density (float, optional): The density parameter used for some merging modes.
                device (torch.DeviceObjType, optional): The device to use for computations (e.g., 'cuda' or 'cpu').
                dtype (torch.dtype, optional): The data type to use for computations (e.g., 'float32', 'float16', 'bfloat16').
                **kwargs: Additional LoRA models to merge.

            Returns:
                tuple: A tuple containing the merged LoRA model.

            Note:
                - The method ensures that all tensors are moved to the specified device and cast to the specified data type.
                - The merging process involves calculating task weights, scaling with alpha values, and combining
                  up and down projection matrices based on the chosen mode.
        """
        loras = [lora1]
        for k, v in kwargs.items():
            loras.append(v)

        self.validate_input(loras, mode)

        dtype = to_dtype(dtype)
        keys = analyse_keys(loras)
        weight = {}

        # lora = up @ down * alpha / rank
        pbar = comfy.utils.ProgressBar(len(keys))
        for key in keys:
            # Build taskTensor weights
            scale_key = "strength_clip" if "lora_te" in key else "strength_model"
            weights = torch.tensor([w[scale_key] for w in loras]).to(device, dtype=dtype)
            alpha_0 = loras[0]["lora"][key + '.alpha']
            alpha_0 = torch.tensor(alpha_0)

            # Calculate up and down nets and their alphas
            ups_downs_alphas = calc_up_down_alphas(loras, key, scale_to_alpha_0=True)

            # Assure that dimensions are equal in every tensor of the same layer
            ups_downs_alphas = adjust_tensor_dims(ups_downs_alphas, False)

            up_tensors = [up.to(device, dtype=dtype) for up, down, alpha in ups_downs_alphas]
            down_tensors = [down.to(device, dtype=dtype) for up, down, alpha in ups_downs_alphas]

            if mode == "add":
                up, down = (task_arithmetic(up_tensors, weights),
                            task_arithmetic(down_tensors, weights))
            elif mode == "concat":
                up, down = (concat(up_tensors, weights, dim=1),
                            concat(down_tensors, weights, dim=0))
            elif mode == "ties":
                up, down = (ties(up_tensors, weights, density),
                            ties(down_tensors, weights, density))
            elif mode == "dare_linear":
                up, down = (dare_linear(up_tensors, weights, density),
                            dare_linear(down_tensors, weights, density))
            elif mode == "dare_ties":
                up, down = (dare_ties(up_tensors, weights, density),
                            dare_ties(down_tensors, weights, density))
            else:
                up, down = (magnitude_prune(up_tensors, weights, density),
                            magnitude_prune(down_tensors, weights, density))

            weight[key + ".lora_up.weight"] = up.to('cpu', dtype=torch.float32)
            weight[key + ".lora_down.weight"] = down.to('cpu', dtype=torch.float32)
            weight[key + ".alpha"] = alpha_0

            pbar.update(1)

        lora_out = {"lora": weight, "strength_model": 1, "strength_clip": 1,
                    "name": "merged_of_" + "_".join([l['name'] for l in loras])}
        return (lora_out,)

    @staticmethod
    def validate_input(loras, mode):
        dims = [find_network_dim(lora['lora']) for lora in loras]
        if min(dims) != max(dims):
            raise Exception("LoRAs with different ranks not allowed in LoraMerger. Use SVD merge.")
        if mode not in get_args(MODES):
            raise Exception(f"Invalid / unsupported mode {mode}")
