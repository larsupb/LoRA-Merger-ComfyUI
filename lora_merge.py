import math
from typing import Literal, get_args

import torch
import comfy

from .peft_utils import task_arithmetic, ties, dare_linear, dare_ties, magnitude_prune, concat
# Removed find_network_dim from this import as we are defining a more robust version below
from .utility import to_dtype

CLAMP_QUANTILE = 0.99
MODES = Literal["add", "concat", "ties", "dare_linear", "dare_ties", "magnitude_prune"]
SVD_MODES = Literal["add_svd", "ties_svd", "dare_linear_svd", "dare_ties_svd", "magnitude_prune_svd"]


def find_network_dim(lora_dict):
    """
    Finds the rank/dimension of a LoRA network.
    Handles both up/down and A/B naming conventions.
    """
    for key, weight in lora_dict.items():
        if ".lora_down.weight" in key or ".lora_A.weight" in key:
            # For down weight (down or A), rank is the output dimension
            return weight.shape[0]
    # Fallback or error if no lora keys found
    print("Warning: Could not determine LoRA rank.")
    return 0


def get_naming_convention(lora_dict):
    """
    Determines the naming convention of a LoRA's weights based on its keys.
    Defaults to the common 'up'/'down' style if 'A'/'B' is not found.
    """
    for key in lora_dict.keys():
        if ".lora_A.weight" in key:
            return ".lora_B.weight", ".lora_A.weight"
    return ".lora_up.weight", ".lora_down.weight"


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
        """
        loras = [lora1]
        for k, v in kwargs.items():
            loras.append(v)

        self.validate_input(loras, mode)

        up_suffix, down_suffix = get_naming_convention(lora1['lora'])

        dtype = to_dtype(dtype)
        keys = analyse_keys(loras)
        weight = {}

        pbar = comfy.utils.ProgressBar(len(keys))
        for key in keys:
            scale_key = "strength_clip" if "lora_te" in key else "strength_model"
            weights = torch.tensor([w[scale_key] for w in loras]).to(device, dtype=dtype)
            ups_downs_alphas = calc_up_down_alphas(loras, key)
            ups_downs_alphas, alpha_1 = scale_alphas(ups_downs_alphas)
            ups_downs_alphas = curate_tensors(ups_downs_alphas)

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

            weight[key + up_suffix] = up.to('cpu', dtype=torch.float32)
            weight[key + down_suffix] = down.to('cpu', dtype=torch.float32)
            weight[key + ".alpha"] = alpha_1.to('cpu', dtype=torch.float32)

            pbar.update(1)

        lora_out = {"lora": weight, "strength_model": 1, "strength_clip": 1,
                    "name": "merged_of_" + "_".join([l['name'] for l in loras])}
        return (lora_out,)

    def validate_input(self, loras, mode):
        dims = [find_network_dim(lora['lora']) for lora in loras]
        if min(dims) != max(dims):
            raise Exception("LoRAs with different ranks not allowed in LoraMerger. Use SVD merge.")
        if mode not in get_args(MODES):
            raise Exception(f"Invalid / unsupported mode {mode}")


class LoraSVDMerger:
    """
        Class for merging LoRA models using Singular Value Decomposition (SVD).
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "lora1": ("LoRA",),
                "mode": (get_args(SVD_MODES),),
                "density": ("FLOAT", {
                    "default": 1.0,
                    "min": 0,
                    "max": 1,
                    "step": 0.01,
                }),
                "svd_rank": ("INT", {
                    "default": 16,
                    "min": 1,
                    "max": 320,
                    "step": 1,
                    "display": "number"
                }),
                "svd_conv_rank": ("INT", {
                    "default": 1,
                    "min": 0,
                    "max": 320,
                    "step": 1,
                    "display": "number"
                }),
                "device": (["cuda", "cpu"],),
                "dtype": (["float32", "float16", "bfloat16"],),
            },
        }

    RETURN_TYPES = ("LoRA",)
    FUNCTION = "lora_svd_merge"
    CATEGORY = "LoRA PowerMerge"

    def lora_svd_merge(self, lora1,
                       mode: SVD_MODES = "add_svd",
                       density: float = None, svd_rank: int = None, svd_conv_rank: int = None, device=None, dtype=None,
                       **kwargs):
        """
           Merge LoRA models using SVD and specified mode.
        """
        loras = [lora1]
        for k, v in kwargs.items():
            loras.append(v)
        dtype = to_dtype(dtype)

        self.validate_input(loras, mode)

        up_suffix, down_suffix = get_naming_convention(lora1['lora'])

        weight = {}
        keys = analyse_keys(loras)

        pb = comfy.utils.ProgressBar(len(keys))
        for key in keys:
            strength_key = "strength_clip" if "lora_te" in key else "strength_model"
            strengths = torch.tensor([w[strength_key] for w in loras]).to(device)
            ups_downs_alphas = calc_up_down_alphas(loras, key, fill_with_empty_tensor=True)
            weights = self.build_weights(ups_downs_alphas, strengths, mode, density, device)
            up, down, alpha = self.svd(weights, svd_rank, svd_conv_rank, device)

            weight[key + up_suffix] = up.to(device='cpu', dtype=dtype)
            weight[key + down_suffix] = down.to(device='cpu', dtype=dtype)
            weight[key + ".alpha"] = alpha.to(device='cpu', dtype=dtype)

            pb.update(1)

        lora_out = {"lora": weight, "strength_model": 1, "strength_clip": 1}
        return (lora_out,)

    def validate_input(self, loras, mode):
        if mode not in get_args(SVD_MODES):
            raise Exception(f"Invalid / unsupported mode {mode}")

    def build_weights(self, ups_downs_alphas, strengths,
                      mode: SVD_MODES, density, device):
        """
            Construct the combined weight tensor from multiple LoRA up and down tensors.
        """
        up_1, down_1, alpha_1 = ups_downs_alphas[0]
        conv2d = len(down_1.size()) == 4
        kernel_size = None if not conv2d else down_1.size()[2:4]

        weights = []
        for up, down, alpha in ups_downs_alphas:
            up, down, alpha = up.to(device), down.to(device), alpha.to(device)
            rank = up.shape[1]
            if conv2d:
                if kernel_size == (1, 1):
                    weight = (up.squeeze(3).squeeze(2) @ down.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(
                        3) * alpha / rank
                else:
                    weight = torch.nn.functional.conv2d(down.permute(1, 0, 2, 3), up).permute(1, 0, 2, 3) * alpha / rank
            else:
                weight = up.view(-1, rank) @ down.view(rank, -1) * alpha / rank
            weights.append(weight)

        if mode == "add_svd":
            weight = task_arithmetic(weights, strengths)
        elif mode == "ties_svd":
            weight = ties(weights, strengths, density)
        elif mode == "dare_linear_svd":
            weight = dare_linear(weights, strengths, density)
        elif mode == "dare_ties_svd":
            weight = dare_ties(weights, strengths, density)
        else:
            weight = magnitude_prune(weights, strengths, density)

        return weight

    def svd(self, weights: torch.Tensor, svd_rank: int, svd_conv_rank: int, device: str):
        """
            Perform Singular Value Decomposition (SVD) on the given weights tensor.
        """
        weights = weights.to(dtype=torch.float32, device=device)

        conv2d = len(weights.size()) == 4
        kernel_size = None if not conv2d else weights.size()[2:4]
        conv2d_3x3 = conv2d and kernel_size != (1, 1)
        out_dim, in_dim = weights.size()[0:2]

        if conv2d:
            if conv2d_3x3:
                weights = weights.flatten(start_dim=1)
            else:
                weights = weights.squeeze()

        module_new_rank = svd_conv_rank if conv2d_3x3 else svd_rank
        module_new_rank = min(module_new_rank, in_dim, out_dim)

        U, S, Vh = torch.linalg.svd(weights)

        U = U[:, :module_new_rank]
        S = S[:module_new_rank]
        U = U @ torch.diag(S)

        Vh = Vh[:module_new_rank, :]

        dist = torch.cat([U.flatten(), Vh.flatten()])
        hi_val = torch.quantile(dist, CLAMP_QUANTILE)
        low_val = -hi_val

        U = U.clamp(low_val, hi_val)
        Vh = Vh.clamp(low_val, hi_val)

        if conv2d:
            U = U.reshape(out_dim, module_new_rank, 1, 1)
            Vh = Vh.reshape(module_new_rank, in_dim, kernel_size[0], kernel_size[1])

        return U, Vh, torch.tensor(module_new_rank)


@torch.no_grad()
def calc_up_down_alphas(loras, key, fill_with_empty_tensor=False):
    """
       Calculate up, down tensors and alphas for a given key.
       If alpha is not present, it defaults to the module's rank.
    """
    alpha_key = key + ".alpha"

    # Find a reference LoRA to determine shapes and a default alpha for placeholders
    ref_lora_data = None
    for lora in loras:
        lora_dict = lora['lora']
        if f"{key}.lora_down.weight" in lora_dict or f"{key}.lora_A.weight" in lora_dict:
            ref_lora_data = lora_dict
            break

    if not ref_lora_data:
        return []

    # Determine shapes from the reference LoRA
    if f"{key}.lora_down.weight" in ref_lora_data:
        up_shape = ref_lora_data[f"{key}.lora_up.weight"].shape
        down_shape = ref_lora_data[f"{key}.lora_down.weight"].shape
    else:  # Assume A/B convention
        up_shape = ref_lora_data[f"{key}.lora_B.weight"].shape
        down_shape = ref_lora_data[f"{key}.lora_A.weight"].shape

    # Determine a default alpha value from the reference LoRA. Default to rank if alpha key is missing.
    rank_ref = down_shape[0]
    alpha_1_val = ref_lora_data.get(alpha_key, rank_ref)
    if not isinstance(alpha_1_val, torch.Tensor):
        alpha_1_val = torch.tensor(float(alpha_1_val))

    out = []
    for lora in loras:
        lora_dict = lora['lora']
        up, down = None, None

        down_key_v1 = f"{key}.lora_down.weight"
        if down_key_v1 in lora_dict:
            up = lora_dict[f"{key}.lora_up.weight"]
            down = lora_dict[down_key_v1]
        else:
            down_key_v2 = f"{key}.lora_A.weight"
            if down_key_v2 in lora_dict:
                up = lora_dict[f"{key}.lora_B.weight"]
                down = lora_dict[down_key_v2]

        if up is not None:
            # If module exists, get alpha or default to its rank
            rank = down.shape[0]
            alpha = lora_dict.get(alpha_key, rank)
            if not isinstance(alpha, torch.Tensor):
                alpha = torch.tensor(float(alpha), device=up.device)
            out.append((up, down, alpha))
        elif fill_with_empty_tensor:
            # If creating placeholder, use the default alpha from the reference lora
            up_placeholder = torch.zeros(up_shape)
            down_placeholder = torch.zeros(down_shape)
            out.append((up_placeholder, down_placeholder, alpha_1_val.clone()))

    return out


def scale_alphas(ups_downs_alphas):
    up_1, down_1, alpha_1 = ups_downs_alphas[0]
    out = []
    for up, down, alpha in ups_downs_alphas:
        # Prevent division by zero if alpha is 0
        if alpha_1 != 0:
            scale = math.sqrt(alpha / alpha_1)
            up = up * scale
            down = down * scale
        out.append((up, down, alpha_1))
    return out, alpha_1


def analyse_keys(loras):
    """
    Analyzes the LoRA models to find all unique module keys, supporting multiple naming conventions.
    """
    module_keys = set()
    for i, lora in enumerate(loras):
        key_count = 0
        lora_name = lora.get('name', f'#{i + 1}')
        for key in lora["lora"].keys():
            if ".lora_down.weight" in key:
                base_key = key[:key.rfind(".lora_down.weight")]
                module_keys.add(base_key)
                key_count += 1
            elif ".lora_A.weight" in key:
                base_key = key[:key.rfind(".lora_A.weight")]
                module_keys.add(base_key)
                key_count += 1
        print(f"LoRA '{lora_name}' has {key_count} modules.")

    print(f"Found {len(module_keys)} unique modules to merge.")
    return module_keys


def curate_tensors(ups_downs_alphas):
    """
    Checks and eventually curates tensor dimensions
    """
    up_1, down_1, alpha_1 = ups_downs_alphas[0]
    out = [ups_downs_alphas[0]]
    for up, down, alpha in ups_downs_alphas[1:]:
        up = adjust_tensor_to_match(up_1, up)
        down = adjust_tensor_to_match(down_1, down)
        out.append((up, down, alpha))
    return out


def adjust_tensor_to_match(tensor1: torch.Tensor, tensor2: torch.Tensor) -> torch.Tensor:
    """
    Adjust tensor2 to match the shape of tensor1.
    """
    shape1 = tensor1.shape
    shape2 = tensor2.shape
    new_shape = shape1
    adjusted_tensor = torch.zeros(new_shape, dtype=tensor2.dtype)
    slices = tuple(slice(0, min(dim1, dim2)) for dim1, dim2 in zip(shape1, shape2))
    adjusted_tensor[slices] = tensor2[slices]
    return adjusted_tensor