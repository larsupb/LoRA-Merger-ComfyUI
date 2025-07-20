import hashlib
import logging
import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import get_args, Dict, Protocol, Optional, Any, Set

import torch
from mergekit.architecture import WeightInfo
from mergekit.common import ModelReference, ModelPath, ImmutableMap
from mergekit.io.tasks import GatherTensors
from mergekit.merge_methods import REGISTERED_MERGE_METHODS
from mergekit.merge_methods.arcee_fusion import ArceeFusionMerge
from mergekit.merge_methods.generalized_task_arithmetic import GTATask
from mergekit.merge_methods.karcher import KarcherMerge
from mergekit.merge_methods.linear import LinearMergeTask
from mergekit.merge_methods.nearswap import nearswap_merge
from mergekit.merge_methods.nuslerp import NuSlerpTask
from mergekit.merge_methods.sce import sce_merge
from mergekit.merge_methods.slerp import SlerpTask
from mergekit.sparsify import RescaleNorm

import comfy
from comfy.weight_adapter import LoRAAdapter
from .mergekit_utils import MERGEKIT_GTA_MODES, load_on_device
from .nodes_experimental_methods import cosine_merge
from .sd_lora import LORAS_LORA_KEY_DICT, LORA_STRENGTHS, UP_DOWN_ALPHA_TUPLE, weights_as_tuple
from .sd_lora import analyse_keys, calc_up_down_alphas
from .utility import map_device, adjust_tensor_dims

LORA_COMPONENT_DICT = Dict[str, UP_DOWN_ALPHA_TUPLE]
LORAS_COMPONENT_DICT = Dict[str, LORA_COMPONENT_DICT]


class MergeMethod(Protocol):
    def __call__(
            *,
            tensors: Dict[ModelReference, torch.Tensor],
            gather_tensors: GatherTensors,
            weight_info: WeightInfo,
            tensor_parameters: Optional[ImmutableMap[ModelReference, Any]] = ...,
            method_args: Optional[Dict] = ...,
    ) -> torch.Tensor: ...


def create_map(key, tensors, dtype):
    return (ImmutableMap({
        r: WeightInfo(name=f'model{i}.{key}', dtype=dtype) for i, r in enumerate(tensors.keys())
    }))


def create_tensor_param(tensor_weight, method_args: Dict) -> Dict:
    out = {"weight": tensor_weight}
    out.update(method_args)
    return out


def parse_layer_filter(layer_filter):
    # Set filter on SD LoRA layers
    if layer_filter == "full":
        layer_filter = None
    elif layer_filter == "attn-mlp":
        layer_filter = {"attn1", "attn2", "ff"}
    elif layer_filter == "attn-only":
        layer_filter = {"attn1", "attn2"}
    return layer_filter


def apply_layer_filter(patch_dict, layer_filter):
    num_keys = len(patch_dict.keys())
    if layer_filter:
        patch_dict = {k0: v0 for k0, v0 in patch_dict.items() if any(layer in k0 for layer in layer_filter)}
    print(f"Stacking {len(patch_dict)} keys with {num_keys - len(patch_dict)} "
          f"filtered out by filter method {layer_filter}.")
    return patch_dict


class LoraStack:
    """
       Node for loading LoRA weights
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The diffusion model the LoRA will be applied to."}),
                "clip": ("CLIP", {"tooltip": "The CLIP model the LoRA will be applied to."}),
                "lora1": ("LoRA",),
                "layer_filter": (
                    ["full", "attn-mlp", "attn-only"], {"default": "full", "tooltip": "Filter for specific layers."}),
            },
        }

    RETURN_TYPES = ("LoRAKeyDict", "LoRAStrengths",)
    FUNCTION = "stack_loras"
    CATEGORY = "LoRA PowerMerge"
    DESCRIPTION = "Stacks LoRA weights from the given LoRA dictionary and applies them to the model."

    def stack_loras(self, model, clip, lora1=None, layer_filter=None, **kwargs) -> (LORAS_LORA_KEY_DICT, LORA_STRENGTHS):
        key_map = {}
        if model is not None:
            key_map = comfy.lora.model_lora_keys_unet(model.model, key_map)
        if clip is not None:
            key_map = comfy.lora.model_lora_keys_clip(clip.cond_stage_model, key_map)

        layer_filter = parse_layer_filter(layer_filter)

        # Load LoRAs and patch key names
        lora_patch_dicts = {}
        lora_strengths = {}
        for v in [lora1] + list(kwargs.values()):
            patch_dict = comfy.lora.load_lora(v['lora_raw'], key_map)
            patch_dict = apply_layer_filter(patch_dict, layer_filter)
            lora_patch_dicts[v['name']] = patch_dict
            lora_strengths[v['name']] = {'strength_model': v['strength_model'], 'strength_clip': v['strength_clip']}

        return lora_patch_dicts, lora_strengths,


class LoraStackFromDir:
    """
       Node for loading LoRA weights
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The diffusion model the LoRA will be applied to."}),
                "clip": ("CLIP", {"tooltip": "The CLIP model the LoRA will be applied to."}),
                "directory": ("STRING",),
                "layer_filter": (
                    ["full", "attn-mlp", "attn-only"], {"default": "full", "tooltip": "Filter for specific layers."}),
                "sort_by": (["name", "name descending", "date", "date descending"],
                            {"default": "name", "tooltip": "Sort LoRAs by name or size."}),
                "limit": ("INT", {"default": -1, "min": -1, "max": 1000, "tooltip": "Limit the number of LoRAs to load."}),
            },
        }

    RETURN_TYPES = ("LoRAKeyDict", "LoRAStrengths",)
    FUNCTION = "stack_loras"
    CATEGORY = "LoRA PowerMerge"
    DESCRIPTION = "Stacks LoRA weights from the given directory and applies them to the model."

    def stack_loras(self, model, clip, directory, layer_filter=None, sort_by: str = None, limit: int = 0) -> \
            (LORAS_LORA_KEY_DICT, LORA_STRENGTHS):
        # check if directory exists
        if not os.path.isdir(directory):
            raise FileNotFoundError(f"Directory {directory} does not exist.")

        key_map = {}
        if model is not None:
            key_map = comfy.lora.model_lora_keys_unet(model.model, key_map)
        if clip is not None:
            key_map = comfy.lora.model_lora_keys_clip(clip.cond_stage_model, key_map)

        layer_filter = parse_layer_filter(layer_filter)

        # Load LoRAs and patch key names
        lora_patch_dicts = {}
        lora_strengths = {}

        # Load LoRAs from the directory
        # walk over files in the directory
        for root, _, files in os.walk(directory):
            # Sort files based on the specified criteria
            if sort_by == "name":
                files = sorted(files)
            elif sort_by == "name descending":
                files = sorted(files, reverse=True)
            elif sort_by == "date":
                files = sorted(files, key=lambda f: os.path.getmtime(os.path.join(root, f)))
            elif sort_by == "date descending":
                files = sorted(files, key=lambda f: os.path.getmtime(os.path.join(root, f)), reverse=True)
            # Limit the number of LoRAs to load
            if limit > 0:
                files = files[:limit]

            for file in files:
                if file.endswith(".safetensors") or file.endswith(".ckpt"):
                    lora_path = os.path.join(root, file)
                    lora_name = os.path.splitext(file)[0]
                    lora_raw = comfy.utils.load_torch_file(lora_path, safe_load=True)
                    patch_dict = comfy.lora.load_lora(lora_raw, key_map)
                    patch_dict = apply_layer_filter(patch_dict, layer_filter)
                    lora_patch_dicts[lora_name] = patch_dict
                    lora_strengths[lora_name] = {
                        'strength_model': 1.0,  # Default strength
                        'strength_clip': 1.0,   # Default strength
                    }

        return lora_patch_dicts, lora_strengths,


class LoRASelect:
    """
    Select one LoRA out of a LoRAKeyDicts Object by its index.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "key_dicts": ("LoRAKeyDict",),
                "index": ("INT", {"default": 0, "min": 0, "max": 1000, "tooltip": "Index of the LoRA to select."}),
            },
        }

    RETURN_TYPES = ("LoRA",)
    FUNCTION = "select_lora"
    CATEGORY = "LoRA PowerMerge"

    def select_lora(self, key_dicts: LORAS_LORA_KEY_DICT, index: int) -> (LORAS_LORA_KEY_DICT,):
        keys = list(key_dicts.keys())
        if index < 0 or index >= len(keys):
            raise IndexError(f"Index {index} out of range for LoRAKeyDict with {len(keys)} items.")
        selected_key = keys[index]

        return ({"lora": key_dicts[selected_key],
                 "strength_model": 1,
                 "strength_clip": 1,
                 "name": selected_key},)


class LoraDecompose:
    """
       Node for decomposing LoRA models into their components
    """

    def __init__(self):
        self.last_lora_names_hash: Optional[list] = None
        self.last_tensor_sum: float = 0.0
        self.last_svd_rank: int = -1
        self.last_layer_filter: Optional[Set[str]] = None
        self.last_result: LORAS_COMPONENT_DICT = {}

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "key_dicts": ("LoRAKeyDict",),
                "svd_if_required": (
                    ["True", "False"], {"default": "True", "tooltip": "Apply SVD to align tensor dimensions."}),
                "svd_rank": ("INT", {"default": -1, "min": -1, "max": 128, "tooltip": "Rank for SVD."}),
                "device": (["cuda", "cpu"],),
            },
        }

    RETURN_TYPES = ("LoRAComponents",)
    FUNCTION = "lora_decompose"
    CATEGORY = "LoRA PowerMerge"

    def lora_decompose(self, key_dicts: LORAS_LORA_KEY_DICT = None,
                       svd_if_required=True, svd_rank=-1, device=None):
        device, _ = map_device(device, "float32")

        # check if key_dicts differs from the previous one
        lora_names_hash_new = self.compute_hash(list(key_dicts.keys()))
        if (self.last_lora_names_hash == lora_names_hash_new
                and self.last_svd_rank == svd_rank
                and self.last_tensor_sum == self.compute_sum(key_dicts)):
            logging.info("Key dicts have not changed, returning last result.")
            if self.last_result is not None:
                return (self.last_result,)
            else:
                logging.warning("No last result available, recomputing.")
        else:
            logging.info("Key dicts have changed, recomputing.")

        self.last_lora_names_hash = lora_names_hash_new
        self.last_tensor_sum = self.compute_sum(key_dicts)
        self.last_svd_rank = svd_rank

        self.last_result = self.decompose(key_dicts=key_dicts, device=device, svd_if_required=svd_if_required,
                                          svd_rank=svd_rank)
        return (self.last_result,)

    @staticmethod
    def compute_hash(value):
        """Computes a hash of the value for change detection."""
        return hashlib.md5(str(value).encode()).hexdigest()

    @staticmethod
    def compute_sum(lora_key_dicts: LORAS_LORA_KEY_DICT):
        """Computes the sum of all up, down, and alpha tensors in the LoRA key dicts."""
        sum_ = 0
        for lora_name, lora_key_dict in lora_key_dicts.items():
            for key in lora_key_dict.keys():
                lora_adapter = lora_key_dict[key]
                up, down, alpha, _, _, _ = lora_adapter.weights
                sum_ += up.sum().item() + down.sum().item() + alpha
        return sum_

    def decompose(self, key_dicts, device, svd_if_required, svd_rank) -> LORAS_COMPONENT_DICT:
        """
        Decomposes LoRA models into their components.
        Args:
            key_dicts: Dictionary of LoRA names and their respective keys.
            device: Device to load tensors on.
            svd_if_required: Whether to apply SVD to align tensor dimensions.
            svd_rank: Rank for SVD.
        Returns:
            Dictionary of LoRA components.
            lora_key -> lora_name -> (up, down, alpha)
        """
        keys = list(analyse_keys(key_dicts))  # [:10]  # Limit to 100 keys for testing

        pbar = comfy.utils.ProgressBar(len(keys))
        start = time.time()

        def process_key(key, device_=device) -> LORA_COMPONENT_DICT:
            uda = calc_up_down_alphas(key_dicts, key, load_device=device_, scale_to_alpha_0=True)
            uda_adjusted = adjust_tensor_dims(uda, apply_svd=svd_if_required, svd_rank=svd_rank)
            return uda_adjusted

        with ThreadPoolExecutor(max_workers=8) as executor:
            # distribute the work across available devices
            devices_avail = list({"cpu", device})

            futures = {
                executor.submit(process_key, key, devices_avail[i % len(devices_avail)]): key
                for i, key in enumerate(keys)
            }

            out = {}
            for future in as_completed(futures):
                key = futures[future]
                result = future.result()
                if result:
                    out[key] = result
                pbar.update(1)

        print(f"Processed {len(keys)} keys in {time.time() - start:.2f} seconds")

        torch.cuda.empty_cache()

        return out


class LoraMergerMergekit:
    """
       Node for merging LoRA models with Mergekit
    """

    def __init__(self):
        self.components: LORAS_COMPONENT_DICT = {}
        self.strengths: LORA_STRENGTHS = {}

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "method": ("PMBaseModelMethod",),
                "components": ("LoRAComponents", {"tooltip": "The decomposed components of the LoRAs to be merged."}),
                "strengths": ("LoRAStrengths", {"tooltip": "The weights of the LoRAs to be merged."}),
                "lambda_": ("FLOAT", {
                    "default": 1,
                    "min": 0,
                    "max": 1,
                    "step": 0.01,
                    "tooltip": "Lambda value for scaling the merged model.",
                }),
                "device": (["cuda", "cpu"],),
                "dtype": (["float16", "bfloat16", "float32"],),
            },
        }

    RETURN_TYPES = ("LoRA",)
    FUNCTION = "lora_mergekit"
    CATEGORY = "LoRA PowerMerge"

    @torch.no_grad()
    def lora_mergekit(self,
                      method: Dict = None,
                      components: LORAS_COMPONENT_DICT = None,
                      strengths: LORA_STRENGTHS = None,
                      lambda_: float = 1.0,
                      device=None, dtype=None):

        if components is None:
            raise Exception("No components provided for merging.")

        self.components = components
        self.strengths = strengths

        device, dtype = map_device(device, dtype)

        if method['name'] == "linear":
            merge_method = linear_merge
        elif method['name'] == "nearswap":
            merge_method = nearswap_merge_
        elif method['name'] == "slerp":
            merge_method = slerp_merge
        elif method['name'] == "nuslerp":
            merge_method = nuslerp_merge
        elif method['name'] == "sce":
            merge_method = sce
        elif method['name'] == "karcher":
            merge_method = kArcher
        elif method['name'] == "arcee_fusion":
            merge_method = arcee_fusion
        elif method['name'] == "cosine_merge":
            merge_method = cosine_merge
        elif method['name'] in get_args(MERGEKIT_GTA_MODES):
            merge_method = generalized_task_arithmetic_merge
        else:
            raise Exception(f"Invalid / unsupported method {method['name']}")

        method_args = {
            "mode": method['name'],
            "int8_mask": False,
            "lambda_": 1.0,  # This is for internal GTA processing -> but since we want to apply lambda_ to every merge method, we use it afterward
        }
        # update method_args with dictionary method['settings']
        method_args.update(method['settings'])

        self.validate_input()

        # Adjust components to match the method requirements
        merge = self.merge(method=merge_method, method_args=method_args, lambda_=lambda_, device=device, dtype=dtype)
        # Clean up VRAM
        torch.cuda.empty_cache()

        return merge

    def merge(self, method: MergeMethod, method_args, lambda_, device, dtype):
        pbar = comfy.utils.ProgressBar(len(self.components.keys()))
        start = time.time()

        def process_key(key):
            lora_key_tuples: Dict[str, UP_DOWN_ALPHA_TUPLE] = self.components[key]

            scale_key = "strength_clip" if "clip" in key else "strength_model"
            weights = [self.strengths[lora_name][scale_key] for lora_name in lora_key_tuples.keys()]
            weights = torch.tensor(weights, dtype=dtype).to(device=device)

            def calculate(tensors_):
                tensor_map = {}
                tensor_weight_map = {}
                weight_info = WeightInfo(name=f'{key}.merge', dtype=dtype, is_embed=False)
                for i, t in enumerate(tensors_):
                    ref = ModelReference(model=ModelPath(path=f'{key}.{i}'))
                    tensor_map[ref] = t
                    tensor_weight_map[ref] = weights[i]

                gather_tensors = GatherTensors(weight_info=create_map(key, tensor_map, dtype))
                tensor_parameters = ImmutableMap(
                    {r: ImmutableMap(create_tensor_param(tensor_weight_map[r], method_args)) for r in
                     tensor_map.keys()})

                # Load to the device
                load_on_device(tensor_map, tensor_weight_map, device, dtype)

                # Call the merge method
                out = method(tensor_map, gather_tensors, weight_info, tensor_parameters, method_args)

                # Apply lambda scaling
                if method_args['lambda_'] != 1.0:
                    out = out * method_args['lambda_']

                # Offload the result to CPU
                load_on_device(tensor_map, tensor_weight_map, "cpu", dtype)
                out = out.to(device='cpu', dtype=torch.float32)
                return out

            up = calculate([u for u, _, _ in lora_key_tuples.values()])
            down = calculate([d for _, d, _ in lora_key_tuples.values()])
            alpha_0 = next(iter(lora_key_tuples.values()))[2]

            return key, (up, down, alpha_0)

        adapter_state_dict = {}
        with ThreadPoolExecutor(max_workers=8) as executor:
            keys = self.components.keys()

            # distribute the work across available devices
            futures = {executor.submit(process_key, key) for key in keys}

            for future in as_completed(futures):
                key, result = future.result()
                if result:
                    up, down, alpha_0 = result
                    adapter_state_dict[key] = LoRAAdapter(weights=weights_as_tuple(up, down, alpha_0),
                                                          loaded_keys=set(keys))
                pbar.update(1)

        print(f"Processed {len(keys)} keys in {time.time() - start:.2f} seconds")

        lora_out = {"lora": adapter_state_dict, "strength_model": 1, "strength_clip": 1, "name": "Merge"}
        return (lora_out,)

    def validate_input(self):
        pass
        #if len(self.loras) < 2:
        #   raise Exception("At least two LoRAs are required for merge.")
        # dims = [find_network_dim(lora['lora']) for lora in self.loras]
        # if min(dims) != max(dims):
        #     raise Exception("LoRAs with different ranks not allowed in LoraMerger. Use SVD merge.")


def generalized_task_arithmetic_merge(
        tensors: Dict[ModelReference, torch.Tensor],
        gather_tensors: GatherTensors,
        weight_info: WeightInfo,
        tensor_parameters: Optional[ImmutableMap[ModelReference, Any]] = ...,
        method_args: Optional[Dict] = ...,
) -> torch.Tensor:
    # Tell GTA what method to use exactly
    mode = method_args['mode']
    if mode == "dare":
        mode = "dare_ties" if method_args['sign_consensus_algorithm'] else "dare_linear"
    elif mode == "breadcrumbs" and method_args['sign_consensus_algorithm']:
        mode = "breadcrumbs_ties"
    elif mode == "della" and not method_args['sign_consensus_algorithm']:
        mode = "della_linear"

    # Merge LoRA tensors using task arithmetic method.
    method = REGISTERED_MERGE_METHODS.get(mode)

    # Add full zeros tensor to the tensors and set it as base tensor
    # This is a dummy tensor (zeros) that will have no effect on the merge
    # Model Reference and zero tensor are created
    zeros_tensor = torch.zeros_like(list(tensors.values())[0])
    base_model_ref = ModelReference(model=ModelPath(path='zeros.base'))
    tensors[base_model_ref] = zeros_tensor
    # Extends TensorParameters is a dictionary of tensors and their weights
    map = {base_model_ref: ImmutableMap(create_tensor_param(zeros_tensor, method_args))}
    # Add the base tensor to the weight info
    for k, v in tensor_parameters.items():
        map[k] = v
    tensor_parameters = ImmutableMap(map)

    rescale_norm = method_args["rescale_norm"]
    if rescale_norm == "default":
        rescale_norm = RescaleNorm.l1 \
            if getattr(method, "default_rescale") else None
    task = GTATask(
        method=method,
        tensors=gather_tensors,
        base_model=base_model_ref,
        weight_info=weight_info,
        gather_tensors=gather_tensors,
        tensor_parameters=tensor_parameters,
        int8_mask=method_args['int8_mask'],
        normalize=method_args['normalize'],
        lambda_=method_args['lambda_'],
        rescale_norm=rescale_norm
    )
    return task.execute(tensors=tensors)


def linear_merge(
        tensors: Dict[ModelReference, torch.Tensor],
        gather_tensors: GatherTensors,
        weight_info: WeightInfo,
        tensor_parameters: Optional[ImmutableMap[ModelReference, Any]] = ...,
        method_args: Optional[Dict] = ...,

) -> torch.Tensor:
    # Merge LoRA tensors using linear method.
    task = LinearMergeTask(
        gather_tensors=gather_tensors,
        tensor_parameters=tensor_parameters,
        normalize=method_args['normalize'],
        weight_info=weight_info,
    )
    return task.execute(tensors=tensors)


def sce(
        tensors: Dict[ModelReference, torch.Tensor],
        gather_tensors: GatherTensors,
        weight_info: WeightInfo,
        tensor_parameters: Optional[ImmutableMap[ModelReference, Any]] = ...,
        method_args: Optional[Dict] = ...,
) -> torch.Tensor:
    first_tensor = next(iter(tensors.values()))

    # # Apply weight to each tensor
    # weighted_tensors = []
    # for ref, tv in tensor_parameters.items():
    #     weighted_tensors.append(tv["weight"] * tensors[ref])  # Ensure that the weight is applied to the tensor

    # Add full zeros tensor to the tensors and set it as base tensor
    # This is a dummy tensor (zeros) that will have no effect on the merge
    zeros_tensor = torch.zeros_like(first_tensor)
    return sce_merge(tensors=list(tensors.values()), base_tensor=zeros_tensor,
                     int8_mask=method_args['int8_mask'],
                     select_topk=method_args['select_topk']) * method_args['lambda_']


def kArcher(
        tensors: Dict[ModelReference, torch.Tensor],
        gather_tensors: GatherTensors,
        weight_info: WeightInfo,
        tensor_parameters: Optional[ImmutableMap[ModelReference, Any]] = ...,
        method_args: Optional[Dict] = ...,
) -> torch.Tensor:
    merge = KarcherMerge()
    task = merge.make_task(
        output_weight=weight_info,
        tensors=gather_tensors,
        base_model=None,  # We do not provide a KArcher base model for LoRA merging here
        parameters=ImmutableMap({
            "max_iter": method_args.get("max_iter", 10),
            "tol": method_args.get("tol", 1e-5)
        }),
        tensor_parameters=tensor_parameters,
    )
    return task.execute(tensors=tensors) * method_args['lambda_']


def slerp_merge(
        tensors: Dict[ModelReference, torch.Tensor],
        gather_tensors: GatherTensors,
        weight_info: WeightInfo,
        tensor_parameters: Optional[ImmutableMap[ModelReference, Any]] = ...,
        method_args: Optional[Dict] = ...,
) -> torch.Tensor:
    first_model_ref = list(tensors.keys())[0]

    task = SlerpTask(gather_tensors=gather_tensors, base_model=first_model_ref,
                     weight_info=weight_info, t=method_args['t'])
    return task.execute(tensors=tensors) * method_args['lambda_']


def nuslerp_merge(
        tensors: Dict[ModelReference, torch.Tensor],
        gather_tensors: GatherTensors,
        weight_info: WeightInfo,
        tensor_parameters: Optional[ImmutableMap[ModelReference, Any]] = ...,
        method_args: Optional[Dict] = ...,
) -> torch.Tensor:
    task = NuSlerpTask(gather_tensors=gather_tensors, tensor_parameters=tensor_parameters, weight_info=weight_info,
                       row_wise=method_args['nuslerp_row_wise'], flatten=method_args['nuslerp_flatten'],
                       base_model=None)
    return task.execute(tensors=tensors) * method_args['lambda_']


def nearswap_merge_(
        tensors: Dict[ModelReference, torch.Tensor],
        gather_tensors: GatherTensors,
        weight_info: WeightInfo,
        tensor_parameters: Optional[ImmutableMap[ModelReference, Any]] = ...,
        method_args: Optional[Dict] = ...,
) -> torch.Tensor:
    method_args = method_args or {}

    # take the first tensor as base tensor
    first_model = tensors.pop(list(tensors.keys())[0])
    # check that there is only one tensor left
    if len(tensors) != 1:
        raise RuntimeError("NearSwap merge expects exactly two models")
    second_model = list(tensors.values())  # Must be length 1

    divisor = 1
    if method_args['normalize']:
        divisor = torch.tensor(2.)
        divisor[divisor.abs() < 1e-8] = 1

    return nearswap_merge(
        base_tensor=first_model,
        tensors=second_model,
        t=method_args.get('similarity_threshold', 0.001)
    ) * method_args['lambda_'] / divisor


def arcee_fusion(
        tensors: Dict[ModelReference, torch.Tensor],
        gather_tensors: GatherTensors,
        weight_info: WeightInfo,
        tensor_parameters: Optional[ImmutableMap[ModelReference, Any]] = ...,
        method_args: Optional[Dict] = ...,
) -> torch.Tensor:
    # take the first tensor as base tensor
    first_model = list(tensors.keys())[0]

    merge = ArceeFusionMerge()
    task = merge.make_task(
        output_weight=weight_info,
        tensors=gather_tensors,
        base_model=first_model,
    )
    return task.execute(
        tensors=tensors,
    ) * method_args['lambda_']
