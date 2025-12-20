from typing import Dict, Any, Protocol, Optional

import torch
from mergekit.architecture import WeightInfo
from mergekit.common import ModelReference, ModelPath, ImmutableMap
from mergekit.io.tasks import GatherTensors
from mergekit.merge_methods import REGISTERED_MERGE_METHODS
from mergekit.merge_methods.generalized_task_arithmetic import GTATask
from mergekit.merge_methods.karcher import KarcherTask
from mergekit.merge_methods.linear import LinearMergeTask
from mergekit.merge_methods.model_stock import ModelStockMergeTask
from mergekit.merge_methods.nearswap import nearswap_merge
from mergekit.merge_methods.nuslerp import NuSlerpTask

import comfy
from ..utility import map_device


class MergeMethod(Protocol):
    def __call__(
            *,
        tensors: Dict[ModelReference, torch.Tensor],
        gather_tensors: GatherTensors,
        base_model: ModelReference,
        weight_info: WeightInfo,
        tensor_parameters: Optional[ImmutableMap[ModelReference, Any]] = ...,
        method_args: Optional[Dict] = ...,
    ) -> torch.Tensor: ...


class CheckpointMergerMergekit:
    """
       Node for merging LoRA models with Mergekit and algorithms that require SVD
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "method": ("PMBaseModelMethod",),
                "base_model": ("MODEL",),
                "model1": ("MODEL",),
                "lambda_": ("FLOAT", {
                    "default": 1,
                    "min": 0,
                    "max": 1,
                    "step": 0.01,
                    "tooltip": "Scaling factor between 0 and 1 applied after weighted sum of task vectors.",
                }),
                "device": (["cuda", "cpu"],),
                "dtype": (["float16", "bfloat16", "float32"],),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "checkpoint_mergekit"
    CATEGORY = "LoRA PowerMerge"

    @torch.no_grad()
    def checkpoint_mergekit(self, method: Dict = None, base_model=None, model1=None, lambda_=None,
                            device=None, dtype=None, **kwargs):
        models = [model1]
        for k, v in kwargs.items():
            models.append(v)

        if method['name'] == "linear":
            merge_method = linear_merge
        elif method['name'] == "model_stock":
            merge_method = model_stock_merge
        elif method['name'] == "nuslerp":
            merge_method = nuslerp_merge
        elif method['name'] == "nearswap":
            merge_method = nearswap_merge_
        elif method['name'] == "task_arithmetic":
            merge_method = task_arithmetic
        else:
            raise Exception(f"Invalid / unsupported method {method['name']}")

        method_args = {
            "normalize": False,  # LinearMerge & Task Arithmetic
            #"max_iter": 10,                    # kArcher only
            #"tol": 0.1,                        # kArcher only
            "row_wise": False,  # NuSlerp only
            "flatten": True,  # NuSlerp only
            "similarity_threshold": 0.5,  # Nearswap only
            "int8_mask": False
        }
        # update method_args with dictionary method['settings']
        method_args.update(method['settings'])

        return checkpoint_process(merge_method=merge_method, method_args=method_args, base_model=base_model,
                                  models=models, lambda_=lambda_, device=device, dtype=dtype)


def checkpoint_process(merge_method: MergeMethod, method_args: Dict, base_model, models, device, dtype, lambda_=1.0):
    device, dtype = map_device(device, dtype)

    keys_base_model = base_model.get_key_patches("diffusion_model.")

    pbar = comfy.utils.ProgressBar(len(keys_base_model))
    for key, v in keys_base_model.items():
        # We need to get the model up and down tensors
        base_model_tensor = v[0][0].to(device=device, dtype=dtype)
        base_model_ref = ModelReference(model=ModelPath(path=key))

        tensors = {base_model_ref: base_model_tensor}
        tensor_weights = {base_model_ref: 0}
        for i, model in enumerate(models):
            ref = ModelReference(model=ModelPath(path=str(i)))
            key_weight, _, _ = comfy.model_patcher.get_key_weight(model.model, key)
            tensors[ref] = key_weight.to(device=device, dtype=dtype)
            tensor_weights[ref] = 1

        # Wrap into mergekit data structure
        immutable_map = ImmutableMap({
            r: WeightInfo(name=f'model{i}.{key}', dtype=dtype) for i, r in enumerate(tensors.keys())
        })
        gather_tensors = GatherTensors(weight_info=immutable_map)
        weight_info = WeightInfo(name='base.' + key, dtype=dtype)

        tensor_parameters = ImmutableMap({r: ImmutableMap({"weight": tensor_weights[r]}) for r in tensors.keys()})

        merge = merge_method(tensors=tensors, base_model=base_model_ref, weight_info=weight_info,
                             gather_tensors=gather_tensors, tensor_parameters=tensor_parameters, method_args=method_args)

        # Apply lambda_ to the merge
        merge *= lambda_

        # pass the merged tensor to the new model
        base_model.add_patches({key: (merge.to(device="cpu"),)}, 1, 0)

        for tv in tensors.values():
            tv.to(device="cpu")

        pbar.update(1)

    return (base_model,)


def linear_merge(
    *,
    tensors: Dict[ModelReference, torch.Tensor],
    base_model: ModelReference,
    weight_info: WeightInfo,
    gather_tensors: GatherTensors,
    tensor_parameters: ImmutableMap[ModelReference, Any],
    method_args: Optional[Dict] = None
) -> torch.Tensor:
    """
    Merges tensors using a linear merge strategy, excluding the base model from the merge.
    """

    method_args = method_args or {}

    # Exclude base_model from tensors
    tensors = {k: v for k, v in tensors.items() if k != base_model}

    task = LinearMergeTask(
        base_model=base_model,
        weight_info=weight_info,
        gather_tensors=gather_tensors,
        tensor_parameters=tensor_parameters,
        normalize=method_args.get("normalize", False)
    )

    return task.execute(tensors=tensors)


def karcher_merge(
    *,
    tensors: Dict[ModelReference, torch.Tensor],
    base_model: ModelReference,
    weight_info: WeightInfo,
    gather_tensors: GatherTensors,
    tensor_parameters: Optional[ImmutableMap[ModelReference, Any]] = None,
    method_args: Optional[Dict] = None,
) -> torch.Tensor:
    method_args = method_args or {}

    # Extract base tensor and remove it from the dictionary
    tensors.pop(base_model)

    task = KarcherTask(
        base_model=base_model,
        weight_info=weight_info,
        gather_tensors=gather_tensors,
        max_iter=method_args.get('max_iter', 10),
        tol=method_args.get('tol', 1e-4),
    )
    return task.execute(tensors=tensors)


def task_arithmetic(
    *,
    tensors: Dict[ModelReference, torch.Tensor],
    base_model: ModelReference,
    weight_info: WeightInfo,
    gather_tensors: GatherTensors,
    tensor_parameters: ImmutableMap[ModelReference, Any],
    method_args: Optional[Dict] = None,
) -> torch.Tensor:
    method_args = method_args or {}

    method = REGISTERED_MERGE_METHODS.get("task_arithmetic")

    task = GTATask(
        method=method,
        base_model=base_model,
        weight_info=weight_info,
        tensors=gather_tensors,
        tensor_parameters=tensor_parameters,
        int8_mask=method_args.get('int8_mask', False),
        normalize=method_args.get('normalize', True),
        lambda_=1.0,
        rescale_norm=None
    )
    return task.execute(tensors=tensors)


def nearswap_merge_(
    *,
    tensors: Dict[ModelReference, torch.Tensor],
    base_model: ModelReference,
    weight_info: WeightInfo,
    gather_tensors: GatherTensors,
    tensor_parameters: ImmutableMap[ModelReference, Any],
    method_args: Optional[Dict] = None,
) -> torch.Tensor:
    method_args = method_args or {}

    # Extract base tensor and remove it from the dictionary
    base_tensor = tensors.pop(base_model)
    other_tensors = list(tensors.values())  # Must be length 1

    return nearswap_merge(
        base_tensor=base_tensor,
        tensors=other_tensors,
        t=method_args.get('similarity_threshold', 0.9)
    )


def model_stock_merge(
    *,
    tensors: Dict[ModelReference, torch.Tensor],
    base_model: ModelReference,
    weight_info: WeightInfo,
    gather_tensors: GatherTensors,
    tensor_parameters: Optional[ImmutableMap[ModelReference, Any]] = None,
    method_args: Optional[Dict] = None,
) -> torch.Tensor:
    task = ModelStockMergeTask(
        base_model=base_model,
        weight_info=weight_info,
        gather_tensors=gather_tensors
    )
    return task.execute(tensors=tensors)


def nuslerp_merge(
    *,
    tensors: Dict[ModelReference, torch.Tensor],
    base_model: ModelReference,
    weight_info: WeightInfo,
    gather_tensors: GatherTensors,
    tensor_parameters: Optional[ImmutableMap[ModelReference, Any]] = None,
    method_args: Optional[Dict] = None,
) -> torch.Tensor:
    method_args = method_args or {}

    task = NuSlerpTask(
        base_model=base_model,
        weight_info=weight_info,
        gather_tensors=gather_tensors,
        tensor_parameters=tensor_parameters,
        row_wise=method_args.get('row_wise', False),
        flatten=method_args.get('flatten', False)
    )
    return task.execute(tensors=tensors)

