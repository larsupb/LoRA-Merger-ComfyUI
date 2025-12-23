from .src.lora_apply import LoraApply
from .src.lora_block_sampler import LoRABlockSampler
from .src.lora_mergekit_merge import LoraMergerMergekit
from .src.lora_selector import LoRASelect
from .src.lora_dir_stacker import LoraStackFromDir
from .src.lora_decompose import LoraDecompose
from .src.lora_parameter_sweep_sampler import LoRAParameterSweepSampler
from .src.lora_power_stacker import LoraPowerStacker
from .src.lora_resize import LoraResizer
from .src.lora_save import LoraSave
from .src.lora_stack_sampler import LoRAStackSampler
from .src.nodes_lora_modifier import LoRAModifier
from .src.nodes_merge_methods import TaskArithmeticMergeMethod, NearSwapMergeMethod, SCEMergeMethod, BreadcrumbsMergeMethod, \
    TIESMergeMethod, DAREMergeMethod, DELLAMergeMethod, SLERPMergeMethod, LinearMergeMethod, NuSlerpMergeMethod, \
    ArceeFusionMergeMethod, KArcherMergeMethod

version_code = [2, 0, 0]
version_str = f"V{version_code[0]}.{version_code[1]}" + (f'.{version_code[2]}' if len(version_code) > 2 else '')
print(f"### Loading: ComfyUI LoRA-PowerMerge ({version_str})")

NODE_CLASS_MAPPINGS = {
    "PM LoRA Merger (Mergekit)": LoraMergerMergekit,

    "PM LoRA Power Stacker": LoraPowerStacker,
    "PM LoRA Stacker (from Directory)": LoraStackFromDir,
    "PM LoRA Select": LoRASelect,
    "PM LoRA Stack Decompose": LoraDecompose,

    "PM LoRA Block Sampler": LoRABlockSampler,
    "PM LoRA Stack Sampler": LoRAStackSampler,
    "PM LoRA Parameter Sweep Sampler": LoRAParameterSweepSampler,

    "PM Slerp (Mergekit)": SLERPMergeMethod,
    "PM NuSlerp (Mergekit)": NuSlerpMergeMethod,
    "PM NearSwap (Mergekit)": NearSwapMergeMethod,
    "PM Arcee Fusion (Mergekit)": ArceeFusionMergeMethod,

    "PM Linear (Mergekit)": LinearMergeMethod,
    "PM SCE (Mergekit)": SCEMergeMethod,
    "PM KArcher (Mergekit)": KArcherMergeMethod,

    "PM Task Arithmetic (Mergekit)": TaskArithmeticMergeMethod,
    "PM Ties (Mergekit)": TIESMergeMethod,
    "PM Breadcrumbs (Mergekit)": BreadcrumbsMergeMethod,
    "PM Dare (Mergekit)": DAREMergeMethod,
    "PM Della (Mergekit)": DELLAMergeMethod,

    "PM LoRA Modifier": LoRAModifier,

    "PM LoRA Resizer": LoraResizer,
    "PM LoRA Apply": LoraApply,
    "PM LoRA Save": LoraSave,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PM LoRA Power Stacker": "PM LoRA Power Stacker",
    "PM LoRA Stacker (from Directory)": "PM LoRA Stacker (from Directory)",
    "PM LoRA Select": "PM LoRA Select",
    "PM LoRA Stack Decompose": "PM LoRA Stack Decompose",

    "PM LoRA Merger (Mergekit)": "PM LoRA Merger (Mergekit)",

    "PM LoRA Block Sampler": "PM LoRA Block Sampler",
    "PM LoRA Stack Sampler": "PM LoRA Stack Sampler",
    "PM LoRA Parameter Sweep Sampler": "PM LoRA Parameter Sweep Sampler",

    "PM Slerp (Mergekit)": "PM Slerp (Mergekit)",
    "PM NuSlerp (Mergekit)": "PM NuSlerp (Mergekit)",
    "PM NearSwap (Mergekit)": "PM NearSwap (Mergekit)",
    "PM Arcee Fusion (Mergekit)": "PM Arcee Fusion (Mergekit)",

    "PM Linear (Mergekit)": "PM Linear (Mergekit)",
    "PM SCE (Mergekit)": "PM SCE (Mergekit)",
    "PM KArcher (Mergekit)": "PM KArcher (Mergekit)",

    "PM Task Arithmetic (Mergekit)": "PM Task Arithmetic (Mergekit)",
    "PM TIES (Mergekit)": "PM TIES (Mergekit)",
    "PM DARE (Mergekit)": "PM DARE (Mergekit)",
    "PM Breadcrumbs (Mergekit)": "PM Breadcrumbs (Mergekit)",
    "PM Della (Mergekit)": "PM Della (Mergekit)",

    "PM LoRA Modifier": "PM LoRA Modifier",

    "PM LoRA Resizer": "PM Resize LoRA",
    "PM LoRA Apply": "PM Apply LoRA",
    "PM LoRA Save": "PM Save LoRA",
}


WEB_DIRECTORY = "./js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
