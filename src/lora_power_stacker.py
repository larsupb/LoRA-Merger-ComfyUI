import os
import folder_paths
import comfy
import comfy.lora
import comfy.utils
from nodes import LoraLoader

from .merge.utils import parse_layer_filter, apply_layer_filter


class AnyType(str):
    """A special class that is always equal in not equal comparisons. Credit to pythongosssss"""

    def __ne__(self, __value: object) -> bool:
        return False


class FlexibleOptionalInputType(dict):
    """A special class to make flexible nodes that pass data to our python handlers.

    Enables both flexible/dynamic input types (like for Any Switch) or a dynamic number of inputs
    (like for Any Switch, Context Switch, Context Merge, Power Lora Loader, etc).

    Note, for ComfyUI, all that's needed is the `__contains__` override below, which tells ComfyUI
    that our node will handle the input, regardless of what it is.

    However, with https://github.com/comfyanonymous/ComfyUI/pull/2666 a large change would occur
    requiring more details on the input itself. There, we need to return a list/tuple where the first
    item is the type. This can be a real type, or use the AnyType for additional flexibility.

    This should be forwards compatible unless more changes occur in the PR.
    """

    def __init__(self, type):
        self.type = type

    def __getitem__(self, key):
        return (self.type,)

    def __contains__(self, key):
        return True


any_type = AnyType("*")


def get_lora_by_filename(file_path, log_node="PM LoRA Power Stacker"):
    """Returns a lora by filename, looking for exact paths and then fuzzier matching.

    Adapted from rgthree's power_prompt_utils.py
    """
    lora_paths = folder_paths.get_filename_list('loras')

    if file_path in lora_paths:
        return file_path

    lora_paths_no_ext = [os.path.splitext(x)[0] for x in lora_paths]

    # See if we've entered the exact path, but without the extension
    if file_path in lora_paths_no_ext:
        found = lora_paths[lora_paths_no_ext.index(file_path)]
        return found

    # Same check, but ensure file_path is without extension.
    file_path_force_no_ext = os.path.splitext(file_path)[0]
    if file_path_force_no_ext in lora_paths_no_ext:
        found = lora_paths[lora_paths_no_ext.index(file_path_force_no_ext)]
        return found

    # See if we passed just the name, without paths.
    lora_filenames_only = [os.path.basename(x) for x in lora_paths]
    if file_path in lora_filenames_only:
        found = lora_paths[lora_filenames_only.index(file_path)]
        print(f'[{log_node}] Matched Lora input "{file_path}" to "{found}".')
        return found

    # Same, but force the input to be without paths
    file_path_force_filename = os.path.basename(file_path)
    lora_filenames_only = [os.path.basename(x) for x in lora_paths]
    if file_path_force_filename in lora_filenames_only:
        found = lora_paths[lora_filenames_only.index(file_path_force_filename)]
        print(f'[{log_node}] Matched Lora input "{file_path}" to "{found}".')
        return found

    # Check the filenames and without extension.
    lora_filenames_and_no_ext = [os.path.splitext(os.path.basename(x))[0] for x in lora_paths]
    if file_path in lora_filenames_and_no_ext:
        found = lora_paths[lora_filenames_and_no_ext.index(file_path)]
        print(f'[{log_node}] Matched Lora input "{file_path}" to "{found}".')
        return found

    # And, one last forcing the input to be the same
    file_path_force_filename_and_no_ext = os.path.splitext(os.path.basename(file_path))[0]
    if file_path_force_filename_and_no_ext in lora_filenames_and_no_ext:
        found = lora_paths[lora_filenames_and_no_ext.index(file_path_force_filename_and_no_ext)]
        print(f'[{log_node}] Matched Lora input "{file_path}" to "{found}".')
        return found

    # Finally, super fuzzy, we'll just check if the input exists in the path at all.
    for index, lora_path in enumerate(lora_paths):
        if file_path in lora_path:
            found = lora_paths[index]
            print(f'[{log_node}] Fuzzy-matched Lora input "{file_path}" to "{found}".')
            return found

    print(f'[{log_node}] WARNING: Lora "{file_path}" not found, skipping.')
    return None


class LoraPowerStacker:
    """The Power LoRA Stacker is a flexible widget-based node to stack multiple LoRAs.

    Similar to rgthree's Power Lora Loader but outputs LoRAKeyDict and LoRAStrengths
    for use with PM LoRA PowerMerge workflow.
    """

    NAME = "PM LoRA Power Stacker"
    CATEGORY = "LoRA PowerMerge"
    DESCRIPTION = """Widget-based LoRA stacker for PowerMerge workflow.

Outputs:
- LoRAStack: Processed model weights (filtered by layer_filter)
- LoRAWeights: Strength metadata for each LoRA
- LoRARawDict: Original raw state dicts (preserves CLIP weights)
- CLIP: Modified CLIP model with all LoRAs applied

Use the widget to add/remove LoRAs dynamically. Connect LoRARawDict to LoRA Select to preserve CLIP weights when saving merged LoRAs."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The diffusion model the LoRA will be applied to."}),
                "clip": ("CLIP", {"tooltip": "The CLIP model the LoRA will be applied to."}),
            },
            "optional": FlexibleOptionalInputType(any_type),
            "hidden": {},
        }

    RETURN_TYPES = ("LoRAStack", "LoRAWeights", "LoRARawDict", "CLIP")
    RETURN_NAMES = ("LoRAStack", "LoRAWeights", "LoRARawDict", "CLIP")
    FUNCTION = "stack_loras_widget"

    def stack_loras_widget(self, model, clip, **kwargs):
        """Loops over the provided loras in kwargs and builds stack outputs."""

        # Extract layer_filter if provided (comes from widget, not LoRA data)
        layer_filter = kwargs.pop("layer_filter", "full")
        layer_filter = parse_layer_filter(layer_filter)

        # Build key_map for LoRA loading
        key_map = {}
        if model is not None:
            key_map = comfy.lora.model_lora_keys_unet(model.model, key_map)

        # Initialize outputs
        lora_patch_dicts = {}
        lora_strengths = {}
        lora_raw_dicts = {}  # Store raw LoRA state dicts for CLIP weights

        # Track how many LoRAs were loaded
        loaded_count = 0

        # Loop through kwargs looking for LoRA widgets
        for key, value in kwargs.items():
            key_upper = key.upper()

            # Check if this is a LoRA widget (must have required fields)
            if (key_upper.startswith('LORA_') and
                isinstance(value, dict) and
                'on' in value and
                'lora' in value and
                'strength' in value):

                # Extract values
                is_on = value.get('on', False)
                lora_name = value.get('lora')
                strength_model = value.get('strength', 1.0)

                # Handle separate model/clip strengths
                # If strengthTwo exists and is not None, use it for clip
                # Otherwise use strength for both model and clip
                strength_clip = value.get('strengthTwo')
                if strength_clip is None:
                    strength_clip = strength_model

                # Skip if disabled or strength is zero
                if not is_on or (strength_model == 0 and strength_clip == 0):
                    continue

                # Skip if no LoRA specified
                if not lora_name or lora_name == "None":
                    continue

                # Find the LoRA file using fuzzy matching
                lora_file = get_lora_by_filename(lora_name, log_node=self.NAME)
                if lora_file is None:
                    continue

                # Load the LoRA
                try:
                    lora_path = folder_paths.get_full_path("loras", lora_file)
                    lora_raw = comfy.utils.load_torch_file(lora_path, safe_load=True)

                    # Get pretty name (without extension)
                    lora_name_pretty = os.path.splitext(os.path.basename(lora_file))[0]

                    # Load LoRA into patch dict
                    patch_dict = comfy.lora.load_lora(lora_raw, key_map)

                    # Apply layer filter
                    patch_dict = apply_layer_filter(patch_dict, layer_filter)

                    # Store in outputs
                    lora_patch_dicts[lora_name_pretty] = patch_dict
                    lora_strengths[lora_name_pretty] = {
                        'strength_model': strength_model,
                    }
                    lora_raw_dicts[lora_name_pretty] = lora_raw  # Store raw state dict

                    # Apply to CLIP
                    # Note: We need a dummy model for LoraLoader, but we only care about CLIP output
                    # So we'll use the standard LoraLoader node's load_lora method
                    _, clip = LoraLoader().load_lora(model, clip, lora_file, strength_model, strength_clip)

                    loaded_count += 1

                except Exception as e:
                    print(f"[{self.NAME}] Error loading LoRA '{lora_name}': {e}")
                    continue

        print(f"[{self.NAME}] Loaded {loaded_count} LoRAs")

        return (lora_patch_dicts, lora_strengths, lora_raw_dicts, clip)
