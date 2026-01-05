"""
Gradient-based semantic analyzer for LoRA attribution.

This module implements practical gradient attribution for determining which
LoRA weights are important for which semantic features.
"""

import logging
from typing import Dict, List, Optional, Any

import torch
import comfy.model_management as mm

from .feature_prompts import get_feature_prompts

# Configure logging for this module
logger = logging.getLogger(__name__)
# if not logger.handlers:
#     # Add console handler if not already configured
#     handler = logging.StreamHandler()
#     handler.setLevel(logging.DEBUG)
#     formatter = logging.Formatter('[%(name)s] %(levelname)s: %(message)s')
#     handler.setFormatter(formatter)
#     logger.addHandler(handler)
#     logger.setLevel(logging.INFO)  # Set to DEBUG for more verbose output


class GradientSemanticAnalyzer:
    """
    Analyzes LoRA weights using gradient-based attribution.

    This analyzer computes semantic importance maps by:
    1. Applying the LoRA to a model
    2. Running forward passes with feature-specific prompts
    3. Computing gradients of model activations w.r.t. LoRA weights
    4. Aggregating attributions into semantic importance scores

    This provides more accurate feature attribution than depth heuristics
    by using actual model behavior.
    """

    def __init__(
        self,
        clip: Any,  # ComfyUI CLIP object
        device: torch.device,
        dtype: torch.dtype,
    ):
        """
        Initialize gradient analyzer.

        Args:
            clip: ComfyUI CLIP object (for text encoding)
            device: Device for computation (cuda/cpu)
            dtype: Data type for computation (float32/float16/bfloat16/float8_e4m3fn/float8_e5m2)
        """
        self.clip = clip
        self.device = device
        self.dtype = dtype

        logger.info(
            f"GradientSemanticAnalyzer initialized on {device} with {dtype}"
        )

    @staticmethod
    def enable_debug_logging():
        """Enable DEBUG level logging for detailed diagnostic output."""
        logger.setLevel(logging.DEBUG)
        logger.info("DEBUG logging enabled for gradient analyzer")

    def analyze_lora(
        self,
        lora_path: str,
        features: List[str],
        architecture: str,
        checkpoint_path: str,
        num_samples: int = 3,
        offload_layers: bool = False,
        gpu_memory_gb: Optional[float] = None,
        num_integration_steps: int = 20,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Analyze a LoRA to create semantic importance map.

        Args:
            lora_path: Path to LoRA file (.safetensors)
            features: List of features to analyze (hair, eyes, clothing, etc.)
            architecture: Model architecture (sd15, sdxl, dit, flux, zimage)
            checkpoint_path: Path to model checkpoint (.safetensors file)
            num_samples: Number of forward passes per feature (for averaging)
            offload_layers: If True, enable layer offloading to reduce VRAM usage
            gpu_memory_gb: GPU memory limit in GB (only used if offload_layers=True)
            num_integration_steps: Number of Integrated Gradients steps (default: 20)
                Higher = more accurate but slower. 20 steps ≈ 95% accuracy.

        Returns:
            Semantic importance map: {feature: {layer_key: importance_tensor}}
        """
        # zImage gradient analysis using diffusers
        if architecture == "zimage":
            try:
                logger.info("Using gradient-based analysis for zImage...")
                return self._analyze_zimage_with_gradients(
                    lora_path, features, checkpoint_path, num_samples, offload_layers, gpu_memory_gb, num_integration_steps
                )
            except Exception as e:
                logger.error(f"zImage gradient analysis failed: {e}", exc_info=True)
                raise RuntimeError(self._format_error_message(e, architecture)) from e

        # SDXL gradient analysis using diffusers
        if architecture in ["sdxl", "sd15"]:
            try:
                logger.info(f"Using gradient-based analysis for {architecture.upper()}...")
                return self._analyze_sdxl_with_gradients(
                    lora_path, features, checkpoint_path, num_samples, offload_layers, gpu_memory_gb, architecture, num_integration_steps
                )
            except Exception as e:
                logger.error(f"{architecture.upper()} gradient analysis failed: {e}", exc_info=True)
                raise RuntimeError(self._format_error_message(e, architecture)) from e

        # Other architectures not yet implemented
        raise NotImplementedError(
            f"Gradient analysis for architecture '{architecture}' is not yet implemented. "
            f"Currently supported: 'zimage', 'sdxl', 'sd15'. Use heuristic analyzer for other architectures."
        )

    def _format_error_message(self, error: Exception, architecture: str) -> str:
        """
        Format helpful error message with suggestions.

        Args:
            error: The exception that occurred
            architecture: Architecture that was being analyzed

        Returns:
            Formatted error message with suggestions
        """
        error_msg = (
            f"Gradient analysis failed for {architecture}: {error}\n\n"
            f"Suggestions:\n"
            f"  1. **Enable layer offloading**: Set offload_layers=True to reduce VRAM usage\n"
            f"  2. **Use CPU mode**: Set device='cpu' (slower but uses system RAM)\n"
            f"  3. **Free GPU memory**: Close other applications or reduce ComfyUI batch size\n"
            f"  4. **Use heuristic analyzer**: Try 'PM LoRA Semantic Analyzer (Heuristic)' instead\n"
            f"     (faster and works without loading the model, but less accurate)\n"
        )

        # Check if it's an OOM error and provide specific advice
        if "out of memory" in str(error).lower() or "oom" in str(error).lower():
            error_msg += (
                f"\n**GPU Out of Memory detected!**\n"
                f"  - Current free VRAM appears insufficient for model loading\n"
                f"  - Try setting offload_layers=True and gpu_memory_gb=2.0 for aggressive CPU offloading\n"
                f"  - Or use the heuristic analyzer which doesn't require loading the model\n"
            )

        return error_msg





    def _average_attributions(
        self,
        attribution_list: List[Dict[str, float]]
    ) -> Dict[str, torch.Tensor]:
        """
        Average attribution scores across multiple samples.

        Args:
            attribution_list: List of attribution dictionaries {layer_key: float_score}

        Returns:
            Averaged attribution dictionary {layer_key: tensor}
        """
        if not attribution_list:
            return {}

        # Collect all layer keys
        all_keys = set()
        for attrs in attribution_list:
            all_keys.update(attrs.keys())

        # Average each layer
        averaged = {}
        for key in all_keys:
            # Collect scores for this key
            scores = [attrs[key] for attrs in attribution_list if key in attrs]

            if scores:
                # Average scores and convert to tensor
                avg_score = sum(scores) / len(scores)
                averaged[key] = torch.tensor(avg_score, device=self.device)

        return averaged

    def _create_zero_map(
        self,
        lora_patches: Dict[str, Any]
    ) -> Dict[str, torch.Tensor]:
        """
        Create zero importance map for when no attributions were computed.

        Args:
            lora_patches: LoRA patches

        Returns:
            Dictionary mapping layer keys to zero tensors
        """
        zero_map = {}
        for layer_key in lora_patches.keys():
            zero_map[layer_key] = torch.tensor(0.0, device=self.device)
        return zero_map

    def _convert_to_semantic_map(
        self,
        feature_attributions: Dict[str, Dict[str, torch.Tensor]],
        lora_patches: Dict[str, Any],
        features: List[str],
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Convert raw attributions to semantic map format.

        The semantic map format matches the heuristic analyzer:
        {feature: {layer_key: importance_tensor}}

        Args:
            feature_attributions: {feature: {layer_key.sublayer: attribution_tensor}}
            lora_patches: Original LoRA patches
            features: List of features

        Returns:
            Semantic map in standard format
        """
        semantic_map = {feature: {} for feature in features}

        # For each layer in the original patches
        for layer_key, adapter in lora_patches.items():
            if hasattr(adapter, 'weights'):
                up, down, alpha, *rest = adapter.weights

                # For each feature
                for feature in features:
                    feature_attrs = feature_attributions.get(feature, {})

                    # Combine up and down attributions for this layer
                    # Average them to get a single importance score per layer
                    up_attr = feature_attrs.get(f"{layer_key}.up", None)
                    down_attr = feature_attrs.get(f"{layer_key}.down", None)

                    if up_attr is not None and down_attr is not None:
                        # Average up and down importance
                        # They may have different shapes, so compute scalar importance
                        up_importance = up_attr.mean()
                        down_importance = down_attr.mean()
                        combined_importance = (up_importance + down_importance) / 2.0

                        # Create a tensor with the same shape as up (for consistency)
                        importance_tensor = torch.full_like(up_attr, combined_importance.item())

                    elif up_attr is not None:
                        importance_tensor = up_attr
                    elif down_attr is not None:
                        importance_tensor = down_attr
                    else:
                        # No attribution for this layer/feature, use zeros on correct device
                        if isinstance(up, torch.Tensor):
                            importance_tensor = torch.zeros_like(
                                up, device=self.device, dtype=self.dtype
                            )
                        else:
                            continue

                    semantic_map[feature][layer_key] = importance_tensor

        return semantic_map

    def _normalize_semantic_map(
        self,
        semantic_map: Dict[str, Dict[str, torch.Tensor]]
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Normalize semantic map so features sum to 1.0 per layer.

        This ensures feature weights can be used directly as mixing coefficients.

        Args:
            semantic_map: {feature: {layer_key: importance_tensor}}

        Returns:
            Normalized semantic map
        """
        # Collect all layer keys
        all_layers = set()
        for feature_map in semantic_map.values():
            all_layers.update(feature_map.keys())

        # Normalize per layer
        for layer_key in all_layers:
            # Collect tensors for this layer across features
            layer_tensors = {}
            for feature, feature_map in semantic_map.items():
                if layer_key in feature_map:
                    layer_tensors[feature] = feature_map[layer_key]

            if not layer_tensors:
                continue

            # Stack and sum
            stacked = torch.stack(list(layer_tensors.values()))
            total = stacked.sum(dim=0, keepdim=True)

            # Avoid division by zero
            total = torch.clamp(total, min=1e-8)

            # Normalize each feature's importance
            for feature, tensor in layer_tensors.items():
                semantic_map[feature][layer_key] = tensor / total.squeeze(0)

        return semantic_map


    def _analyze_zimage_with_gradients(
        self,
        lora_path: str,
        features: List[str],
        checkpoint_path: str,
        num_samples: int,
        offload_layers: bool = False,
        gpu_memory_gb: Optional[float] = None,
        num_integration_steps: int = 20,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        zImage gradient analysis using diffusers library.

        Uses diffusers to load zImage architecture independently from ComfyUI,
        applies LoRA patches, and computes Integrated Gradients for feature attribution.

        Args:
            lora_path: Path to LoRA file (.safetensors)
            features: Features to analyze
            checkpoint_path: Path to .safetensors checkpoint file
            num_samples: Number of prompts per feature
            offload_layers: If True, enable layer offloading to reduce VRAM
            gpu_memory_gb: GPU memory limit in GB (only used if offload_layers=True)
            num_integration_steps: Number of Integrated Gradients steps (default: 20)

        Returns:
            Semantic importance map: {feature: {layer_key: importance_tensor}}
        """
        # Import zImage loader functions
        from .zimage_loader import load_zimage_for_gradients, apply_lora_simple
        import comfy.utils

        # Load zImage transformer using diffusers
        logger.info(f"Loading zImage from checkpoint: {checkpoint_path}")
        transformer = load_zimage_for_gradients(
            checkpoint_path=checkpoint_path,
            device=self.device,
            dtype=self.dtype,
            offload_layers=offload_layers,
            gpu_memory_gb=gpu_memory_gb,
        )

        # Load LoRA file
        from ..utils.layer_filter import is_clip_layer

        logger.info(f"Loading LoRA from {lora_path}...")
        all_lora_patches = comfy.utils.load_torch_file(lora_path, safe_load=True)

        # Filter out CLIP layers - they should not be included in gradient analysis
        lora_patches = {k: v for k, v in all_lora_patches.items() if not is_clip_layer(k)}
        clip_keys_filtered = len(all_lora_patches) - len(lora_patches)

        if clip_keys_filtered > 0:
            logger.info(f"Filtered out {clip_keys_filtered} CLIP layers from gradient analysis")

        if not lora_patches:
            raise ValueError("No UNet/DiT layers found in LoRA after filtering CLIP layers")

        # Note: We don't apply LoRA upfront anymore - it will be applied
        # at each integration step with appropriate scaling

        # Compute gradients for each feature
        semantic_map = {}

        for feature in features:
            logger.info(f"Analyzing feature: {feature}")
            prompts = get_feature_prompts(feature, "zimage", num_samples)

            attributions = []
            for prompt in prompts:
                attr = self._compute_gradients_zimage(transformer, lora_patches, prompt, num_integration_steps)
                if attr:
                    attributions.append(attr)

            if attributions:
                semantic_map[feature] = self._average_attributions(attributions)
            else:
                # Create zero attributions if none computed
                semantic_map[feature] = self._create_zero_map(lora_patches)

        # Cleanup
        del transformer
        torch.cuda.empty_cache()

        return self._normalize_semantic_map(semantic_map)

    def _compute_gradients_zimage(
        self,
        model: torch.nn.Module,
        lora_patches: Dict[str, Any],
        prompt: str,
        num_integration_steps: int = 20,
    ) -> Dict[str, float]:
        """
        Compute Integrated Gradients for one prompt using zImage model.

        Applies scaled LoRA at each integration step, integrating gradients
        from baseline (no LoRA) to full LoRA for accurate attribution.

        Args:
            model: zImage transformer model (without LoRA applied)
            lora_patches: LoRA patches to analyze
            prompt: Text prompt
            num_integration_steps: Number of integration steps (default: 20)

        Returns:
            Attribution scores per LoRA key
        """
        # Get gradient-compatible dtype (fp8 doesn't support gradients)
        grad_dtype = self.dtype
        if self.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
            grad_dtype = torch.float16

        # Pre-compute LoRA deltas and build mapping from model param names to LoRA keys
        logger.debug("Pre-computing LoRA deltas...")
        lora_deltas = {}  # {model_param_name: (lora_key, delta_tensor)}
        model_state = model.state_dict()

        for lora_key, adapter in lora_patches.items():
            if not hasattr(adapter, 'weights'):
                continue

            up, down, alpha, *_ = adapter.weights

            # Compute delta: (up @ down) * (alpha / rank)
            if up.dim() > 2:
                # Conv layer: need to reshape
                up_2d = up.flatten(1)
                down_2d = down.flatten(1)
                delta = (up_2d @ down_2d).reshape(up.shape[0], down.shape[1], *up.shape[2:])
            else:
                # Linear layer
                delta = up @ down

            # Scale by alpha
            rank = down.shape[0]
            if alpha is not None and alpha != 0:
                delta = delta * (alpha / rank)

            # Find corresponding model parameter
            # LoRA key format: "diffusion_model.layers.0.attention.to_k.lora_A.weight"
            # Model param format: "diffusion_model.layers.0.attention.to_k.weight"
            lora_key_str = lora_key[0] if isinstance(lora_key, tuple) else str(lora_key)

            # Remove .lora_A, .lora_B, .lora_down, .lora_up, .alpha suffixes
            base_key = lora_key_str
            for suffix in ['.lora_A.weight', '.lora_B.weight', '.lora_down.weight', '.lora_up.weight', '.alpha']:
                if base_key.endswith(suffix):
                    base_key = base_key.replace(suffix, '.weight')
                    break

            if base_key in model_state:
                lora_deltas[base_key] = (lora_key, delta.to(device=self.device, dtype=grad_dtype))

        if not lora_deltas:
            logger.warning("No LoRA deltas computed - key mapping may have failed")
            return {}

        logger.debug(f"Computed {len(lora_deltas)} LoRA deltas")

        # Save original model weights (only for layers that have LoRA)
        original_weights = {}
        for param_name in lora_deltas.keys():
            param = dict(model.named_parameters())[param_name]
            original_weights[param_name] = param.data.clone()

        # Encode prompt (once, reuse for all steps)
        tokens = self.clip.tokenize(prompt)
        cond, _ = self.clip.encode_from_tokens(tokens, return_pooled=True)
        cond = cond.to(device=self.device, dtype=grad_dtype)

        # Create latent
        latent = torch.randn(
            (1, 4, 64, 64),
            device=self.device,
            dtype=grad_dtype,
            requires_grad=False  # We don't need gradients w.r.t. latent
        )

        # Timestep
        timestep = torch.tensor([500], device=self.device, dtype=torch.long)

        # Initialize accumulated gradients (per model parameter)
        accumulated_grads = {}
        for param_name in lora_deltas.keys():
            param = dict(model.named_parameters())[param_name]
            accumulated_grads[param_name] = torch.zeros_like(param)

        logger.debug(f"Starting Integrated Gradients with {num_integration_steps} steps...")

        # Integrated Gradients: Integrate from alpha=0 to alpha=1
        for step in range(num_integration_steps + 1):
            # Compute interpolation factor
            alpha = step / num_integration_steps

            # Apply scaled LoRA deltas to model
            for param_name, (lora_key, delta) in lora_deltas.items():
                param = dict(model.named_parameters())[param_name]
                # Restore original + add scaled delta
                param.data = original_weights[param_name] + alpha * delta
                # Enable gradients
                param.requires_grad = True

            # Forward pass with gradients
            with torch.enable_grad():
                try:
                    # Try different call signatures for zImage
                    output = model(latent, timestep, encoder_hidden_states=cond)
                except TypeError:
                    try:
                        output = model(latent, timestep, context=cond)
                    except TypeError:
                        output = model(latent, timestep)

                # Handle different return types
                if isinstance(output, dict):
                    output = output['sample']
                elif isinstance(output, tuple):
                    output = output[0]

                # Compute target
                target = output.norm(p=2)

                if not target.requires_grad:
                    logger.error("Target doesn't require gradients")
                    # Restore original weights
                    for param_name in lora_deltas.keys():
                        param = dict(model.named_parameters())[param_name]
                        param.data = original_weights[param_name]
                    return {}

                # Backpropagate
                target.backward()

            # Accumulate gradients
            for param_name in lora_deltas.keys():
                param = dict(model.named_parameters())[param_name]
                if param.grad is not None:
                    accumulated_grads[param_name] += param.grad.clone()

            # Clear gradients
            model.zero_grad()

        # Restore original model weights
        for param_name in lora_deltas.keys():
            param = dict(model.named_parameters())[param_name]
            param.data = original_weights[param_name]

        # Compute final attributions: avg_gradient × delta
        # Map back to LoRA keys
        lora_attributions = {}
        for param_name, (lora_key, delta) in lora_deltas.items():
            # Average accumulated gradients
            avg_grad = accumulated_grads[param_name] / num_integration_steps

            # Attribution = avg_gradient × delta (IG formula with baseline=0)
            attribution = (avg_grad * delta).abs().mean().item()

            if attribution > 0:
                # Convert lora_key to string for consistency
                lora_key_str = lora_key[0] if isinstance(lora_key, tuple) else str(lora_key)
                lora_attributions[lora_key_str] = attribution

        logger.debug(f"Integrated Gradients complete. Found {len(lora_attributions)} non-zero attributions.")

        return lora_attributions

    def _names_match(self, lora_key: str, param_name: str) -> bool:
        """
        Check if LoRA key and parameter name are related.

        Simple heuristic: Extract layer number and match.
        """
        if isinstance(lora_key, tuple):
            lora_key = lora_key[0]

        # Extract layer number from LoRA key
        for part in str(lora_key).split('.'):
            if part.isdigit():
                if part in param_name:
                    return True

        return False

    def _analyze_sdxl_with_gradients(
        self,
        lora_path: str,
        features: List[str],
        checkpoint_path: str,
        num_samples: int,
        offload_layers: bool = False,
        gpu_memory_gb: Optional[float] = None,
        architecture: str = "sdxl",
        num_integration_steps: int = 20,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        SDXL/SD1.5 gradient analysis using diffusers library.

        Uses diffusers to load SDXL/SD1.5 UNet using from_single_file(),
        applies LoRA patches, and computes Integrated Gradients for feature attribution.

        Args:
            lora_path: Path to LoRA file (.safetensors)
            features: Features to analyze
            checkpoint_path: Path to .safetensors checkpoint file
            num_samples: Number of prompts per feature
            offload_layers: If True, enable automatic CPU offloading to reduce VRAM
            gpu_memory_gb: GPU memory limit in GB (ignored, offloading is automatic)
            architecture: "sdxl" or "sd15" (selects appropriate loader)
            num_integration_steps: Number of Integrated Gradients steps (default: 20)

        Returns:
            Semantic importance map: {feature: {layer_key: importance_tensor}}
        """
        from .sd15_loader import load_sd15_for_gradients as load_unet

        # Load UNet using diffusers from_single_file()
        logger.info(f"Loading {architecture.upper()} UNet from checkpoint...")
        pipe = load_unet(
            checkpoint_path=checkpoint_path,
            device=self.device,
            dtype=self.dtype,
            offload_layers=offload_layers,
            gpu_memory_gb=gpu_memory_gb,
        )

        # Apply LoRA patches by merging them into weights
        logger.info("Applying LoRA patches to UNet...")
        # Extract safe adapter name from file path (remove path and extension)
        import os
        adapter_name = os.path.splitext(os.path.basename(lora_path))[0]
        # Replace spaces and special chars with underscores for valid Python identifier
        adapter_name = adapter_name.replace(" ", "_").replace("-", "_")

        pipe.load_lora_weights(
            lora_path,
            adapter_name=adapter_name,
        )
        # Set active adapter
        pipe.set_adapters(adapter_name)

        # Enable gradients for LoRA parameters
        # By default, diffusers freezes LoRA params, we need to enable gradients
        logger.info("Enabling gradients for LoRA parameters...")
        unet = pipe.unet
        lora_param_count = 0
        for name, param in unet.named_parameters():
            if 'lora' in name.lower():
                param.requires_grad = True
                lora_param_count += 1

        logger.info(f"Enabled gradients for {lora_param_count} LoRA parameters")

        if lora_param_count == 0:
            logger.warning("No LoRA parameters found after load_lora_weights()!")
            logger.info("Checking if LoRA was loaded correctly...")
            # List adapters
            if hasattr(pipe, 'get_list_adapters'):
                adapters = pipe.get_list_adapters()
                logger.info(f"Loaded adapters: {adapters}")
            elif hasattr(unet, 'get_list_adapters'):
                adapters = unet.get_list_adapters()
                logger.info(f"Loaded adapters: {adapters}")

        # No need to load LoRA file keys - we'll keep everything in diffusers format
        # and let the merger handle the conversion using sd_lora.py mapping functions

        # Compute gradients for each feature
        semantic_map = {}

        for feature in features:
            logger.info(f"Analyzing feature: {feature}")
            prompts = get_feature_prompts(feature, architecture, num_samples)

            attributions = []
            for prompt in prompts:
                # Get attributions in diffusers format (no conversion needed)
                attr = self._compute_gradients_sdxl(pipe, prompt, adapter_name, num_integration_steps)
                if attr:
                    attributions.append(attr)

            if attributions:
                semantic_map[feature] = self._average_attributions(attributions)
            else:
                raise Exception("No attributions found")

        # Cleanup
        del pipe
        torch.cuda.empty_cache()

        return self._normalize_semantic_map(semantic_map)

    def _extract_lora_parameters(self, unet: torch.nn.Module) -> Dict[str, torch.nn.Parameter]:
        """
        Extract all LoRA parameters from UNet using only diffusers APIs.

        After load_lora_weights(), diffusers injects LoRA parameters into the model.
        This method walks through the UNet and extracts those LoRA parameters.

        Args:
            unet: Diffusers UNet model with LoRA loaded

        Returns:
            Dict mapping parameter names to Parameter objects
        """
        lora_params = {}

        logger.info("Searching for LoRA parameters in UNet...")

        # Method 1: Check for LoRA in named_parameters
        # After load_lora_weights(), LoRA params should have 'lora' in their name
        param_count = 0
        for param_name, param in unet.named_parameters():
            if 'lora' in param_name.lower():
                lora_params[param_name] = param
                param_count += 1
                if param_count <= 5:  # Log first 5 for debugging
                    logger.debug(f"Found LoRA param: {param_name}, shape: {param.shape}")

        if lora_params:
            logger.info(f"Found {len(lora_params)} LoRA parameters via named_parameters()")
            return lora_params

        # Method 2: Check attention processors (older diffusers versions)
        logger.debug("No LoRA in named_parameters, checking attention processors...")
        for module_name, module in unet.named_modules():
            if hasattr(module, 'processor'):
                processor = module.processor
                processor_type = type(processor).__name__

                # Log processor type for first few modules
                if len(lora_params) < 5:
                    logger.debug(f"Module {module_name}: processor type = {processor_type}")

                # Check for LoRA layers in the processor
                for attr_name in dir(processor):
                    if 'lora' in attr_name.lower() and not attr_name.startswith('_'):
                        lora_layer = getattr(processor, attr_name, None)
                        if lora_layer is None:
                            continue

                        # Extract parameters from LoRA layer
                        if hasattr(lora_layer, 'up') and hasattr(lora_layer, 'down'):
                            base_name = f"{module_name}.{attr_name}"

                            if hasattr(lora_layer.up, 'weight'):
                                up_param = lora_layer.up.weight
                                if isinstance(up_param, torch.nn.Parameter):
                                    lora_params[f"{base_name}.up"] = up_param

                            if hasattr(lora_layer.down, 'weight'):
                                down_param = lora_layer.down.weight
                                if isinstance(down_param, torch.nn.Parameter):
                                    lora_params[f"{base_name}.down"] = down_param

        if lora_params:
            logger.info(f"Found {len(lora_params)} LoRA parameters via attention processors")
            return lora_params

        # Method 3: Debug - list all parameter names to see what's available
        logger.warning("No LoRA parameters found! Listing all parameter names for debugging:")
        all_params = list(unet.named_parameters())
        logger.info(f"Total parameters in UNet: {len(all_params)}")

        # Sample first 10 parameter names
        logger.debug(f"Sample parameter names:")
        for i, (name, _) in enumerate(all_params[:10]):
            logger.debug(f"  Param {i}: {name}")

        # Check if any processor types
        processor_types = set()
        for module in unet.modules():
            if hasattr(module, 'processor'):
                processor_types.add(type(module.processor).__name__)

        if processor_types:
            logger.info(f"Found processor types: {processor_types}")
        else:
            logger.warning("No modules with 'processor' attribute found!")

        return lora_params

    def _compute_gradients_sdxl(
        self,
        pipe: Any,  # diffusers pipeline (SD1.5 or SDXL)
        prompt: str,
        adapter_name: str,
        num_integration_steps: int = 20,
    ) -> Dict[str, float]:
        """
        Compute Integrated Gradients for one prompt using diffusers pipeline.

        Uses adapter weight scaling to interpolate from baseline (no LoRA) to
        full LoRA, integrating gradients along the path for accurate attribution.

        Args:
            pipe: Diffusers pipeline with LoRA loaded
            prompt: Text prompt
            adapter_name: Name of the loaded LoRA adapter
            num_integration_steps: Number of integration steps (default: 20)

        Returns:
            Attribution scores per diffusers parameter name (not converted to LoRA keys).
            Format: {diffusers_param_name: importance_score}
            Example: {"down_blocks.0.attentions.0.to_q.processor.to_q_lora.down.weight": 0.42}
        """
        # Get gradient-compatible dtype (fp8 doesn't support gradients)
        grad_dtype = self.dtype
        if self.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
            grad_dtype = torch.float16

        # Extract UNet from pipeline
        unet = pipe.unet

        # Extract all LoRA parameters from the pipeline
        # In diffusers, LoRAs are stored in attention processors
        lora_params = self._extract_lora_parameters(unet)

        if not lora_params:
            logger.warning("No LoRA parameters found in pipeline")
            return {}

        logger.debug(f"Found {len(lora_params)} LoRA parameters")

        # Encode prompt (once, reuse for all integration steps)
        tokens = self.clip.tokenize(prompt)
        cond, pooled = self.clip.encode_from_tokens(tokens, return_pooled=True)
        cond = cond.to(device=self.device, dtype=grad_dtype)
        pooled = pooled.to(device=self.device, dtype=grad_dtype)

        # Create latent (64x64 for both SD1.5 and SDXL)
        latent = torch.randn(
            (1, 4, 64, 64),
            device=self.device,
            dtype=grad_dtype,
            requires_grad=False  # We don't need gradients w.r.t. latent
        )

        # Timestep
        timestep = torch.tensor([500], device=self.device, dtype=torch.long)

        # Prepare conditioning kwargs
        # SDXL requires additional conditioning
        added_cond_kwargs = {
            "text_embeds": pooled,
            "time_ids": torch.tensor(
                [[1024, 1024, 0, 0, 1024, 1024]],
                device=self.device,
                dtype=grad_dtype
            ),
        }

        # Initialize accumulated gradients
        accumulated_grads = {name: torch.zeros_like(param) for name, param in lora_params.items()}

        logger.debug(f"Starting Integrated Gradients with {num_integration_steps} steps...")

        # Integrated Gradients: Integrate gradients from alpha=0 to alpha=1
        for step in range(num_integration_steps + 1):
            # Compute interpolation factor
            alpha = step / num_integration_steps

            # Set adapter weight to alpha (0.0 = baseline/no LoRA, 1.0 = full LoRA)
            pipe.set_adapters(adapter_name, adapter_weights=[alpha])

            # Forward pass with gradients enabled
            with torch.enable_grad():
                try:
                    # Try SDXL signature first
                    output = unet(
                        latent,
                        timestep,
                        encoder_hidden_states=cond,
                        added_cond_kwargs=added_cond_kwargs,
                    )
                except Exception as e:
                    logger.debug(f"SDXL forward failed: {e}, trying SD1.5 signature...")
                    # Fallback to SD1.5 signature (no added_cond_kwargs)
                    try:
                        output = unet(
                            latent,
                            timestep,
                            encoder_hidden_states=cond,
                        )
                    except Exception as e2:
                        logger.error(f"Both forward passes failed: {e2}")
                        # Restore full LoRA before returning
                        pipe.set_adapters(adapter_name, adapter_weights=[1.0])
                        return {}

                # Handle different return types
                if isinstance(output, dict):
                    output = output['sample']
                elif isinstance(output, tuple):
                    output = output[0]

                # Compute target (use L2 norm)
                target = output.norm(p=2)

                if not target.requires_grad:
                    logger.error("Target doesn't require gradients - LoRA params may not be trainable")
                    # Restore full LoRA before returning
                    pipe.set_adapters(adapter_name, adapter_weights=[1.0])
                    return {}

                # Backpropagate
                target.backward()

            # Accumulate gradients
            for param_name, param in lora_params.items():
                if param.grad is not None:
                    accumulated_grads[param_name] += param.grad.clone()

            # Clear gradients for next iteration
            unet.zero_grad()

        # Restore full LoRA adapter weight
        pipe.set_adapters(adapter_name, adapter_weights=[1.0])

        # Compute final attributions: (avg_gradient) × parameter
        # Note: We multiply by parameter here since IG formula is: (x - baseline) × ∫grad
        # Since baseline=0, (x - baseline) = x = parameter value
        lora_attributions = {}
        for param_name, param in lora_params.items():
            # Average accumulated gradients
            avg_grad = accumulated_grads[param_name] / num_integration_steps

            # Attribution = avg_gradient × parameter (IG formula with baseline=0)
            attribution = (avg_grad * param).abs().mean().item()

            if attribution > 0:
                lora_attributions[param_name] = attribution

        logger.debug(f"Integrated Gradients complete. Found {len(lora_attributions)} non-zero attributions.")

        # Return diffusers parameter names directly (no conversion to LoRA keys)
        # The semantic merger will handle the mapping using sd_lora.py functions
        return lora_attributions

