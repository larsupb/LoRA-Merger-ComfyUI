"""
Gradient-based semantic analyzer for LoRA attribution.

This module implements practical gradient attribution for determining which
LoRA weights are important for which semantic features.
"""

import logging
from typing import Dict, List, Optional, Any

import torch

from .feature_prompts import get_feature_prompts
from .feature_timesteps import TimestepConfig
from .sd_loader import load_sd_for_gradients
from ..utils import is_clip_layer

# Configure logging for this module
logger = logging.getLogger(__name__)

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
            num_integration_steps: Number of Integrated Gradients steps (default: 20)
                Higher = more accurate but slower. 20 steps ≈ 95% accuracy.

        Returns:
            Semantic importance map: {feature: {layer_key: importance_tensor}}
        """
        # SDXL gradient analysis using diffusers
        if architecture in ["sdxl", "sd15"]:
            try:
                logger.info(f"Using gradient-based analysis for {architecture.upper()}...")
                return self._analyze_sd_with_gradients(
                    lora_path, features, checkpoint_path, num_samples, architecture, num_integration_steps
                )
            except Exception as e:
                logger.error(f"{architecture.upper()} gradient analysis failed: {e}", exc_info=True)
                raise RuntimeError((
                    f"Gradient analysis failed for {architecture}: {e}\n\n"
                )) from e

        # Other architectures not yet implemented
        raise NotImplementedError(
            f"Gradient analysis for architecture '{architecture}' is not yet implemented. "
            f"Currently supported: 'sdxl', 'sd15'. Use heuristic analyzer for other architectures."
        )

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

    def _analyze_sd_with_gradients(
            self,
            lora_path: str,
            features: List[str],
            checkpoint_path: str,
            num_samples: int,
            architecture: str = "sdxl",
            num_integration_steps: int = 20,
            use_multi_timestep: bool = True,
            use_feature_specific_timesteps: bool = True,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        SDXL/SD1.5 gradient analysis with multi-timestep support.

        Args:
            lora_path: Path to LoRA file
            features: Features to analyze
            checkpoint_path: Path to checkpoint
            num_samples: Prompts per feature
            architecture: "sdxl" or "sd15"
            num_integration_steps: IG steps per timestep
            use_multi_timestep: If True, use 3 timesteps per feature
            use_feature_specific_timesteps: If True, optimize timesteps per feature

        Returns:
            Semantic importance map
        """
        from .feature_timesteps import get_timestep_config
        pipe = load_sd_for_gradients(
            checkpoint_path, self.device, self.dtype
        )

        # Load LoRA
        import os
        adapter_name = os.path.splitext(os.path.basename(lora_path))[0]
        adapter_name = adapter_name.replace(" ", "_").replace("-", "_")

        pipe.load_lora_weights(lora_path, adapter_name=adapter_name)
        pipe.set_adapters(adapter_name)

        # Enable gradients for LoRA params
        unet = pipe.unet
        for name, param in unet.named_parameters():
            if 'lora' in name.lower():
                param.requires_grad = True

        # Compute gradients for each feature
        semantic_map = {}
        total_features = len(features)

        for feat_idx, feature in enumerate(features):
            logger.info(f"Analyzing feature [{feat_idx + 1}/{total_features}]: {feature}")

            # Get timestep configuration for this feature
            timestep_config = get_timestep_config(
                feature=feature,
                use_feature_specific=use_feature_specific_timesteps,
                use_multi_timestep=use_multi_timestep,
            )

            logger.debug(
                f"  Timesteps: {timestep_config.timesteps}, "
                f"Weights: {[f'{w:.2f}' for w in timestep_config.weights]}"
            )

            prompts = get_feature_prompts(feature, architecture, num_samples)
            attributions = []

            for prompt_idx, prompt in enumerate(prompts):
                logger.debug(f"  Prompt [{prompt_idx + 1}/{len(prompts)}]: {prompt[:50]}...")

                attr = self._compute_gradients_sd(
                    pipe=pipe,
                    architecture=architecture,
                    prompt=prompt,
                    adapter_name=adapter_name,
                    num_integration_steps=num_integration_steps,
                    timestep_config=timestep_config,
                )

                if attr:
                    attributions.append(attr)

            if attributions:
                semantic_map[feature] = self._average_attributions(attributions)
                logger.info(
                    f"  Feature '{feature}': {len(semantic_map[feature])} attributed params"
                )
            else:
                logger.warning(f"  No attributions computed for feature '{feature}'")
                semantic_map[feature] = {}

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
            return lora_params

        # Method 2: Check attention processors (older diffusers versions)
        logger.info("No LoRA in named_parameters, checking attention processors...")
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

    def _compute_gradients_sd(
            self,
            pipe: Any,
            architecture: str,
            prompt: str,
            adapter_name: str,
            num_integration_steps: int = 20,
            timestep_config: Optional[TimestepConfig] = None,
    ) -> Dict[str, float]:
        """
        Compute Integrated Gradients for one prompt using diffusers pipeline.

        Supports multi-timestep analysis for more accurate attribution across
        different noise levels.

        Args:
            pipe: Diffusers pipeline with LoRA loaded
            architecture: "sdxl" or "sd15"
            prompt: Text prompt
            adapter_name: Name of the loaded LoRA adapter
            num_integration_steps: Number of integration steps per timestep
            timestep_config: Timestep configuration (defaults to single t=500)

        Returns:
            Attribution scores per diffusers parameter name
        """
        from .feature_timesteps import SINGLE_TIMESTEP

        # Default to single timestep for backward compatibility
        if timestep_config is None:
            timestep_config = SINGLE_TIMESTEP

        # Get gradient-compatible dtype
        grad_dtype = self.dtype
        if self.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
            grad_dtype = torch.float16

        # Extract UNet and LoRA parameters
        unet = pipe.unet
        lora_params = self._extract_lora_parameters(unet)
        lora_params = {k: v for k, v in lora_params.items() if not is_clip_layer(k)}

        if not lora_params:
            logger.warning("No LoRA parameters found in pipeline")
            return {}

        logger.debug(f"Found {len(lora_params)} LoRA parameters")

        # Encode prompt once (reused for all timesteps)
        tokens = self.clip.tokenize(prompt)
        cond, pooled = self.clip.encode_from_tokens(tokens, return_pooled=True)
        cond = cond.to(device=self.device, dtype=grad_dtype)
        pooled = pooled.to(device=self.device, dtype=grad_dtype)

        # Create latent
        if architecture == "sdxl":
            latent_size = 128
        else:
            latent_size = 64

        latent = torch.randn(
            (1, 4, latent_size, latent_size),
            device=self.device,
            dtype=grad_dtype,
            requires_grad=False,
        )

        # Prepare SDXL conditioning
        latent_h, latent_w = latent.shape[2], latent.shape[3]
        image_h, image_w = latent_h * 8, latent_w * 8

        added_cond_kwargs = {
            "text_embeds": pooled,
            "time_ids": torch.tensor(
                [[image_h, image_w, 0, 0, image_h, image_w]],
                device=self.device,
                dtype=grad_dtype,
            ),
        }

        # Compute attributions for each timestep, then combine
        timestep_attributions = []

        for timestep_value, weight in zip(timestep_config.timesteps, timestep_config.weights):
            logger.debug(f"Computing gradients at timestep {timestep_value} (weight={weight:.2f})")

            attr = self._compute_gradients_single_timestep(
                unet=unet,
                pipe=pipe,
                lora_params=lora_params,
                latent=latent,
                cond=cond,
                added_cond_kwargs=added_cond_kwargs,
                timestep_value=timestep_value,
                adapter_name=adapter_name,
                num_integration_steps=num_integration_steps,
                grad_dtype=grad_dtype,
            )

            # Weight the attributions
            weighted_attr = {k: v * weight for k, v in attr.items()}
            timestep_attributions.append(weighted_attr)

        # Combine attributions across timesteps
        combined = self._combine_timestep_attributions(timestep_attributions)

        logger.debug(
            f"Multi-timestep IG complete. "
            f"Timesteps: {timestep_config.timesteps}, "
            f"Non-zero attributions: {len(combined)}"
        )

        return combined

    def _compute_gradients_single_timestep(
            self,
            unet: torch.nn.Module,
            pipe: Any,
            lora_params: Dict[str, torch.nn.Parameter],
            latent: torch.Tensor,
            cond: torch.Tensor,
            added_cond_kwargs: Dict[str, torch.Tensor],
            timestep_value: int,
            adapter_name: str,
            num_integration_steps: int,
            grad_dtype: torch.dtype,
    ) -> Dict[str, float]:
        """
        Compute Integrated Gradients at a single timestep.

        This is the core IG computation, extracted to support multi-timestep.

        Args:
            unet: UNet model
            pipe: Pipeline for adapter weight control
            lora_params: Dict of LoRA parameters
            latent: Input latent tensor
            cond: Text conditioning
            added_cond_kwargs: Additional conditioning (SDXL)
            timestep_value: Diffusion timestep (e.g., 500)
            adapter_name: LoRA adapter name
            num_integration_steps: Number of integration steps
            grad_dtype: Gradient-compatible dtype

        Returns:
            Attribution scores for this timestep
        """
        timestep = torch.tensor([timestep_value], device=self.device, dtype=torch.long)

        # Initialize accumulated gradients
        accumulated_grads = {
            name: torch.zeros_like(param)
            for name, param in lora_params.items()
        }

        # Integrated Gradients: alpha from 0 to 1
        for step in range(num_integration_steps + 1):
            alpha = step / num_integration_steps

            # Scale LoRA adapter
            pipe.set_adapters(adapter_name, adapter_weights=[alpha])

            with torch.enable_grad():
                # Forward pass
                try:
                    output = unet(
                        latent,
                        timestep,
                        encoder_hidden_states=cond,
                        added_cond_kwargs=added_cond_kwargs,
                    )
                except Exception:
                    # Fallback for SD1.5 (no added_cond_kwargs)
                    try:
                        output = unet(
                            latent,
                            timestep,
                            encoder_hidden_states=cond,
                        )
                    except Exception as e:
                        logger.error(f"Forward pass failed at timestep {timestep_value}: {e}")
                        pipe.set_adapters(adapter_name, adapter_weights=[1.0])
                        return {}

                # Extract sample
                if isinstance(output, dict):
                    output = output['sample']
                elif isinstance(output, tuple):
                    output = output[0]

                # Compute scalar target
                target = output.norm(p=2)

                if not target.requires_grad:
                    logger.error("Target doesn't require gradients")
                    pipe.set_adapters(adapter_name, adapter_weights=[1.0])
                    return {}

                # Backward pass
                target.backward()

            # Accumulate gradients
            for param_name, param in lora_params.items():
                if param.grad is not None:
                    accumulated_grads[param_name] += param.grad.clone()

            # Clear gradients
            unet.zero_grad()

        # Restore full LoRA
        pipe.set_adapters(adapter_name, adapter_weights=[1.0])

        # Compute attributions: avg_grad × param
        attributions = {}
        for param_name, param in lora_params.items():
            avg_grad = accumulated_grads[param_name] / num_integration_steps
            attribution = (avg_grad * param).abs().mean().item()

            if attribution > 0:
                attributions[param_name] = attribution

        return attributions

    def _combine_timestep_attributions(
            self,
            timestep_attributions: List[Dict[str, float]],
    ) -> Dict[str, float]:
        """
        Combine attributions from multiple timesteps.

        Simply sums the weighted attributions (weights already applied).

        Args:
            timestep_attributions: List of {param_name: weighted_score} dicts

        Returns:
            Combined attribution scores
        """
        if not timestep_attributions:
            return {}

        if len(timestep_attributions) == 1:
            return timestep_attributions[0]

        # Collect all keys
        all_keys = set()
        for attr in timestep_attributions:
            all_keys.update(attr.keys())

        # Sum weighted attributions
        combined = {}
        for key in all_keys:
            total = sum(attr.get(key, 0.0) for attr in timestep_attributions)
            if total > 0:
                combined[key] = total

        return combined