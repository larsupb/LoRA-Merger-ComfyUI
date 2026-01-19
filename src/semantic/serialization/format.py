"""
Adapter serialization format.

Provides standardized save/load for adapter packages using SafeTensors.
"""

import json
import logging
from pathlib import Path
from typing import Dict
import safetensors.torch


class AdapterSerializer:
    """
    Save/load adapter packages in standardized format.

    Format:
    adapter_package/
        config.json          - Metadata and configuration
        adapters.safetensors - Adapter weights in SafeTensors format
    """

    @staticmethod
    def save(adapter_package: Dict, output_dir: str):
        """
        Save adapter package to disk.

        Args:
            adapter_package: Dict with keys:
                - lora_names: List of LoRA names
                - feature_names: List of feature names
                - merge_spec: Merge specification
                - adapter_state_dict: Adapter state dict from registry
            output_dir: Directory to save package
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save config (metadata)
        config = {
            "lora_names": adapter_package["lora_names"],
            "feature_names": adapter_package["feature_names"],
            "merge_spec": adapter_package["merge_spec"],
            "adapter_types": adapter_package["adapter_state_dict"]["adapter_types"],
        }

        config_path = output_path / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        logging.info(f"Saved config to {config_path}")

        # Flatten adapter weights for SafeTensors
        # Convert from nested dict to flat dict with dotted keys
        adapter_tensors = {}
        for layer_key, adapter_state in adapter_package["adapter_state_dict"]["adapters"].items():
            for param_name, tensor in adapter_state.items():
                # Create flat key: "layer_key.param_name"
                full_key = f"{layer_key}.{param_name}"
                adapter_tensors[full_key] = tensor

        # Save adapter weights
        safetensors_path = output_path / "adapters.safetensors"
        safetensors.torch.save_file(adapter_tensors, safetensors_path)

        logging.info(
            f"Saved {len(adapter_tensors)} adapter parameters to {safetensors_path}"
        )

    @staticmethod
    def load(input_dir: str) -> Dict:
        """
        Load adapter package from disk.

        Args:
            input_dir: Directory containing adapter package

        Returns:
            Dict with same structure as save() input
        """
        input_path = Path(input_dir)

        if not input_path.exists():
            raise FileNotFoundError(f"Adapter package not found: {input_dir}")

        # Load config
        config_path = input_path / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r") as f:
            config = json.load(f)

        logging.info(f"Loaded config from {config_path}")

        # Load adapter weights
        safetensors_path = input_path / "adapters.safetensors"
        if not safetensors_path.exists():
            raise FileNotFoundError(
                f"Adapter weights not found: {safetensors_path}"
            )

        adapter_tensors = safetensors.torch.load_file(safetensors_path)

        logging.info(
            f"Loaded {len(adapter_tensors)} adapter parameters from {safetensors_path}"
        )

        # Reconstruct nested adapter_state_dict from flat keys
        # Convert "layer_key.param_name" back to nested dict
        adapters = {}
        for full_key, tensor in adapter_tensors.items():
            # Split on last dot to separate layer_key from param_name
            if "." not in full_key:
                logging.warning(f"Unexpected key format: {full_key}, skipping")
                continue

            layer_key, param_name = full_key.rsplit(".", 1)

            if layer_key not in adapters:
                adapters[layer_key] = {}

            adapters[layer_key][param_name] = tensor

        # Reconstruct full adapter package
        adapter_package = {
            "lora_names": config["lora_names"],
            "feature_names": config["feature_names"],
            "merge_spec": config["merge_spec"],
            "adapter_state_dict": {
                "adapter_types": config["adapter_types"],
                "adapters": adapters,
                "feature_names": config["feature_names"],
            },
        }

        return adapter_package

    @staticmethod
    def validate(adapter_package: Dict) -> bool:
        """
        Validate adapter package structure.

        Args:
            adapter_package: Package to validate

        Returns:
            True if valid, False otherwise
        """
        required_keys = ["lora_names", "feature_names", "merge_spec", "adapter_state_dict"]

        for key in required_keys:
            if key not in adapter_package:
                logging.error(f"Missing required key: {key}")
                return False

        # Validate adapter_state_dict structure
        adapter_state = adapter_package["adapter_state_dict"]
        required_state_keys = ["adapter_types", "adapters", "feature_names"]

        for key in required_state_keys:
            if key not in adapter_state:
                logging.error(f"Missing required state key: {key}")
                return False

        # Check that lists are non-empty
        if not adapter_package["lora_names"]:
            logging.error("lora_names is empty")
            return False

        if not adapter_package["feature_names"]:
            logging.error("feature_names is empty")
            return False

        logging.info("Adapter package validation passed")
        return True
