"""
Pytest configuration and fixtures for LoRA Power-Merger tests

This file sets up the test environment by mocking ComfyUI dependencies
that may not be available in the test environment.
"""

import sys
import os
from unittest.mock import MagicMock
from typing import Tuple
import torch

# Add the src directory to the path to allow imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Prevent pytest from treating the project root as a package
# by ensuring tests are collected from tests directory only
import pytest

def pytest_ignore_collect(collection_path, config):
    """Ignore collection of __init__.py in project root"""
    # Convert to string for comparison
    path_str = str(collection_path)
    # Ignore the root __init__.py
    if path_str.endswith('__init__.py') and 'tests' not in path_str:
        return True
    # Ignore src directory
    if '/src/' in path_str or path_str.endswith('/src'):
        return True
    return False

# Mock ComfyUI modules before any test imports
def pytest_configure(config):
    """Configure pytest and mock unavailable modules"""

    # Mock comfy module
    comfy_mock = MagicMock()
    sys.modules['comfy'] = comfy_mock
    sys.modules['comfy.utils'] = MagicMock()
    sys.modules['comfy.model_management'] = MagicMock()
    sys.modules['comfy.lora'] = MagicMock()
    sys.modules['comfy.weight_adapter'] = MagicMock()
    sys.modules['comfy.model_patcher'] = MagicMock()
    sys.modules['comfy.sd'] = MagicMock()

    # Create a LoRAAdapter mock class
    class LoRAAdapterMock:
        def __init__(self, *args, **kwargs):
            self.state_dict = {}

    comfy_mock.weight_adapter.LoRAAdapter = LoRAAdapterMock

    # Mock comfy_util module
    comfy_util_mock = MagicMock()
    sys.modules['comfy_util'] = comfy_util_mock

    # Mock nodes module (ComfyUI nodes)
    nodes_mock = MagicMock()
    sys.modules['nodes'] = nodes_mock

    # Mock folder_paths module (ComfyUI utility)
    folder_paths_mock = MagicMock()
    folder_paths_mock.get_folder_paths.return_value = []
    folder_paths_mock.folder_names_and_paths = {}
    sys.modules['folder_paths'] = folder_paths_mock

    # Mock comfy_extras module (ComfyUI extras)
    comfy_extras_mock = MagicMock()
    sys.modules['comfy_extras'] = comfy_extras_mock
    sys.modules['comfy_extras.nodes_custom_sampler'] = MagicMock()

    # Mock architectures module to avoid relative import issues
    architectures_mock = MagicMock()
    # Define the UP_DOWN_ALPHA_TUPLE type alias
    architectures_mock.sd_lora.UP_DOWN_ALPHA_TUPLE = Tuple[torch.Tensor, torch.Tensor, float]
    sys.modules['architectures'] = architectures_mock
    sys.modules['architectures.sd_lora'] = architectures_mock.sd_lora
    sys.modules['architectures.general_architecture'] = MagicMock()
    sys.modules['architectures.wan_lora'] = MagicMock()