#!/usr/bin/env python3
"""
Test runner that sets up mocking before pytest runs
"""

import sys
from unittest.mock import MagicMock
from typing import Tuple
import torch

# Mock ComfyUI modules BEFORE any imports
sys.modules['comfy'] = MagicMock()
sys.modules['comfy.utils'] = MagicMock()
sys.modules['comfy.model_management'] = MagicMock()
sys.modules['comfy.lora'] = MagicMock()

# Mock architectures module
architectures_mock = MagicMock()
architectures_mock.sd_lora.UP_DOWN_ALPHA_TUPLE = Tuple[torch.Tensor, torch.Tensor, float]
sys.modules['architectures'] = architectures_mock
sys.modules['architectures.sd_lora'] = architectures_mock.sd_lora
sys.modules['architectures.general_architecture'] = MagicMock()
sys.modules['architectures.wan_lora'] = MagicMock()

# Now run pytest
import pytest

if __name__ == "__main__":
    sys.exit(pytest.main(sys.argv[1:] or ["test_utility.py", "-v"]))