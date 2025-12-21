"""
Unified device and dtype management.

Consolidates all device/dtype conversion logic into a single class.
Replaces scattered `map_device()` and `str_to_dtype()` functions.
"""

import torch
from typing import Tuple, Optional, Union

from ..types import DeviceType, DtypeType


class DeviceManager:
    """
    Unified device and dtype management for tensor operations.

    Provides methods for:
    - Converting string representations to torch objects
    - Moving tensors between devices
    - Checking device availability
    - Selecting appropriate devices for operations
    """

    # Supported device strings
    SUPPORTED_DEVICES = ["cpu", "cuda", "mps", "auto"]

    # Supported dtype strings
    SUPPORTED_DTYPES = [
        "float16", "float32", "float64",
        "bfloat16",
        "int8", "int16", "int32", "int64",
    ]

    @staticmethod
    def parse(
        device: DeviceType,
        dtype: DtypeType
    ) -> Tuple[torch.device, torch.dtype]:
        """
        Convert device and dtype from string representation to torch objects.

        Backward compatible with the old `map_device()` function.

        Args:
            device: Device specification as string or torch.device
                   Supports: "cpu", "cuda", "cuda:0", "auto", or torch.device object
            dtype: Data type specification as string or torch.dtype
                  Supports: "float16", "float32", "bfloat16", etc., or torch.dtype object

        Returns:
            Tuple of (torch.device, torch.dtype)

        Raises:
            ValueError: If device or dtype string is invalid

        Examples:
            >>> DeviceManager.parse("cuda", "float16")
            (device(type='cuda'), torch.float16)

            >>> DeviceManager.parse(torch.device("cpu"), torch.float32)
            (device(type='cpu'), torch.float32)
        """
        # Convert device
        if isinstance(device, str):
            if device == "auto":
                device = DeviceManager.get_default_device()
            else:
                try:
                    device = torch.device(device)
                except RuntimeError as e:
                    raise ValueError(f"Invalid device string: {device}") from e
        elif not isinstance(device, torch.device):
            raise TypeError(f"Device must be str or torch.device, got {type(device)}")

        # Convert dtype
        if isinstance(dtype, str):
            if not hasattr(torch, dtype):
                raise ValueError(
                    f"Invalid dtype string: {dtype}. "
                    f"Supported: {', '.join(DeviceManager.SUPPORTED_DTYPES)}"
                )
            dtype = getattr(torch, dtype)
        elif not isinstance(dtype, torch.dtype):
            raise TypeError(f"Dtype must be str or torch.dtype, got {type(dtype)}")

        return device, dtype

    @staticmethod
    def get_default_device() -> torch.device:
        """
        Get the default device for operations.

        Checks availability in order: CUDA > MPS > CPU

        Returns:
            torch.device for the best available device
        """
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

    @staticmethod
    def is_device_available(device: Union[str, torch.device]) -> bool:
        """
        Check if a device is available.

        Args:
            device: Device to check (string or torch.device)

        Returns:
            True if device is available, False otherwise
        """
        if isinstance(device, str):
            device = torch.device(device)

        if device.type == "cpu":
            return True
        elif device.type == "cuda":
            return torch.cuda.is_available()
        elif device.type == "mps":
            return hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        else:
            return False

    @staticmethod
    def to_device(
        tensor: torch.Tensor,
        device: DeviceType,
        dtype: Optional[DtypeType] = None,
        non_blocking: bool = False
    ) -> torch.Tensor:
        """
        Move tensor to device with optional dtype conversion.

        Args:
            tensor: Tensor to move
            device: Target device
            dtype: Target dtype (optional, keeps current dtype if None)
            non_blocking: Whether to use non-blocking transfer

        Returns:
            Tensor on target device

        Example:
            >>> tensor = torch.randn(10, 10)
            >>> DeviceManager.to_device(tensor, "cuda", "float16")
        """
        device_obj, _ = DeviceManager.parse(device, dtype or tensor.dtype)

        if dtype is not None:
            _, dtype_obj = DeviceManager.parse(device, dtype)
            return tensor.to(device=device_obj, dtype=dtype_obj, non_blocking=non_blocking)
        else:
            return tensor.to(device=device_obj, non_blocking=non_blocking)

    @staticmethod
    def get_device_memory_info(device: Union[str, torch.device]) -> Optional[dict]:
        """
        Get memory information for a device.

        Args:
            device: Device to query

        Returns:
            Dictionary with memory info, or None if not available
            Keys: "allocated", "reserved", "total" (all in bytes)
        """
        if isinstance(device, str):
            device = torch.device(device)

        if device.type == "cuda" and torch.cuda.is_available():
            return {
                "allocated": torch.cuda.memory_allocated(device),
                "reserved": torch.cuda.memory_reserved(device),
                "total": torch.cuda.get_device_properties(device).total_memory,
            }
        else:
            return None

    @staticmethod
    def empty_cache(device: Optional[Union[str, torch.device]] = None):
        """
        Empty the cache for a device.

        Args:
            device: Device to clear cache for (defaults to CUDA if available)
        """
        if device is None:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            if isinstance(device, str):
                device = torch.device(device)

            if device.type == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
