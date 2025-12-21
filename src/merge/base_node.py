"""
Base class for merge method nodes.

Provides common functionality for all merge method nodes to eliminate boilerplate code.
Each merge method node only needs to define INPUT_TYPES and get_settings().
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, ClassVar


class BaseMergeMethodNode(ABC):
    """
    Abstract base class for merge method nodes.

    Subclasses should:
    1. Define INPUT_TYPES() classmethod with method-specific parameters
    2. Implement get_settings(**kwargs) to return settings dictionary
    3. Set METHOD_NAME class variable

    The base class provides:
    - Common RETURN_TYPES, FUNCTION, CATEGORY
    - get_method() implementation
    - Consistent structure across all merge method nodes

    Example:
        class LinearMergeMethod(BaseMergeMethodNode):
            METHOD_NAME = "linear"
            CATEGORY = "LoRA PowerMerge"

            @classmethod
            def INPUT_TYPES(cls):
                return {
                    "required": {
                        "normalize": ("BOOLEAN", {"default": True}),
                    }
                }

            def get_settings(self, normalize: bool):
                return {"normalize": normalize}
    """

    # Must be set by subclass
    METHOD_NAME: ClassVar[str] = ""

    # ComfyUI node configuration
    RETURN_TYPES = ("MergeMethod",)
    FUNCTION = "get_method"
    CATEGORY = "LoRA PowerMerge"  # Can be overridden by subclass

    @classmethod
    @abstractmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        """
        Define input types for this merge method.

        Should return a dictionary with "required" and optionally "optional" keys.

        Returns:
            Dictionary defining input parameters
        """
        pass

    @abstractmethod
    def get_settings(self, **kwargs) -> Dict[str, Any]:
        """
        Convert input parameters to settings dictionary.

        Args:
            **kwargs: Input parameters from ComfyUI

        Returns:
            Settings dictionary for this merge method
        """
        pass

    def get_method(self, **kwargs) -> tuple:
        """
        Create merge method definition dictionary.

        This method is called by ComfyUI. It combines the method name
        with the settings returned by get_settings().

        Args:
            **kwargs: Input parameters from ComfyUI

        Returns:
            Tuple containing method definition dictionary
        """
        if not self.METHOD_NAME:
            raise NotImplementedError(
                f"{self.__class__.__name__} must set METHOD_NAME class variable"
            )

        method_def = {
            "name": self.METHOD_NAME,
            "settings": self.get_settings(**kwargs)
        }

        return (method_def,)


class BaseTaskArithmeticNode(BaseMergeMethodNode):
    """
    Base class for task arithmetic merge method nodes.

    Provides common parameters shared by GTA methods (TIES, DARE, DELLA, etc.).

    Subclasses only need to set METHOD_NAME and optionally override get_extra_inputs()
    for method-specific parameters.
    """

    CATEGORY = "LoRA PowerMerge/Task Arithmetic"

    @classmethod
    def get_extra_inputs(cls) -> Dict[str, Any]:
        """
        Define extra method-specific inputs.

        Override this in subclasses to add method-specific parameters.

        Returns:
            Dictionary of extra input parameters
        """
        return {}

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        """
        Define common task arithmetic inputs.

        Includes rescale_norm and normalize, plus any extra inputs
        from get_extra_inputs().
        """
        base_inputs = {
            "required": {
                "rescale_norm": (
                    ["default", "l1", "none"],
                    {
                        "default": "default",
                        "tooltip": "Norm rescaling strategy for task vectors"
                    }
                ),
                "normalize": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Normalize weights to sum to 1"
                    }
                ),
            }
        }

        # Merge with extra inputs
        extra_inputs = cls.get_extra_inputs()
        if extra_inputs:
            base_inputs["required"].update(extra_inputs)

        return base_inputs

    def get_settings(
        self,
        rescale_norm: str = "default",
        normalize: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Convert inputs to settings dictionary.

        Args:
            rescale_norm: Norm rescaling strategy
            normalize: Whether to normalize weights
            **kwargs: Extra method-specific parameters

        Returns:
            Settings dictionary
        """
        settings = {
            "rescale_norm": rescale_norm,
            "normalize": normalize,
        }

        # Add any extra settings
        settings.update(kwargs)

        return settings
