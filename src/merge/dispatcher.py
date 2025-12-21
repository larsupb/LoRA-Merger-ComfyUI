"""
Merge method dispatcher for LoRA Power-Merger.

Handles routing merge requests to the appropriate algorithm implementation
based on method name. Replaces the large if-elif chain with a registry pattern.
"""

from typing import get_args, Callable

from ..mergekit_utils import MERGEKIT_GTA_MODES
from .algorithms import (
    MERGE_ALGORITHMS,
    generalized_task_arithmetic_merge,
)


# Extended registry including GTA methods
def get_merge_method(method_name: str) -> Callable:
    """
    Get merge method function by name.

    Supports both direct algorithm names and GTA mode variants.

    Args:
        method_name: Name of the merge method

    Returns:
        Merge method function

    Raises:
        ValueError: If method name is unknown/unsupported

    Examples:
        >>> get_merge_method("linear")
        <function linear_merge>
        >>> get_merge_method("dare")  # GTA mode
        <function generalized_task_arithmetic_merge>
    """
    # Check direct algorithm registry first
    if method_name in MERGE_ALGORITHMS:
        return MERGE_ALGORITHMS[method_name]

    # Check if it's a GTA mode
    if method_name in get_args(MERGEKIT_GTA_MODES):
        return generalized_task_arithmetic_merge

    # Unknown method
    available_methods = list(MERGE_ALGORITHMS.keys()) + list(get_args(MERGEKIT_GTA_MODES))
    raise ValueError(
        f"Invalid / unsupported merge method: {method_name}. "
        f"Available methods: {', '.join(sorted(available_methods))}"
    )


def prepare_method_args(method_name: str, method_settings: dict) -> dict:
    """
    Prepare method arguments dictionary for merge execution.

    Combines method name, default settings, and user-provided settings.

    Args:
        method_name: Name of the merge method
        method_settings: User-provided method settings

    Returns:
        Complete method arguments dictionary

    Example:
        >>> prepare_method_args("dare", {"density": 0.8})
        {
            "mode": "dare",
            "int8_mask": False,
            "lambda_": 1.0,
            "density": 0.8
        }
    """
    # Base method arguments
    method_args = {
        "mode": method_name,
        "int8_mask": False,
        "lambda_": 1.0,  # Internal GTA processing (applied separately at the end)
    }

    # Merge with user settings
    method_args.update(method_settings)

    return method_args
