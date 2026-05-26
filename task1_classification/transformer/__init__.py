"""
Initialize transformer directory with __init__.py

Makes transformer a Python package for easy imports.
"""
from .transformer import run_transformer
from .config import (
    DEVICE, DEVICE_NAME, print_config,
    TRANSFORMER_OUTPUT_DIR, TRANSFORMER_CHECKPOINT_DIR,
)

__all__ = [
    "DEVICE","DEVICE_NAME", "print_config",
    "TRANSFORMER_OUTPUT_DIR", "TRANSFORMER_CHECKPOINT_DIR",
    "run_transformer",
]
