"""Transformer config -- delegates to project root config.py."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import config as _root_config

# Transformer-specific paths
TRANSFORMER_OUTPUT_DIR = os.path.join(_root_config.OUTPUT_DIR, "fine_tuned_model")
TRANSFORMER_CHECKPOINT_DIR = os.path.join(_root_config.OUTPUT_DIR, "checkpoints")

# Data / device
_device, _device_name = _root_config.get_device()
DEVICE = _device
DEVICE_NAME = _device_name

# Offline mode
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"


def print_config():
    print("\n" + "=" * 80)
    print("TRANSFORMER FINE-TUNING CONFIGURATION")
    print("=" * 80)
    print(f"Model:         {_root_config.TRANSFORMER_MODEL_NAME}")
    print(f"Num Labels:    {_root_config.TRANSFORMER_NUM_LABELS}")
    print(f"Device:        {DEVICE_NAME}")
    print(f"Batch Size:    {_root_config.TRANSFORMER_BATCH_SIZE}")
    print(f"Learning Rate: {_root_config.TRANSFORMER_LEARNING_RATE}")
    print(f"Epochs:        {_root_config.TRANSFORMER_EPOCHS}")
    print(f"Max Length:    {_root_config.TRANSFORMER_MAX_LENGTH}")
    print(f"Weight Decay:    {_root_config.TRANSFORMER_WEIGHT_DECAY}")
    print(f"Warmup Steps:    {_root_config.TRANSFORMER_WARMUP_STEPS}")
    print(f"Output Dir:    {TRANSFORMER_OUTPUT_DIR}")
    print(f"Label Smoothing: {_root_config.LABEL_SMOOTHING}")
    print("=" * 80 + "\n")
