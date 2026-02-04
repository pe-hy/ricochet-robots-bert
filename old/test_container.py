#!/usr/bin/env python3
"""Test script to verify the PyTorch container setup."""

import torch
import transformers
import datasets
import wandb
import sys

def test_imports():
    """Test that all required packages can be imported."""
    print("=" * 60)
    print("Testing Container Setup")
    print("=" * 60)

    print(f"\n✓ Python: {sys.version.split()[0]}")
    print(f"✓ PyTorch: {torch.__version__}")
    print(f"✓ Transformers: {transformers.__version__}")
    print(f"✓ Datasets: {datasets.__version__}")
    print(f"✓ Wandb: {wandb.__version__}")

    print(f"\n✓ CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"✓ CUDA device count: {torch.cuda.device_count()}")
        print(f"✓ CUDA device name: {torch.cuda.get_device_name(0)}")
    else:
        print("  (CUDA will be available when running on GPU nodes)")

    print("\n" + "=" * 60)
    print("All packages loaded successfully!")
    print("=" * 60)

if __name__ == "__main__":
    test_imports()
