#!/usr/bin/env python3
"""Debug script to inspect checkpoint hparams."""

import torch
import sys

if len(sys.argv) < 2:
    print("Usage: python debug_checkpoint.py <checkpoint_path>")
    sys.exit(1)

checkpoint_path = sys.argv[1]
print(f"Loading checkpoint: {checkpoint_path}")

checkpoint = torch.load(checkpoint_path, map_location='cpu')

print("\nKeys in checkpoint:")
for key in checkpoint.keys():
    print(f"  {key}")

print("\nHyperparameters:")
if 'hyper_parameters' in checkpoint:
    for key, value in checkpoint['hyper_parameters'].items():
        print(f"  {key}: {value}")
else:
    print("  No hyper_parameters found")
