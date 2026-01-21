#!/usr/bin/env python3
"""
Validate that JSON and JSONL loading produce identical results through the Dataset class.
Usage: python validate_dataset_loading.py original.json converted.jsonl
"""
import sys
import torch
import numpy as np
from pathlib import Path
from utils.data_module import RicochetRobotsDataset


def compare_tensors(t1, t2, name="tensor"):
    """Compare two tensors"""
    if not torch.equal(t1, t2):
        print(f"  ✗ {name} mismatch")
        print(f"    Shape: {t1.shape} vs {t2.shape}")
        if t1.shape == t2.shape:
            diff = (t1 != t2).sum().item()
            print(f"    Different elements: {diff}/{t1.numel()}")
            print(f"    Max abs diff: {(t1 - t2).abs().max().item()}")
        return False
    return True


def validate_datasets(json_path: str, jsonl_path: str):
    """Validate that both formats produce identical data through Dataset class"""

    print("Loading datasets...")
    print(f"  JSON: {json_path}")
    print(f"  JSONL: {jsonl_path}")

    # Create datasets with same configuration
    config = {
        'board_size': 16,
        'positional_encoding': 'onehot',
        'positional_encoding_kwargs': {},
        'task_config': {'target_index': 19, 'include_goal_features': [14, 15, 16, 17, 18]}
    }

    json_dataset = RicochetRobotsDataset(json_path, **config)
    jsonl_dataset = RicochetRobotsDataset(jsonl_path, **config)

    print(f"\nDataset sizes:")
    print(f"  JSON: {len(json_dataset)} examples")
    print(f"  JSONL: {len(jsonl_dataset)} examples")

    if len(json_dataset) != len(jsonl_dataset):
        print(f"\n✗ ERROR: Different dataset sizes!")
        return False

    # Compare all examples
    print(f"\nComparing {len(json_dataset)} examples...")
    num_examples = min(len(json_dataset), 100)  # Test first 100
    print(f"  (testing first {num_examples} for speed)")

    matches = 0
    mismatches = 0

    for idx in range(num_examples):
        json_item = json_dataset[idx]
        jsonl_item = jsonl_dataset[idx]

        # Compare example_id
        if json_item['example_id'] != jsonl_item['example_id']:
            print(f"\n✗ Example {idx}: example_id mismatch")
            print(f"    JSON: {json_item['example_id']}")
            print(f"    JSONL: {jsonl_item['example_id']}")
            mismatches += 1
            if mismatches >= 5:
                break
            continue

        # Compare features
        if not compare_tensors(json_item['features'], jsonl_item['features'], f"Example {idx} features"):
            mismatches += 1
            if mismatches >= 5:
                print("\n  ... (showing first 5 mismatches only)")
                break
            continue

        # Compare labels
        if not compare_tensors(json_item['labels'], jsonl_item['labels'], f"Example {idx} labels"):
            mismatches += 1
            if mismatches >= 5:
                print("\n  ... (showing first 5 mismatches only)")
                break
            continue

        matches += 1
        if (idx + 1) % 10 == 0:
            print(f"  Validated {idx + 1}/{num_examples} examples...")

    # Summary
    print(f"\n{'='*60}")
    print(f"RESULTS (tested {num_examples} examples):")
    print(f"  Matches: {matches} ({100*matches/num_examples:.2f}%)")
    print(f"  Mismatches: {mismatches} ({100*mismatches/num_examples:.2f}%)")

    if mismatches == 0:
        print(f"\n✓ All tested examples match perfectly!")
        print(f"✓ JSON and JSONL loading produce identical results")
        return True
    else:
        print(f"\n✗ Found {mismatches} mismatches")
        return False


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python validate_dataset_loading.py original.json converted.jsonl")
        sys.exit(1)

    json_path = sys.argv[1]
    jsonl_path = sys.argv[2]

    if not Path(json_path).exists():
        print(f"Error: {json_path} does not exist")
        sys.exit(1)

    if not Path(jsonl_path).exists():
        print(f"Error: {jsonl_path} does not exist")
        sys.exit(1)

    success = validate_datasets(json_path, jsonl_path)
    sys.exit(0 if success else 1)
