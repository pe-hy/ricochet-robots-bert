#!/usr/bin/env python3
"""Quick validation script to check all task configs are correct."""

import sys
import os
import json
from omegaconf import OmegaConf

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.data_module import RicochetRobotsDataModule


def validate_all_tasks():
    """Validate all task configurations."""

    # Load base config
    base_config = OmegaConf.load('config/node_classifier.yaml')

    # Load metadata
    with open(base_config.data.train_path, 'r') as f:
        metadata = json.loads(f.readline())
        sample_example = json.loads(f.readline())

    sample_node = sample_example['nodes'][0]

    print("="*80)
    print("TASK CONFIGURATION VALIDATION")
    print("="*80)
    print(f"\nData file: {base_config.data.train_path}")
    print(f"Features per node: {len(sample_node)}")
    print()

    tasks = [
        ('subgoal_label', 20, []),
        ('helper_aggregate', 17, [20]),
        ('target_pos', 18, [17, 20]),
        ('chosen_helper', 19, [17, 18, 20])
    ]

    all_valid = True

    for task_name, expected_target_idx, expected_features in tasks:
        print(f"\n{'='*80}")
        print(f"Task: {task_name}")
        print(f"{'='*80}")

        # Load task config
        task_config = OmegaConf.load(f'config/task/{task_name}.yaml')

        # Validate target index
        print(f"  Target index: {task_config.target_index}")
        print(f"    -> {metadata['feature_description'][task_config.target_index]}")

        if task_config.target_index != expected_target_idx:
            print(f"    ✗ ERROR: Expected {expected_target_idx}, got {task_config.target_index}")
            all_valid = False
        else:
            print(f"    ✓ Correct")

        # Validate target is binary
        sample_value = sample_node[task_config.target_index]
        is_binary = sample_value in [0, 1]
        print(f"  Sample value: {sample_value} (binary: {is_binary})")

        if not is_binary:
            print(f"    ✗ ERROR: Not a binary value!")
            all_valid = False

        # Validate included features
        print(f"  Included features: {task_config.include_goal_features}")
        if list(task_config.include_goal_features) != expected_features:
            print(f"    ✗ ERROR: Expected {expected_features}")
            all_valid = False
        else:
            if expected_features:
                for idx in expected_features:
                    print(f"    [{idx}] {metadata['feature_description'][idx]}")
            print(f"    ✓ Correct")

        # Calculate expected feature dimension
        base_features = 12  # robot_type(5) + has_goal(2) + walls(5)
        num_included = len(task_config.include_goal_features)
        pos_encoding_dim = 32  # onehot for 16x16 board
        expected_dim = base_features + num_included + pos_encoding_dim

        # Validate with data module
        task_config_dict = {
            'target_index': task_config.target_index,
            'include_goal_features': list(task_config.include_goal_features)
        }

        data_module = RicochetRobotsDataModule(
            train_path=base_config.data.train_path,
            board_size=base_config.data.board_size,
            batch_size=64,
            num_workers=0,
            val_size=64,
            test_size=0,
            positional_encoding='onehot',
            positional_encoding_kwargs={},
            task_config=task_config_dict
        )

        actual_dim = data_module.feature_dim
        print(f"  Feature dimension: {actual_dim} (expected: {expected_dim})")

        if actual_dim != expected_dim:
            print(f"    ✗ ERROR: Dimension mismatch!")
            all_valid = False
        else:
            print(f"    ✓ Correct")

    print(f"\n{'='*80}")
    if all_valid:
        print("✓✓✓ ALL TASK CONFIGURATIONS VALID ✓✓✓")
    else:
        print("✗✗✗ SOME CONFIGURATIONS INVALID ✗✗✗")
    print(f"{'='*80}\n")

    return 0 if all_valid else 1


if __name__ == '__main__':
    sys.exit(validate_all_tasks())
