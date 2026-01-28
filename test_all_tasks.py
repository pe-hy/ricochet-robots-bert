#!/usr/bin/env python3
"""
Comprehensive test script to validate all task configurations.
Tests all 4 binary classification tasks for 1 epoch each.
"""

import sys
import os
import json
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from omegaconf import OmegaConf
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.node_classifier import NodeClassifierConfig
from model.lightning_module import NodeClassifierLightningModule
from utils.data_module import RicochetRobotsDataModule


def validate_task_config(task_name: str):
    """Validate that a task config extracts the correct label."""
    print(f"\n{'='*80}")
    print(f"VALIDATING TASK: {task_name}")
    print(f"{'='*80}")

    # Load configs
    base_config = OmegaConf.load('config/node_classifier.yaml')
    task_config = OmegaConf.load(f'config/task/{task_name}.yaml')

    # Load metadata
    with open(base_config.data.train_path, 'r') as f:
        metadata = json.loads(f.readline())
        example = json.loads(f.readline())

    node = example['nodes'][0]

    print(f"\nTask Configuration:")
    print(f"  Name: {task_config.name}")
    print(f"  Target index: {task_config.target_index}")
    print(f"  Include features: {task_config.include_goal_features}")
    print(f"\nData Validation:")
    print(f"  Node has {len(node)} features")
    print(f"  Target index {task_config.target_index} -> '{metadata['feature_description'][task_config.target_index]}'")

    # Validate included features
    if task_config.include_goal_features:
        print(f"\n  Additional features included:")
        for idx in task_config.include_goal_features:
            print(f"    [{idx}] {metadata['feature_description'][idx]}")

    # Check that target is binary
    target_value = node[task_config.target_index]
    print(f"\n  Sample target value: {target_value}")
    print(f"  Is binary (0 or 1): {target_value in [0, 1]}")

    # Calculate expected feature dimension
    base_features = 12  # robot_type(5) + has_goal(2) + walls(5)
    num_included = len(task_config.include_goal_features)
    pos_encoding_dim = 32  # onehot encoding for 16x16 board
    expected_dim = base_features + num_included + pos_encoding_dim

    print(f"\n  Expected feature dim: {base_features} (base) + {num_included} (included) + {pos_encoding_dim} (pos) = {expected_dim}")

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
    print(f"  Actual feature dim: {actual_dim}")
    print(f"  Match: {'✓' if actual_dim == expected_dim else '✗ MISMATCH!'}")

    if actual_dim != expected_dim:
        print(f"\n  ERROR: Feature dimension mismatch!")
        return False

    print(f"\n{'='*80}")
    print(f"VALIDATION PASSED ✓")
    print(f"{'='*80}")
    return True


def test_task_training(task_name: str, epochs: int = 1):
    """Train a task for specified epochs and validate metrics."""
    print(f"\n\n{'='*80}")
    print(f"TESTING TASK: {task_name.upper()}")
    print(f"{'='*80}\n")

    # Load configs
    base_config = OmegaConf.load('config/node_classifier.yaml')
    task_config = OmegaConf.load(f'config/task/{task_name}.yaml')
    base_config.task = task_config

    # Setup data module
    task_config_dict = {
        'target_index': task_config.target_index,
        'include_goal_features': list(task_config.include_goal_features)
    }

    data_module = RicochetRobotsDataModule(
        train_path=base_config.data.train_path,
        board_size=base_config.data.board_size,
        batch_size=256,
        num_workers=0,
        val_size=256,
        test_size=0,
        positional_encoding='onehot',
        positional_encoding_kwargs={},
        task_config=task_config_dict
    )

    # Calculate steps per epoch
    with open(base_config.data.train_path, 'r') as f:
        metadata = json.loads(f.readline())
        total_examples = metadata['num_examples']

    train_size = total_examples - 256  # val_size
    steps_per_epoch = train_size // 256  # batch_size

    print(f"Training Configuration:")
    print(f"  Task: {task_config.name}")
    print(f"  Target: '{metadata['feature_description'][task_config.target_index]}'")
    print(f"  Feature dim: {data_module.feature_dim}")
    print(f"  Train size: {train_size}")
    print(f"  Val size: 256")
    print(f"  Batch size: 256")
    print(f"  Steps per epoch: {steps_per_epoch}")
    print(f"  Epochs: {epochs}")

    # Create model config
    model_config = NodeClassifierConfig(
        feature_dim=data_module.feature_dim,
        d_model=256,
        nhead=8,
        num_layers=4,  # Smaller for faster testing
        dim_feedforward=1024,
        dropout=0.1,
        activation='gelu',
        positional_encoding='onehot',
        board_size=base_config.data.board_size,
        pos_encoding_dim=0,
        pos_combine_method='concat'
    )

    # Create lightning module
    lightning_module = NodeClassifierLightningModule(
        model_config=model_config,
        max_lr=0.005,
        weight_decay=0.1,
        warmup_epochs=0,  # No warmup for quick test
        total_epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        pos_weight=1.0,
        log_predictions=False
    )

    # Setup callbacks
    checkpoint_dir = f'./tmp/test_checkpoints/{task_name}'
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    callbacks = [
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename=f'{task_name}-epoch={{epoch:02d}}-val_em={{val_exact_match:.4f}}',
            monitor='val_exact_match',
            mode='max',
            save_top_k=1,
            save_last=False,
            auto_insert_metric_name=False
        ),
        LearningRateMonitor(logging_interval='step')
    ]

    # Create trainer
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator='auto',
        devices=1,
        precision='bf16-mixed',
        logger=False,  # No wandb for testing
        callbacks=callbacks,
        log_every_n_steps=50,
        val_check_interval=1.0,
        gradient_clip_val=1.0,
        enable_progress_bar=True,
        enable_model_summary=False
    )

    # Print expected metrics
    print(f"\nExpected Metrics after {epochs} epoch(s):")
    print(f"  - All tasks are BINARY classification (not regression)")
    print(f"  - train/accuracy: ~98-99%")
    print(f"  - val/accuracy: ~98-99%")
    print(f"  - val_exact_match: ~10-20% (this is normal for graph-level exact match)")
    print(f"\nStarting training...\n")

    # Train
    trainer.fit(lightning_module, data_module)

    # Get final metrics
    val_metrics = trainer.callback_metrics

    print(f"\n{'='*80}")
    print(f"RESULTS FOR: {task_name.upper()}")
    print(f"{'='*80}")
    print(f"Final Metrics:")
    if 'val/accuracy' in val_metrics:
        print(f"  val/accuracy: {val_metrics['val/accuracy']:.4f}")
    if 'val/precision' in val_metrics:
        print(f"  val/precision: {val_metrics['val/precision']:.4f}")
    if 'val/recall' in val_metrics:
        print(f"  val/recall: {val_metrics['val/recall']:.4f}")
    if 'val_exact_match' in val_metrics:
        print(f"  val_exact_match: {val_metrics['val_exact_match']:.4f}")

    # Validate results
    success = True
    if 'val/accuracy' in val_metrics:
        acc = val_metrics['val/accuracy'].item()
        if acc < 0.90:  # Should be >90% for binary classification
            print(f"\n  ✗ WARNING: Accuracy too low ({acc:.2%}) - expected >90%")
            success = False
        else:
            print(f"\n  ✓ Accuracy looks good ({acc:.2%})")

    print(f"{'='*80}\n")

    return success


def main():
    """Run validation and testing for all tasks."""
    print("\n" + "="*80)
    print("COMPREHENSIVE TASK VALIDATION AND TESTING")
    print("="*80)

    tasks = ['subgoal_label', 'helper_aggregate', 'target_pos', 'chosen_helper']

    # Step 1: Validate all task configurations
    print("\n" + "="*80)
    print("STEP 1: VALIDATING ALL TASK CONFIGURATIONS")
    print("="*80)

    validation_results = {}
    for task in tasks:
        try:
            validation_results[task] = validate_task_config(task)
        except Exception as e:
            print(f"\n✗ ERROR validating {task}: {e}")
            validation_results[task] = False

    # Print validation summary
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)
    for task, success in validation_results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {task:20s} {status}")

    if not all(validation_results.values()):
        print("\n✗ Some validations failed. Fix errors before training.")
        return 1

    print("\n✓ All validations passed!\n")

    # Step 2: Train each task for 1 epoch
    print("\n" + "="*80)
    print("STEP 2: TRAINING ALL TASKS (1 EPOCH EACH)")
    print("="*80)

    training_results = {}
    for task in tasks:
        try:
            training_results[task] = test_task_training(task, epochs=1)
        except Exception as e:
            print(f"\n✗ ERROR training {task}: {e}")
            import traceback
            traceback.print_exc()
            training_results[task] = False

    # Final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    print("\nValidation Results:")
    for task, success in validation_results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {task:20s} {status}")

    print("\nTraining Results:")
    for task, success in training_results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {task:20s} {status}")

    all_passed = all(validation_results.values()) and all(training_results.values())

    if all_passed:
        print("\n" + "="*80)
        print("✓✓✓ ALL TESTS PASSED! ✓✓✓")
        print("="*80)
        print("\nAll 4 binary classification tasks are working correctly!")
        print("You can now run sweeps with confidence.")
        return 0
    else:
        print("\n" + "="*80)
        print("✗ SOME TESTS FAILED")
        print("="*80)
        return 1


if __name__ == '__main__':
    sys.exit(main())
