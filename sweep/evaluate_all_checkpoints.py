#!/usr/bin/env python3
"""
Evaluate all checkpoints across tasks on test sets.

Finds the best checkpoint for each task and evaluates on both IID and OOD test sets.
Test data is loaded once for efficiency.

Usage:
    python evaluate_all_checkpoints.py ./tmp/checkpoints --output results.csv
    python evaluate_all_checkpoints.py ./tmp/checkpoints --test-iid data/test_multiple_iid.jsonl --test-ood data/test_multiple_ood.jsonl
"""

import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
import pandas as pd
import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf
from tqdm import tqdm

from model.lightning_module import NodeClassifierLightningModule
from utils.data_module import RicochetRobotsDataModule


# Mapping from directory names to task config names
TASK_DIR_TO_CONFIG = {
    'ha_onehot': 'helper_aggregate',
    'tp_onehot': 'target_pos',
    'ch_onehot': 'chosen_helper',
}


def find_checkpoints(root_dir: str, pattern: str = "*.ckpt") -> list:
    """Recursively find all checkpoint files."""
    root_path = Path(root_dir)
    checkpoints = list(root_path.rglob(pattern))
    return sorted([str(p) for p in checkpoints])


def group_checkpoints_by_task(checkpoints: list) -> dict:
    """
    Group checkpoints by task based on directory structure.

    Returns:
        Dict mapping task names to lists of checkpoint paths
    """
    task_checkpoints = {}

    for ckpt_path in checkpoints:
        path_parts = Path(ckpt_path).parts

        # Find task directory in path
        task_dir = None
        for part in path_parts:
            if part in TASK_DIR_TO_CONFIG:
                task_dir = part
                break

        if task_dir:
            task_name = TASK_DIR_TO_CONFIG[task_dir]
            if task_name not in task_checkpoints:
                task_checkpoints[task_name] = []
            task_checkpoints[task_name].append(ckpt_path)

    return task_checkpoints


def evaluate_checkpoint(
    checkpoint_path: str,
    task_name: str,
    data_modules: dict,
    test_set_names: list
) -> dict:
    """
    Evaluate a single checkpoint on pre-loaded test data modules.

    Args:
        checkpoint_path: Path to checkpoint file
        task_name: Name of the task (helper_aggregate, target_pos, chosen_helper)
        data_modules: Dict mapping test set names to pre-loaded data modules
        test_set_names: List of test set names to evaluate

    Returns:
        Dict with checkpoint info and metrics for each test set
    """
    results = {
        'task': task_name,
        'checkpoint': checkpoint_path
    }

    # Extract epoch and val_em from filename
    filename = Path(checkpoint_path).stem
    if 'epoch=' in filename:
        try:
            epoch_str = filename.split('epoch=')[1].split('-')[0]
            results['epoch'] = int(epoch_str)
        except:
            results['epoch'] = -1

    # Extract validation exact match from filename
    if 'val_exact_match=' in filename or 'val_em=' in filename:
        try:
            if 'val_exact_match=' in filename:
                val_em_str = filename.split('val_exact_match=')[1].split('-')[0].split('.ckpt')[0]
            else:
                val_em_str = filename.split('val_em=')[1].split('-')[0].split('.ckpt')[0]
            results['val_em'] = float(val_em_str)
        except:
            results['val_em'] = -1.0
    else:
        results['val_em'] = -1.0

    # Extract run ID from path
    path_parts = Path(checkpoint_path).parts
    for i, part in enumerate(path_parts):
        if part in TASK_DIR_TO_CONFIG:
            if i + 1 < len(path_parts):
                results['run_id'] = path_parts[i + 1]
            break

    # Load the model
    try:
        model = NodeClassifierLightningModule.load_from_checkpoint(checkpoint_path)
        model.eval()
    except Exception as e:
        print(f"  ERROR loading model: {e}")
        return None

    # Evaluate on each test set using pre-loaded data modules
    for test_name in test_set_names:
        if test_name not in data_modules:
            print(f"  WARNING: Test set '{test_name}' not found in data modules")
            continue

        try:
            data_module = data_modules[test_name]

            # Create trainer
            trainer = pl.Trainer(
                accelerator='auto',
                devices=1,
                logger=False,
                enable_progress_bar=False,
                enable_model_summary=False
            )

            # Test
            test_results = trainer.test(model, data_module.train_dataloader(), verbose=False)

            # Store results with test set prefix
            for key, value in test_results[0].items():
                # Remove 'test/' prefix if present
                key_clean = key.replace('test/', '')
                results[f'{test_name}_{key_clean}'] = value

        except Exception as e:
            print(f"  ERROR evaluating on {test_name}: {e}")
            results[f'{test_name}_error'] = str(e)

    return results


def create_data_modules_for_task(
    task_name: str,
    test_sets: list,
    board_size: int = 16,
    batch_size: int = 512,
    num_workers: int = 0,
    positional_encoding: str = 'onehot'
) -> dict:
    """
    Create data modules for all test sets for a specific task.
    Data modules are loaded once and can be reused across checkpoint evaluations.

    Args:
        task_name: Name of the task (helper_aggregate, target_pos, chosen_helper)
        test_sets: List of test set dicts with 'name' and 'path' keys
        board_size: Board size
        batch_size: Batch size for evaluation
        num_workers: Number of data loader workers
        positional_encoding: Type of positional encoding

    Returns:
        Dict mapping test set names to data modules
    """
    # Load task config
    try:
        task_config_path = f'config/task/{task_name}.yaml'
        task_config = OmegaConf.load(task_config_path)
    except Exception as e:
        print(f"ERROR: Could not load task config for {task_name}: {e}")
        return {}

    data_modules = {}

    for test_set in test_sets:
        test_name = test_set['name']
        test_path = test_set['path']

        try:
            # Create data module
            data_module = RicochetRobotsDataModule(
                train_path=test_path,
                board_size=board_size,
                batch_size=batch_size,
                num_workers=num_workers,
                val_size=0,
                test_size=0,
                positional_encoding=positional_encoding,
                positional_encoding_kwargs={},
                task_config=OmegaConf.to_container(task_config, resolve=True)
            )
            data_module.setup()
            data_modules[test_name] = data_module

        except Exception as e:
            print(f"ERROR: Could not create data module for {test_name}: {e}")

    return data_modules


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate all checkpoints across tasks on test sets'
    )
    parser.add_argument('checkpoint_dir', type=str,
                       help='Root directory containing task checkpoint subdirectories')
    parser.add_argument('--output', type=str, default='best_checkpoints.csv',
                       help='Output CSV file for best checkpoints (default: best_checkpoints.csv)')
    parser.add_argument('--output-all', type=str, default=None,
                       help='Optional: Output CSV file for all checkpoint results')
    parser.add_argument('--test-iid', type=str, default='data/test_multiple_iid.jsonl',
                       help='IID test set path (default: data/test_multiple_iid.jsonl)')
    parser.add_argument('--test-ood', type=str, default='data/test_multiple_ood.jsonl',
                       help='OOD test set path (default: data/test_multiple_ood.jsonl)')
    parser.add_argument('--batch-size', type=int, default=512,
                       help='Batch size for evaluation (default: 512)')
    parser.add_argument('--num-workers', type=int, default=0,
                       help='Number of data loader workers (default: 0)')
    parser.add_argument('--pattern', type=str, default='*.ckpt',
                       help='Checkpoint filename pattern (default: *.ckpt)')
    parser.add_argument('--positional-encoding', type=str, default='onehot',
                       help='Positional encoding type (default: onehot)')

    args = parser.parse_args()

    # Check if checkpoint dir exists
    if not Path(args.checkpoint_dir).exists():
        print(f"ERROR: Directory not found: {args.checkpoint_dir}")
        sys.exit(1)

    # Find all checkpoints
    print(f"Searching for checkpoints in: {args.checkpoint_dir}")
    checkpoints = find_checkpoints(args.checkpoint_dir, args.pattern)

    if not checkpoints:
        print(f"ERROR: No checkpoints found matching pattern '{args.pattern}'")
        sys.exit(1)

    print(f"Found {len(checkpoints)} total checkpoints")

    # Group checkpoints by task
    task_checkpoints = group_checkpoints_by_task(checkpoints)

    if not task_checkpoints:
        print("ERROR: No checkpoints found for any recognized tasks")
        print(f"Expected task directories: {list(TASK_DIR_TO_CONFIG.keys())}")
        sys.exit(1)

    print(f"\nCheckpoints by task:")
    for task_name, ckpts in task_checkpoints.items():
        print(f"  {task_name}: {len(ckpts)} checkpoints")

    # Set up test sets
    test_sets = [
        {'name': 'iid', 'path': args.test_iid},
        {'name': 'ood', 'path': args.test_ood}
    ]
    test_set_names = [ts['name'] for ts in test_sets]

    print(f"\nTest sets: {test_set_names}")

    # Evaluate checkpoints for each task
    all_results = []
    best_checkpoints = []

    for task_name in sorted(task_checkpoints.keys()):
        ckpts = task_checkpoints[task_name]
        print(f"\n{'='*70}")
        print(f"Task: {task_name} ({len(ckpts)} checkpoints)")
        print(f"{'='*70}")

        # Load test data once for this task
        print(f"Loading test data modules for {task_name}...")
        data_modules = create_data_modules_for_task(
            task_name=task_name,
            test_sets=test_sets,
            board_size=16,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            positional_encoding=args.positional_encoding
        )

        if not data_modules:
            print(f"ERROR: Could not create data modules for {task_name}")
            continue

        print(f"Loaded {len(data_modules)} test data modules")

        # Get dataset sizes
        for test_name, dm in data_modules.items():
            n_samples = len(dm.train_dataset)
            print(f"  {test_name}: {n_samples} samples")

        # Evaluate all checkpoints for this task
        print(f"\nEvaluating {len(ckpts)} checkpoints...")
        task_results = []

        for checkpoint_path in tqdm(ckpts, desc=f"{task_name} checkpoints"):
            result = evaluate_checkpoint(
                checkpoint_path=checkpoint_path,
                task_name=task_name,
                data_modules=data_modules,
                test_set_names=test_set_names
            )

            if result is not None:
                task_results.append(result)
                all_results.append(result)

        # Find best checkpoint for this task (by validation EM)
        if task_results:
            df_task = pd.DataFrame(task_results)

            # Sort by validation exact match
            if 'val_em' in df_task.columns and df_task['val_em'].max() > 0:
                df_task = df_task.sort_values('val_em', ascending=False)
                best = df_task.iloc[0].to_dict()
            else:
                # If no val_em, pick first checkpoint
                print(f"  WARNING: No validation EM found, using first checkpoint")
                best = df_task.iloc[0].to_dict()

            best_checkpoints.append(best)

            # Print best checkpoint info
            print(f"\nBest checkpoint for {task_name}:")
            print(f"  Path: {best['checkpoint']}")
            print(f"  Epoch: {best.get('epoch', 'N/A')}")
            print(f"  Val EM: {best.get('val_em', -1):.4f}")
            for test_name in test_set_names:
                em_key = f'{test_name}_exact_match'
                if em_key in best:
                    print(f"  {test_name.upper()} EM: {best[em_key]:.4f}")

    # Save results
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    if best_checkpoints:
        # Save best checkpoints
        df_best = pd.DataFrame(best_checkpoints)
        df_best.to_csv(args.output, index=False)
        print(f"\nBest checkpoints saved to: {args.output}")

        # Print summary table
        print("\nBest Checkpoints Summary:")
        print(f"{'Task':<20} {'Val EM':<10} {'IID EM':<10} {'OOD EM':<10}")
        print("-" * 70)
        for _, row in df_best.iterrows():
            task = row['task']
            val_em = row.get('val_em', -1)
            iid_em = row.get('iid_exact_match', -1)
            ood_em = row.get('ood_exact_match', -1)
            print(f"{task:<20} {val_em:<10.4f} {iid_em:<10.4f} {ood_em:<10.4f}")

        # Save all results if requested
        if args.output_all and all_results:
            df_all = pd.DataFrame(all_results)
            df_all.to_csv(args.output_all, index=False)
            print(f"\nAll checkpoint results saved to: {args.output_all}")
            print(f"  Total checkpoints evaluated: {len(df_all)}")

    else:
        print("\nERROR: No checkpoints successfully evaluated")
        sys.exit(1)


if __name__ == '__main__':
    main()
