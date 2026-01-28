#!/usr/bin/env python3
"""
Evaluate all checkpoints in a directory on test sets.

Usage:
    python evaluate_all_checkpoints.py ./tmp/checkpoints --output results.csv
    python evaluate_all_checkpoints.py ./tmp/checkpoints/learned/c2euc1c6 --test-set data/test_multiple_iid.jsonl
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


def find_checkpoints(root_dir: str, pattern: str = "*.ckpt") -> list:
    """Recursively find all checkpoint files."""
    root_path = Path(root_dir)
    checkpoints = list(root_path.rglob(pattern))
    return sorted([str(p) for p in checkpoints])


def evaluate_checkpoint(
    checkpoint_path: str,
    test_sets: list,
    board_size: int = 16,
    batch_size: int = 512,
    num_workers: int = 0,
    max_samples: int = None
) -> dict:
    """
    Evaluate a single checkpoint on all test sets.

    Returns:
        Dict with test set names as keys and metrics as values
    """
    results = {'checkpoint': checkpoint_path}

    # Try to extract metadata from checkpoint path
    path_parts = Path(checkpoint_path).parts
    if len(path_parts) > 2:
        results['sweep_id'] = path_parts[-3] if len(path_parts) > 3 else 'unknown'
        results['run_id'] = path_parts[-2] if len(path_parts) > 2 else 'unknown'

    # Extract epoch and val_em from filename
    filename = Path(checkpoint_path).stem
    if 'epoch=' in filename:
        try:
            epoch_str = filename.split('epoch=')[1].split('-')[0]
            results['epoch'] = int(epoch_str)
        except:
            results['epoch'] = -1

    if 'val_em=' in filename:
        try:
            val_em_str = filename.split('val_em=')[1].split('-')[0].split('.ckpt')[0]
            results['val_em'] = float(val_em_str)
        except:
            results['val_em'] = -1.0

    # Load checkpoint to extract config (don't load model yet)
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        hparams = checkpoint.get('hyper_parameters', {})
        model_config = hparams.get('model_config', {})
    except Exception as e:
        print(f"  ERROR loading checkpoint: {e}")
        return None

    # Extract positional encoding config from model_config
    positional_encoding = model_config.get('positional_encoding', 'onehot')
    board_size_from_model = model_config.get('board_size', board_size)
    pos_encoding_dim = model_config.get('pos_encoding_dim', 0)
    pos_combine_method = model_config.get('pos_combine_method', 'concat')

    # Store config info in results
    results['positional_encoding'] = positional_encoding
    results['pos_encoding_dim'] = pos_encoding_dim
    results['pos_combine_method'] = pos_combine_method
    results['d_model'] = model_config.get('d_model', -1)
    results['num_layers'] = model_config.get('num_layers', -1)
    results['nhead'] = model_config.get('nhead', -1)

    # Build positional_encoding_kwargs for data module
    if positional_encoding == 'learned':
        pos_encoding_kwargs = {
            'encoding_dim': pos_encoding_dim,
            'combine_method': pos_combine_method
        }
    else:
        pos_encoding_kwargs = {}

    # Load base config for task config
    try:
        base_config = OmegaConf.load('config/node_classifier.yaml')
        task_config = OmegaConf.load('config/task/subgoal_label.yaml')
        base_config.task = task_config
    except Exception as e:
        print(f"  ERROR loading config: {e}")
        return None

    # Now load the model
    try:
        model = NodeClassifierLightningModule.load_from_checkpoint(checkpoint_path)
    except Exception as e:
        print(f"  ERROR loading model: {e}")
        return None

    # Evaluate on each test set
    for test_set in test_sets:
        test_name = test_set.get('name', Path(test_set['path']).stem)

        try:
            # Create data module
            data_module = RicochetRobotsDataModule(
                train_path=test_set['path'],
                board_size=board_size_from_model,
                batch_size=batch_size,
                num_workers=num_workers,
                val_size=0,
                test_size=0,
                positional_encoding=positional_encoding,
                positional_encoding_kwargs=pos_encoding_kwargs,
                task_config=OmegaConf.to_container(base_config.task, resolve=True)
            )
            data_module.setup()

            # Limit samples if specified
            if max_samples is not None and max_samples > 0:
                from torch.utils.data import Subset
                dataset = data_module.train_dataset
                if len(dataset) > max_samples:
                    indices = list(range(max_samples))
                    data_module.train_dataset = Subset(dataset, indices)

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


def main():
    parser = argparse.ArgumentParser(description='Evaluate all checkpoints on test sets')
    parser.add_argument('checkpoint_dir', type=str,
                       help='Directory containing checkpoints (will search recursively)')
    parser.add_argument('--output', type=str, default='checkpoint_results.csv',
                       help='Output CSV file (default: checkpoint_results.csv)')
    parser.add_argument('--test-set', type=str, action='append',
                       help='Test set path (can specify multiple times). If not specified, uses config.')
    parser.add_argument('--batch-size', type=int, default=512,
                       help='Batch size for evaluation (default: 512)')
    parser.add_argument('--num-workers', type=int, default=0,
                       help='Number of data loader workers (default: 0)')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Maximum samples per test set (default: None = all)')
    parser.add_argument('--pattern', type=str, default='*.ckpt',
                       help='Checkpoint filename pattern (default: *.ckpt)')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit number of checkpoints to evaluate (for testing)')

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

    print(f"Found {len(checkpoints)} checkpoints")

    # Limit checkpoints if specified
    if args.limit is not None:
        checkpoints = checkpoints[:args.limit]
        print(f"Limiting evaluation to first {args.limit} checkpoints")

    # Set up test sets
    if args.test_set:
        # Use specified test sets
        test_sets = [
            {'name': f'test{i}', 'path': path}
            for i, path in enumerate(args.test_set)
        ]
    else:
        # Use test sets from config
        try:
            config = OmegaConf.load('config/node_classifier.yaml')
            test_sets = [
                {'name': ts.name, 'path': ts.path}
                for ts in config.test_evaluation.test_sets
            ]
        except Exception as e:
            print(f"ERROR: Could not load test sets from config: {e}")
            print("Please specify test sets using --test-set")
            sys.exit(1)

    print(f"Test sets: {[ts['name'] for ts in test_sets]}")

    # Evaluate all checkpoints
    all_results = []
    print("\nEvaluating checkpoints...")

    for checkpoint_path in tqdm(checkpoints, desc="Checkpoints"):
        print(f"\n{checkpoint_path}")
        result = evaluate_checkpoint(
            checkpoint_path=checkpoint_path,
            test_sets=test_sets,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_samples=args.max_samples
        )

        if result is not None:
            all_results.append(result)
            # Print key metrics
            for test_set in test_sets:
                test_name = test_set['name']
                em_key = f'{test_name}_exact_match'
                if em_key in result:
                    print(f"  {test_name}: EM = {result[em_key]:.4f}")

    # Convert to DataFrame and save
    if all_results:
        df = pd.DataFrame(all_results)

        # Sort by validation exact match if available
        if 'val_em' in df.columns:
            df = df.sort_values('val_em', ascending=False)

        # Save to CSV
        df.to_csv(args.output, index=False)
        print(f"\n✓ Results saved to: {args.output}")
        print(f"  Total checkpoints evaluated: {len(df)}")

        # Print summary statistics
        print("\nSummary Statistics:")
        for col in df.columns:
            if any(metric in col for metric in ['exact_match', 'accuracy', 'f1', 'precision', 'recall']):
                if df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                    print(f"  {col}:")
                    print(f"    Mean: {df[col].mean():.4f}")
                    print(f"    Std:  {df[col].std():.4f}")
                    print(f"    Min:  {df[col].min():.4f}")
                    print(f"    Max:  {df[col].max():.4f}")
    else:
        print("\nERROR: No checkpoints successfully evaluated")
        sys.exit(1)


if __name__ == '__main__':
    main()
