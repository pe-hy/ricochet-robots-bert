"""Evaluate best checkpoint from sweep on test sets."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import wandb
import pytorch_lightning as pl
from omegaconf import OmegaConf

from model.node_classifier import NodeClassifierConfig
from model.lightning_module import NodeClassifierLightningModule
from utils.data_module import RicochetRobotsDataModule


def evaluate_best_checkpoint(sweep_id: str, project: str, checkpoint_dir: str = None, max_samples: int = None):
    """
    Find best run from sweep and evaluate on test sets.

    Args:
        sweep_id: WandB sweep ID (e.g., 'mh1kmu6a')
        project: WandB project name (e.g., 'ricochet-robots-node-classifier-onehot')
        checkpoint_dir: Custom checkpoint directory (optional, defaults to ./tmp/sweep_checkpoints/{sweep_id}/{run_id})
        max_samples: Maximum number of samples to evaluate per test set (optional, None = all samples)
    """

    # Initialize wandb API
    api = wandb.Api()

    # Get sweep (need to include entity)
    entity = os.environ.get('WANDB_ENTITY', 'petr-hyner10')
    sweep = api.sweep(f"{entity}/{project}/{sweep_id}")
    print(f"\nSweep: {sweep.name} ({sweep_id})")
    print(f"Method: {sweep.config.get('method', 'unknown')}")

    # Get runs from this sweep, ordered by val_exact_match (descending)
    print(f"\nQuerying runs from sweep...")
    runs = api.runs(
        path=f"{entity}/{project}",
        filters={"sweep": sweep_id, "state": "finished"},
        order="-summary_metrics.val_exact_match"
    )

    # Get the best run (first in the ordered list)
    runs_list = list(runs)
    if not runs_list:
        print("ERROR: No finished runs found in sweep")
        return

    best_run = runs_list[0]

    print(f"\nBest run: {best_run.name} (ID: {best_run.id})")
    print(f"Run URL: {best_run.url}")
    print(f"Config: {best_run.config}")

    # Find checkpoint
    if checkpoint_dir is None:
        checkpoint_dir = f"./tmp/sweep_checkpoints/{sweep_id}/{best_run.id}"
    else:
        # If checkpoint_dir is provided, it should point to the sweep directory
        # We need to append the run_id to get the full path
        checkpoint_dir = os.path.join(checkpoint_dir, best_run.id)

    if not os.path.exists(checkpoint_dir):
        print(f"\nERROR: Checkpoint directory not found: {checkpoint_dir}")
        return

    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.ckpt')]
    if not checkpoints:
        print(f"\nERROR: No checkpoints found in {checkpoint_dir}")
        return

    # Use the best checkpoint (should be only one with save_top_k=1)
    checkpoint_path = os.path.join(checkpoint_dir, checkpoints[0])

    # Extract val_exact_match from checkpoint filename (e.g., "epoch=30-val_em=0.6797.ckpt")
    checkpoint_name = checkpoints[0]
    import re
    val_em_match = re.search(r'val_em=(\d+\.\d+)', checkpoint_name)
    if val_em_match:
        best_val_em = float(val_em_match.group(1))
        print(f"Val exact match: {best_val_em:.4f}")

    print(f"\nLoading checkpoint: {checkpoint_path}")

    # Load base config
    base_config = OmegaConf.load('config/node_classifier.yaml')
    task_config = OmegaConf.load('config/task/subgoal_label.yaml')
    base_config.task = task_config

    # Reconstruct data module with run config
    # Access config directly - wandb API returns nested dicts with "value" keys
    try:
        positional_encoding = best_run.config['positional_encoding']['value']
    except (KeyError, TypeError):
        positional_encoding = 'onehot'

    try:
        batch_size = best_run.config['batch_size']['value']
    except (KeyError, TypeError):
        batch_size = 512

    try:
        encoding_dim = best_run.config['encoding_dim']['value']
    except (KeyError, TypeError):
        encoding_dim = 128

    try:
        combine_method = best_run.config['combine_method']['value']
    except (KeyError, TypeError):
        combine_method = 'joint'

    if positional_encoding == 'learned':
        pos_encoding_kwargs = {
            'encoding_dim': encoding_dim,
            'combine_method': combine_method
        }
    else:
        pos_encoding_kwargs = {}

    # Create data module for each test set
    test_sets = base_config.test_evaluation.test_sets

    for test_set in test_sets:
        print(f"\n{'='*60}")
        print(f"Evaluating on: {test_set.name}")
        print(f"Path: {test_set.path}")
        print(f"{'='*60}")

        # Create data module for this test set
        data_module = RicochetRobotsDataModule(
            train_path=test_set.path,  # Use test set as "train" to load it
            board_size=base_config.data.board_size,
            batch_size=batch_size,
            num_workers=base_config.data.num_workers,
            val_size=0,  # No validation split needed
            test_size=0,  # Load all as train, then use as test
            positional_encoding=positional_encoding,
            positional_encoding_kwargs=pos_encoding_kwargs,
            task_config=OmegaConf.to_container(base_config.task, resolve=True)
        )
        data_module.setup()

        # Limit samples if max_samples is specified
        if max_samples is not None and max_samples > 0:
            from torch.utils.data import Subset
            dataset = data_module.train_dataset
            if len(dataset) > max_samples:
                print(f"Limiting evaluation to {max_samples} samples (out of {len(dataset)})")
                indices = list(range(max_samples))
                data_module.train_dataset = Subset(dataset, indices)

        # Load model from checkpoint
        model = NodeClassifierLightningModule.load_from_checkpoint(
            checkpoint_path
        )

        # Create trainer for testing
        trainer = pl.Trainer(
            accelerator='auto',
            devices=1,
            logger=False
        )

        # Test using train_dataloader (since we loaded test set as train)
        test_results = trainer.test(model, data_module.train_dataloader())

        print(f"\nResults for {test_set.name}:")
        for key, value in test_results[0].items():
            print(f"  {key}: {value:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate best checkpoint from sweep')
    parser.add_argument('sweep_id', type=str, help='WandB sweep ID (e.g., mh1kmu6a)')
    parser.add_argument('--project', type=str,
                       default='ricochet-robots-node-classifier-onehot',
                       help='WandB project name')
    parser.add_argument('--checkpoint-dir', type=str, default=None,
                       help='Custom checkpoint directory (default: ./tmp/sweep_checkpoints/{sweep_id}/{run_id})')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Maximum number of samples to evaluate per test set (default: None = all samples)')

    args = parser.parse_args()

    # Set wandb entity (username)
    os.environ['WANDB_ENTITY'] = 'petr-hyner10'

    evaluate_best_checkpoint(args.sweep_id, args.project, args.checkpoint_dir, args.max_samples)
