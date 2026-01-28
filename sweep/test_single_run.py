"""Test single run to verify metrics match old behavior."""

import sys
import os
import json

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import wandb
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from omegaconf import OmegaConf

from model.node_classifier import NodeClassifierConfig
from model.lightning_module import NodeClassifierLightningModule
from utils.data_module import RicochetRobotsDataModule


def test_run():
    """Test run with fixed hyperparameters."""

    # Initialize WandB with test config
    wandb.init(
        project="ricochet-robots-test",
        config={
            'positional_encoding': 'learned',
            'encoding_dim': 128,
            'combine_method': 'joint',
            'nhead': 8,
            'num_layers': 16,
            'dim_feedforward': 1024,
            'dropout': 0.1,
            'max_lr': 0.005,
            'weight_decay': 0.1,
            'warmup_epochs': 20,
            'epochs': 5,  # Only 5 epochs for quick test
            'd_model': 256,
            'batch_size': 256,
            'gradient_clip_val': 1.0
        }
    )
    config = wandb.config

    # Load base config
    base_config = OmegaConf.load('config/node_classifier.yaml')

    # Load task config
    task_config = OmegaConf.load('config/task/subgoal_label.yaml')
    base_config.task = task_config

    # Update with test params
    if config.positional_encoding == 'learned':
        base_config.data.positional_encoding_kwargs = {
            'encoding_dim': config.encoding_dim,
            'combine_method': config.combine_method
        }
    else:
        base_config.data.positional_encoding_kwargs = {}

    base_config.data.positional_encoding = config.positional_encoding
    base_config.data.batch_size = config.batch_size

    base_config.model.d_model = config.d_model
    base_config.model.nhead = config.nhead
    base_config.model.num_layers = config.num_layers
    base_config.model.dim_feedforward = config.dim_feedforward
    base_config.model.dropout = config.dropout

    base_config.training.max_lr = config.max_lr
    base_config.training.weight_decay = config.weight_decay
    base_config.training.warmup_epochs = config.warmup_epochs

    base_config.trainer.epochs = config.epochs
    base_config.trainer.gradient_clip_val = config.gradient_clip_val

    # Setup data
    data_module = RicochetRobotsDataModule(
        train_path=base_config.data.train_path,
        board_size=base_config.data.board_size,
        batch_size=config.batch_size,
        num_workers=base_config.data.num_workers,
        val_size=base_config.data.val_size,
        test_size=base_config.data.test_size,
        positional_encoding=config.positional_encoding,
        positional_encoding_kwargs=OmegaConf.to_container(
            base_config.data.positional_encoding_kwargs, resolve=True
        ),
        task_config=OmegaConf.to_container(base_config.task, resolve=True)
    )

    # Calculate steps_per_epoch manually (don't call setup() - let Lightning do it)
    if base_config.data.train_path.endswith('.jsonl'):
        with open(base_config.data.train_path, 'r') as f:
            metadata_line = f.readline()
            metadata = json.loads(metadata_line)
            total_examples = metadata['num_examples']
    else:
        with open(base_config.data.train_path, 'r') as f:
            data = json.load(f)
        total_examples = data['metadata']['num_examples']

    train_size = total_examples - base_config.data.val_size - base_config.data.test_size
    steps_per_epoch = train_size // config.batch_size  # Use sweep config batch_size, not base_config

    # Get positional encoding config from test parameters
    pos_encoding_dim = config.encoding_dim if config.positional_encoding == 'learned' else 0
    pos_combine_method = config.combine_method if config.positional_encoding == 'learned' else 'concat'

    print(f"\nTest Run Configuration:")
    print(f"  Total examples: {total_examples}")
    print(f"  Train size: {train_size}")
    print(f"  Val size: {base_config.data.val_size}")
    print(f"  Batch size: {base_config.data.batch_size}")
    print(f"  Steps per epoch: {steps_per_epoch}")
    print(f"  Positional encoding: {base_config.data.positional_encoding}")
    print(f"  Pos encoding dim: {pos_encoding_dim}")
    print(f"  Pos combine method: {pos_combine_method}")

    # Create model
    model_config = NodeClassifierConfig(
        feature_dim=data_module.feature_dim,
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=config.num_layers,
        dim_feedforward=config.dim_feedforward,
        dropout=config.dropout,
        activation=base_config.model.activation,
        positional_encoding=config.positional_encoding,
        board_size=base_config.data.board_size,
        pos_encoding_dim=pos_encoding_dim,
        pos_combine_method=pos_combine_method
    )

    lightning_module = NodeClassifierLightningModule(
        model_config=model_config,
        max_lr=config.max_lr,
        weight_decay=config.weight_decay,
        warmup_epochs=config.warmup_epochs,
        total_epochs=config.epochs,
        steps_per_epoch=steps_per_epoch,
        pos_weight=base_config.training.pos_weight,
        log_predictions=False
    )

    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=f'./tmp/test_checkpoints/{wandb.run.id}',
            filename='epoch={epoch:02d}-val_em={val_exact_match:.4f}',
            monitor='val_exact_match',
            mode='max',
            save_top_k=1,
            save_last=False,
            auto_insert_metric_name=False
        ),
        LearningRateMonitor(logging_interval='step')
    ]

    # Trainer
    trainer = pl.Trainer(
        max_epochs=config.epochs,
        accelerator=base_config.trainer.accelerator,
        devices=base_config.trainer.devices,
        precision=base_config.trainer.precision,
        logger=WandbLogger(),
        callbacks=callbacks,
        log_every_n_steps=base_config.trainer.log_every_n_steps,
        val_check_interval=base_config.trainer.val_check_interval,
        gradient_clip_val=config.gradient_clip_val,
        deterministic=base_config.trainer.deterministic,
        benchmark=base_config.trainer.benchmark,
        enable_progress_bar=True
    )

    # Train
    print("\nStarting test run...")
    print("Expected at epoch 1:")
    print("  - train/accuracy: ~98%")
    print("  - val/accuracy: ~99%")
    print("  - val_exact_match: ~10-15%")
    print("\nIf you see different values, there's still a bug.\n")

    trainer.fit(lightning_module, data_module)

    print("\n" + "="*60)
    print("Test complete! Check WandB for metrics.")
    print("="*60)


if __name__ == '__main__':
    test_run()
