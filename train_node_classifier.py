"""
Training script for Ricochet Robots node classifier.

Usage:
    python train_node_classifier.py --config config/node_classifier.yaml
"""

import argparse
import yaml
from pathlib import Path
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
import wandb

from model.node_classifier import NodeClassifierConfig
from model.lightning_module import NodeClassifierLightningModule
from utils.data_module import RicochetRobotsDataModule


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main(config_path: str = None, **kwargs):
    """
    Main training function.

    Args:
        config_path: Path to config YAML file
        **kwargs: Override config values
    """
    # Load config
    if config_path is not None:
        config = load_config(config_path)
    else:
        config = {}

    # Override with kwargs
    config.update(kwargs)

    # Set random seed for reproducibility
    seed = config.get('seed', 42)
    pl.seed_everything(seed, workers=True)

    # Initialize wandb
    wandb_config = config.get('wandb', {})
    logger = WandbLogger(
        project=wandb_config.get('project', 'ricochet-robots-node-classifier'),
        name=wandb_config.get('name', None),
        config=config,
        save_dir=wandb_config.get('save_dir', './wandb_logs'),
        log_model=wandb_config.get('log_model', True),
    )

    # Log code to wandb
    if wandb_config.get('log_code', True):
        wandb.run.log_code(
            root='.',
            include_fn=lambda path: path.endswith('.py') or path.endswith('.yaml')
        )

    # Create data module
    data_config = config.get('data', {})
    data_module = RicochetRobotsDataModule(
        train_path=data_config.get('train_path', 'data/ricochet_data/dataset.json'),
        val_path=data_config.get('val_path', None),
        test_path=data_config.get('test_path', None),
        board_size=data_config.get('board_size', 16),
        batch_size=data_config.get('batch_size', 32),
        num_workers=data_config.get('num_workers', 4),
        train_val_split=data_config.get('train_val_split', 0.8),
        positional_encoding=data_config.get('positional_encoding', 'onehot'),
        positional_encoding_kwargs=data_config.get('positional_encoding_kwargs', {})
    )

    # Create model config (use computed feature_dim from data_module)
    model_config_dict = config.get('model', {})
    model_config = NodeClassifierConfig(
        feature_dim=data_module.feature_dim,  # Computed automatically based on positional encoding
        d_model=model_config_dict.get('d_model', 256),
        nhead=model_config_dict.get('nhead', 8),
        num_layers=model_config_dict.get('num_layers', 6),
        dim_feedforward=model_config_dict.get('dim_feedforward', 1024),
        dropout=model_config_dict.get('dropout', 0.1),
        activation=model_config_dict.get('activation', 'gelu'),
    )

    # Create Lightning module
    training_config = config.get('training', {})
    lightning_module = NodeClassifierLightningModule(
        model_config=model_config,
        learning_rate=training_config.get('learning_rate', 1e-4),
        weight_decay=training_config.get('weight_decay', 0.01),
        warmup_steps=training_config.get('warmup_steps', 1000),
        max_steps=training_config.get('max_steps', 100000),
        pos_weight=training_config.get('pos_weight', None),
        log_predictions=training_config.get('log_predictions', True),
        log_every_n_steps=training_config.get('log_every_n_steps', 100),
    )

    # Create callbacks
    callbacks = []

    # Checkpoint callback
    checkpoint_config = config.get('checkpoint', {})
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_config.get('dirpath', './checkpoints'),
        filename=checkpoint_config.get('filename', 'node_classifier-{epoch:02d}-{val/f1:.4f}'),
        monitor=checkpoint_config.get('monitor', 'val/f1'),
        mode=checkpoint_config.get('mode', 'max'),
        save_top_k=checkpoint_config.get('save_top_k', 3),
        save_last=checkpoint_config.get('save_last', True),
        verbose=True
    )
    callbacks.append(checkpoint_callback)

    # Early stopping callback
    early_stop_config = config.get('early_stopping', {})
    if early_stop_config.get('enabled', True):
        early_stopping_callback = EarlyStopping(
            monitor=early_stop_config.get('monitor', 'val/f1'),
            patience=early_stop_config.get('patience', 10),
            mode=early_stop_config.get('mode', 'max'),
            verbose=True
        )
        callbacks.append(early_stopping_callback)

    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_monitor)

    # Create trainer
    trainer_config = config.get('trainer', {})
    trainer = pl.Trainer(
        max_epochs=trainer_config.get('max_epochs', 100),
        max_steps=trainer_config.get('max_steps', -1),
        accelerator=trainer_config.get('accelerator', 'auto'),
        devices=trainer_config.get('devices', 'auto'),
        strategy=trainer_config.get('strategy', 'auto'),
        precision=trainer_config.get('precision', '32-true'),
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=trainer_config.get('log_every_n_steps', 50),
        val_check_interval=trainer_config.get('val_check_interval', 1.0),
        gradient_clip_val=trainer_config.get('gradient_clip_val', 1.0),
        accumulate_grad_batches=trainer_config.get('accumulate_grad_batches', 1),
        deterministic=trainer_config.get('deterministic', False),
        benchmark=trainer_config.get('benchmark', True),
    )

    # Print configuration
    print("=" * 80)
    print("TRAINING CONFIGURATION")
    print("=" * 80)
    print(f"Model: Transformer Node Classifier")
    print(f"  - d_model: {model_config.d_model}")
    print(f"  - num_layers: {model_config.num_layers}")
    print(f"  - nhead: {model_config.nhead}")
    print(f"  - feature_dim: {model_config.feature_dim}")
    print(f"\nData:")
    print(f"  - train_path: {data_config.get('train_path')}")
    print(f"  - batch_size: {data_config.get('batch_size')}")
    print(f"  - board_size: {data_config.get('board_size')}")
    print(f"\nTraining:")
    print(f"  - learning_rate: {training_config.get('learning_rate')}")
    print(f"  - max_epochs: {trainer_config.get('max_epochs')}")
    print(f"  - warmup_steps: {training_config.get('warmup_steps')}")
    print(f"  - weight_decay: {training_config.get('weight_decay')}")
    print(f"\nWandB:")
    print(f"  - project: {wandb_config.get('project')}")
    print(f"  - name: {wandb_config.get('name', 'auto')}")
    print("=" * 80)

    # Count parameters
    total_params = sum(p.numel() for p in lightning_module.parameters())
    trainable_params = sum(p.numel() for p in lightning_module.parameters() if p.requires_grad)
    print(f"\nModel Parameters:")
    print(f"  - Total: {total_params:,}")
    print(f"  - Trainable: {trainable_params:,}")
    print("=" * 80)

    # Train model
    print("\nStarting training...")
    trainer.fit(lightning_module, datamodule=data_module)

    # Test model
    if data_module.test_dataset is not None:
        print("\nTesting model...")
        trainer.test(lightning_module, datamodule=data_module)

    # Finish wandb run
    wandb.finish()

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Ricochet Robots node classifier')
    parser.add_argument(
        '--config',
        type=str,
        default='config/node_classifier.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--data_path',
        type=str,
        default=None,
        help='Override data path'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=None,
        help='Override batch size'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=None,
        help='Override learning rate'
    )
    parser.add_argument(
        '--max_epochs',
        type=int,
        default=None,
        help='Override max epochs'
    )

    args = parser.parse_args()

    # Build kwargs from args
    kwargs = {}
    if args.data_path is not None:
        kwargs['data'] = {'train_path': args.data_path}
    if args.batch_size is not None:
        kwargs.setdefault('data', {})['batch_size'] = args.batch_size
    if args.learning_rate is not None:
        kwargs['training'] = {'learning_rate': args.learning_rate}
    if args.max_epochs is not None:
        kwargs['trainer'] = {'max_epochs': args.max_epochs}

    main(config_path=args.config, **kwargs)
