"""
Training script for Ricochet Robots node classifier using Hydra.

Usage:
    # Basic training
    python train_node_classifier.py

    # Override config values
    python train_node_classifier.py trainer.epochs=50 training.max_lr=2e-3

    # Change data split
    python train_node_classifier.py data.val_size=32 data.test_size=32

    # Use different config
    python train_node_classifier.py --config-name=my_config
"""

import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
import wandb

from model.node_classifier import NodeClassifierConfig
from model.lightning_module import NodeClassifierLightningModule
from utils.data_module import RicochetRobotsDataModule


@hydra.main(version_base=None, config_path="config", config_name="node_classifier")
def main(cfg: DictConfig) -> None:
    """
    Main training function using Hydra configuration.

    Args:
        cfg: Hydra configuration object
    """
    # Print config
    print("=" * 80)
    print("CONFIGURATION")
    print("=" * 80)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 80)

    # Set random seed for reproducibility
    pl.seed_everything(cfg.seed, workers=True)

    # Initialize wandb
    logger = WandbLogger(
        project=cfg.wandb.project,
        name=cfg.wandb.name,
        config=OmegaConf.to_container(cfg, resolve=True),
        save_dir=cfg.wandb.save_dir,
        log_model=cfg.wandb.log_model,
    )

    # Log code to wandb
    if cfg.wandb.log_code:
        wandb.run.log_code(
            root='.',
            include_fn=lambda path: path.endswith('.py') or path.endswith('.yaml')
        )

    # Create data module
    data_module = RicochetRobotsDataModule(
        train_path=cfg.data.train_path,
        board_size=cfg.data.board_size,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        val_size=cfg.data.val_size,
        test_size=cfg.data.test_size,
        positional_encoding=cfg.data.positional_encoding,
        positional_encoding_kwargs=OmegaConf.to_container(cfg.data.positional_encoding_kwargs)
    )

    # Create model config (use computed feature_dim from data_module)
    model_config = NodeClassifierConfig(
        feature_dim=data_module.feature_dim,
        d_model=cfg.model.d_model,
        nhead=cfg.model.nhead,
        num_layers=cfg.model.num_layers,
        dim_feedforward=cfg.model.dim_feedforward,
        dropout=cfg.model.dropout,
        activation=cfg.model.activation,
    )

    # Create Lightning module
    lightning_module = NodeClassifierLightningModule(
        model_config=model_config,
        max_lr=cfg.training.max_lr,
        weight_decay=cfg.training.weight_decay,
        warmup_epochs=cfg.training.warmup_epochs,
        total_epochs=cfg.trainer.epochs,
        pos_weight=cfg.training.pos_weight,
        log_predictions=cfg.training.log_predictions,
        log_every_n_steps=cfg.training.log_every_n_steps,
    )

    # Create callbacks
    callbacks = []

    # Checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.checkpoint.dirpath,
        filename=cfg.checkpoint.filename,
        monitor=cfg.checkpoint.monitor,
        mode=cfg.checkpoint.mode,
        save_top_k=cfg.checkpoint.save_top_k,
        save_last=cfg.checkpoint.save_last,
        verbose=True
    )
    callbacks.append(checkpoint_callback)

    # Early stopping callback
    if cfg.early_stopping.enabled:
        early_stopping_callback = EarlyStopping(
            monitor=cfg.early_stopping.monitor,
            patience=cfg.early_stopping.patience,
            mode=cfg.early_stopping.mode,
            verbose=True
        )
        callbacks.append(early_stopping_callback)

    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    callbacks.append(lr_monitor)

    # Create trainer
    trainer = pl.Trainer(
        max_epochs=cfg.trainer.epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        strategy=cfg.trainer.strategy,
        precision=cfg.trainer.precision,
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        val_check_interval=cfg.trainer.val_check_interval,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
        deterministic=cfg.trainer.deterministic,
        benchmark=cfg.trainer.benchmark,
    )

    # Print configuration summary
    print("\n" + "=" * 80)
    print("TRAINING CONFIGURATION")
    print("=" * 80)
    print(f"Model: Transformer Node Classifier")
    print(f"  - d_model: {model_config.d_model}")
    print(f"  - num_layers: {model_config.num_layers}")
    print(f"  - nhead: {model_config.nhead}")
    print(f"  - feature_dim: {model_config.feature_dim}")
    print(f"\nData:")
    print(f"  - train_path: {cfg.data.train_path}")
    print(f"  - batch_size: {cfg.data.batch_size}")
    print(f"  - board_size: {cfg.data.board_size}")
    print(f"  - val_size: {cfg.data.val_size}")
    print(f"  - test_size: {cfg.data.test_size}")
    print(f"\nTraining:")
    print(f"  - max_lr: {cfg.training.max_lr}")
    print(f"  - epochs: {cfg.trainer.epochs}")
    print(f"  - warmup_epochs: {cfg.training.warmup_epochs}")
    print(f"  - weight_decay: {cfg.training.weight_decay}")
    print(f"\nWandB:")
    print(f"  - project: {cfg.wandb.project}")
    print(f"  - name: {cfg.wandb.name if cfg.wandb.name else 'auto'}")
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
    main()
