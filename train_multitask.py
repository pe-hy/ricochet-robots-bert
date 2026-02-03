"""
Training script for multi-task Ricochet Robots models.

Trains all 4 tasks jointly: subgoal_label, helper_aggregate, target_pos, chosen_helper

Uses PyTorch Lightning, Hydra for configuration, and WandB for logging.
"""

import os
import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
import torch
from pathlib import Path

from model.models import MultiTaskConfig, create_multitask_model
from model.multitask_lightning_module import MultiTaskLightningModule
from utils.multitask_data_module import MultiTaskDataModule


@hydra.main(version_base=None, config_path="config", config_name="multitask")
def main(cfg: DictConfig):
    """
    Main training function.

    Args:
        cfg: Hydra configuration
    """
    # Print config
    print("=" * 80)
    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))
    print("=" * 80)

    # Set random seed for reproducibility
    pl.seed_everything(cfg.seed, workers=True)

    # Create data module
    data_module = MultiTaskDataModule(
        train_path=cfg.data.train_path,
        board_size=cfg.data.board_size,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        val_size=cfg.data.val_size,
        test_size=cfg.data.test_size,
        positional_encoding=cfg.data.positional_encoding,
        positional_encoding_kwargs=cfg.data.get('positional_encoding_kwargs', {}),
    )

    # Setup data module to compute feature_dim
    data_module.setup('fit')
    feature_dim = data_module.feature_dim
    print(f"Feature dimension: {feature_dim}")

    # Create model config
    model_config = MultiTaskConfig(
        feature_dim=feature_dim,
        d_model=cfg.model.d_model,
        nhead=cfg.model.nhead,
        num_layers=cfg.model.num_layers,
        dim_feedforward=cfg.model.dim_feedforward,
        dropout=cfg.model.dropout,
        activation=cfg.model.activation,
        architecture=cfg.model.architecture,
        num_comp_vectors=cfg.model.get('num_comp_vectors', 3),
        positional_encoding=cfg.data.positional_encoding,
        board_size=cfg.data.board_size,
        pos_encoding_dim=cfg.data.get('pos_encoding_dim', 0),
        pos_combine_method=cfg.data.get('pos_combine_method', 'concat'),
    )

    # Create model
    model = create_multitask_model(model_config)
    print(f"Created model: {cfg.model.architecture}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Calculate steps per epoch
    train_size = len(data_module.train_dataset)
    steps_per_epoch = train_size // cfg.data.batch_size

    # Create Lightning module
    lightning_module = MultiTaskLightningModule(
        model=model,
        model_config=model_config,
        max_lr=cfg.training.max_lr,
        weight_decay=cfg.training.weight_decay,
        warmup_epochs=cfg.training.warmup_epochs,
        total_epochs=cfg.training.max_epochs,
        steps_per_epoch=steps_per_epoch,
        task_weights=cfg.training.get('task_weights', None),
        log_predictions=cfg.training.get('log_predictions', True),
    )

    # Setup WandB logger
    wandb_logger = WandbLogger(
        project=cfg.wandb.project,
        name=cfg.wandb.name,
        save_dir=cfg.wandb.save_dir,
        log_model=cfg.wandb.get('log_model', False),
    )

    # Log config to wandb
    wandb_logger.experiment.config.update(OmegaConf.to_container(cfg, resolve=True))

    # Setup callbacks
    callbacks = []

    # Model checkpoint - save best model based on average validation accuracy
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.checkpoint.dirpath,
        filename=f"{cfg.model.architecture}-{{epoch:02d}}-{{val_avg_accuracy:.4f}}",
        monitor='val_avg_accuracy',
        mode='max',
        save_top_k=cfg.checkpoint.save_top_k,
        save_last=True,
        verbose=True,
    )
    callbacks.append(checkpoint_callback)

    # Early stopping
    if cfg.training.get('early_stopping', False):
        early_stop_callback = EarlyStopping(
            monitor='val_avg_accuracy',
            patience=cfg.training.get('early_stopping_patience', 10),
            mode='max',
            verbose=True,
        )
        callbacks.append(early_stop_callback)

    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_monitor)

    # Create trainer
    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator=cfg.training.accelerator,
        devices=cfg.training.devices,
        precision=cfg.training.get('precision', '32-true'),
        logger=wandb_logger,
        callbacks=callbacks,
        gradient_clip_val=cfg.training.get('gradient_clip_val', 0.0),
        accumulate_grad_batches=cfg.training.get('accumulate_grad_batches', 1),
        deterministic=cfg.get('deterministic', True),
        log_every_n_steps=cfg.training.get('log_every_n_steps', 50),
    )

    # Train
    print("=" * 80)
    print("Starting training...")
    print("=" * 80)
    trainer.fit(lightning_module, data_module)

    # Test
    if cfg.data.test_size > 0:
        print("=" * 80)
        print("Starting testing...")
        print("=" * 80)
        trainer.test(lightning_module, data_module)

    # Print best checkpoint path
    print("=" * 80)
    print(f"Best checkpoint: {checkpoint_callback.best_model_path}")
    print(f"Best val_avg_accuracy: {checkpoint_callback.best_model_score:.4f}")
    print("=" * 80)

    # Save final model
    if cfg.checkpoint.get('save_final', True):
        final_path = Path(cfg.checkpoint.dirpath) / f"{cfg.model.architecture}_final.ckpt"
        trainer.save_checkpoint(final_path)
        print(f"Saved final model to: {final_path}")

    wandb_logger.experiment.finish()


if __name__ == "__main__":
    main()
