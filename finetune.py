"""
Finetuning script for multi-task Ricochet Robots models.

Loads a pre-trained checkpoint and finetunes with teacher forcing.
Logs all metrics to WandB as if training from scratch.

Usage:
    python finetune.py checkpoint_path=/path/to/checkpoint.ckpt
    python finetune.py checkpoint_path=/path/to/checkpoint.ckpt training.max_lr=1e-4 training.max_epochs=20
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


@hydra.main(version_base=None, config_path="config", config_name="finetune")
def main(cfg: DictConfig):
    """
    Main finetuning function.

    Args:
        cfg: Hydra configuration
    """
    # Print config
    print("=" * 80)
    print("Finetuning Configuration:")
    print(OmegaConf.to_yaml(cfg))
    print("=" * 80)

    # Resolve checkpoint path
    checkpoint_path = Path(cfg.checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading checkpoint: {checkpoint_path}")

    # Load checkpoint to extract model config
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model_config_dict = ckpt['hyper_parameters']['model_config']
    model_config = MultiTaskConfig(**model_config_dict)

    print(f"Loaded model config: architecture={model_config.architecture}, "
          f"d_model={model_config.d_model}, num_layers={model_config.num_layers}")

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

    # Create model from saved config
    model = create_multitask_model(model_config)

    # Load model weights from checkpoint
    model_state_dict = {}
    for key, value in ckpt['state_dict'].items():
        # Lightning saves as "model.xxx", strip the prefix
        if key.startswith('model.'):
            model_state_dict[key[len('model.'):]] = value

    missing, unexpected = model.load_state_dict(model_state_dict, strict=False)
    if missing:
        print(f"Warning: Missing keys in checkpoint: {missing}")
    if unexpected:
        print(f"Warning: Unexpected keys in checkpoint: {unexpected}")

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Loaded model: {model_config.architecture} ({param_count:,} parameters)")

    # Calculate steps per epoch (account for gradient accumulation)
    train_size = len(data_module.train_dataset)
    accumulate = cfg.training.get('accumulate_grad_batches', 1)
    steps_per_epoch = train_size // (cfg.data.batch_size * accumulate)

    # Create Lightning module with finetuning training params
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

    # Log config to wandb (include checkpoint path for traceability)
    wandb_config = OmegaConf.to_container(cfg, resolve=True)
    wandb_config['finetuned_from'] = str(checkpoint_path)
    wandb_config['model'] = model_config.to_dict()
    wandb_logger.experiment.config.update(wandb_config)

    # Setup callbacks
    callbacks = []

    # Model checkpoint
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.checkpoint.dirpath,
        filename=f"{model_config.architecture}-finetune-{{epoch:02d}}-{{val_exact_match_all_tasks:.4f}}",
        monitor='val_exact_match_all_tasks',
        mode='max',
        save_top_k=cfg.checkpoint.save_top_k,
        save_last=True,
        verbose=True,
    )
    callbacks.append(checkpoint_callback)

    # Early stopping
    if cfg.training.get('early_stopping', False):
        early_stop_callback = EarlyStopping(
            monitor='val_exact_match_all_tasks',
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

    # Finetune
    print("=" * 80)
    print("Starting finetuning...")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Learning rate: {cfg.training.max_lr}")
    print(f"  Epochs: {cfg.training.max_epochs}")
    print(f"  Warmup epochs: {cfg.training.warmup_epochs}")
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
    print(f"Best val_exact_match_all_tasks: {checkpoint_callback.best_model_score:.4f}")
    print("=" * 80)

    # Save final model
    if cfg.checkpoint.get('save_final', True):
        final_path = Path(cfg.checkpoint.dirpath) / f"{model_config.architecture}_finetune_final.ckpt"
        trainer.save_checkpoint(final_path)
        print(f"Saved final model to: {final_path}")

    wandb_logger.experiment.finish()


if __name__ == "__main__":
    main()
