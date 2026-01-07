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
import json
import pickle
import numpy as np
from tqdm import tqdm

from model.node_classifier import NodeClassifierConfig
from model.lightning_module import NodeClassifierLightningModule
from utils.data_module import RicochetRobotsDataModule, RicochetRobotsDataset


# Register custom OmegaConf resolver to read num_examples from JSON
def get_num_examples_from_json(json_path: str) -> int:
    """Read num_examples from dataset JSON metadata"""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        return data.get('metadata', {}).get('num_examples', 0)
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return 0

OmegaConf.register_new_resolver("get_num_examples", get_num_examples_from_json)


def evaluate_on_test_set(
    checkpoint_path: str,
    test_data_path: str,
    output_path: str,
    board_size: int = 16,
    threshold: float = 0.5,
    positional_encoding: str = 'onehot',
    positional_encoding_kwargs: dict = None,
    log_to_wandb: bool = True
):
    """
    Evaluate the best model on held-out test set and save predictions.

    Args:
        checkpoint_path: Path to best checkpoint
        test_data_path: Path to test.json
        output_path: Path to save predictions pickle
        board_size: Board size
        threshold: Prediction threshold
        positional_encoding: Type of positional encoding used during training
        positional_encoding_kwargs: Kwargs for positional encoding
        log_to_wandb: Whether to log metrics to WandB

    Returns:
        Dictionary of metrics
    """
    print("\n" + "=" * 80)
    print("EVALUATING ON HELD-OUT TEST SET")
    print("=" * 80)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Test data: {test_data_path}")
    print(f"Output: {output_path}")
    print("=" * 80)

    # Check if test data exists
    if not Path(test_data_path).exists():
        print(f"WARNING: Test data not found at {test_data_path}. Skipping evaluation.")
        return None

    # Load model from checkpoint
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = NodeClassifierLightningModule.load_from_checkpoint(checkpoint_path)
    model = model.to(device)
    model.eval()

    # Load test dataset
    test_dataset = RicochetRobotsDataset(
        data_path=test_data_path,
        board_size=board_size,
        positional_encoding=positional_encoding,
        positional_encoding_kwargs=positional_encoding_kwargs or {}
    )

    print(f"Test dataset size: {len(test_dataset)} examples")

    examples = []

    # Metrics tracking
    total_nodes = 0
    correct_nodes = 0
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    exact_matches = 0

    with torch.no_grad():
        for idx in tqdm(range(len(test_dataset)), desc="Evaluating"):
            sample = test_dataset[idx]
            features = sample['features']  # [num_nodes, feature_dim]
            labels = sample['labels']      # [num_nodes]

            # Get scored_subgoals from the raw data
            raw_example = test_dataset.data[idx]
            scored_subgoals = raw_example.get('scored_subgoals', {})

            # Convert to batch format
            features_batch = features.unsqueeze(0).to(device)  # [1, num_nodes, feature_dim]

            # Get predictions
            logits = model(features_batch)  # [1, num_nodes, 1]
            probs = torch.sigmoid(logits.squeeze(-1))  # [1, num_nodes]
            preds = (probs > threshold).long()  # [1, num_nodes]

            # Convert to numpy
            preds_np = preds.squeeze(0).cpu().numpy()  # [num_nodes]
            labels_np = labels.cpu().numpy()  # [num_nodes]
            features_np = features.cpu().numpy()  # [num_nodes, feature_dim]

            # Compute metrics for this example
            total_nodes += len(preds_np)
            correct_nodes += (preds_np == labels_np).sum()

            for pred, label in zip(preds_np, labels_np):
                if pred == 1 and label == 1:
                    true_positives += 1
                elif pred == 1 and label == 0:
                    false_positives += 1
                elif pred == 0 and label == 1:
                    false_negatives += 1

            # Check exact match (all nodes correct)
            if (preds_np == labels_np).all():
                exact_matches += 1

            # Extract (x, y) coordinates where prediction == 1
            predicted_coords = []
            for node_idx in range(len(preds_np)):
                if preds_np[node_idx] == 1:
                    x = node_idx % board_size
                    y = node_idx // board_size
                    predicted_coords.append((x, y))

            # Extract (x, y) coordinates where ground truth label == 1
            true_coords = []
            for node_idx in range(len(labels_np)):
                if labels_np[node_idx] == 1:
                    x = node_idx % board_size
                    y = node_idx // board_size
                    true_coords.append((x, y))

            # Store in format: (node_features, predicted_coords, true_coords, scored_subgoals)
            examples.append((features_np, predicted_coords, true_coords, scored_subgoals))

    # Calculate final metrics
    accuracy = correct_nodes / total_nodes if total_nodes > 0 else 0
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    exact_match_rate = exact_matches / len(examples) if len(examples) > 0 else 0

    metrics = {
        'test_final/accuracy': accuracy,
        'test_final/precision': precision,
        'test_final/recall': recall,
        'test_final/f1': f1,
        'test_final/exact_match': exact_match_rate,
        'test_final/exact_matches': exact_matches,
        'test_final/total_examples': len(examples)
    }

    # Print metrics
    print("\n" + "=" * 80)
    print("HELD-OUT TEST SET RESULTS")
    print("=" * 80)
    print(f"Accuracy:       {accuracy:.4f}")
    print(f"Precision:      {precision:.4f}")
    print(f"Recall:         {recall:.4f}")
    print(f"F1 Score:       {f1:.4f}")
    print(f"Exact Match:    {exact_match_rate:.4f} ({exact_matches}/{len(examples)})")
    print("=" * 80)

    # Log to WandB
    if log_to_wandb and wandb.run is not None:
        wandb.log(metrics)
        print("✓ Metrics logged to WandB")

    # Save predictions
    print(f"\nSaving predictions to: {output_path}")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump({'examples': examples}, f)

    print(f"✓ Saved {len(examples)} examples with predictions")
    print("=" * 80 + "\n")

    return metrics


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

    # Calculate steps per epoch from dataset size
    with open(cfg.data.train_path, 'r') as f:
        data = json.load(f)
    total_examples = data['metadata']['num_examples']
    train_size = total_examples - cfg.data.val_size - cfg.data.test_size
    steps_per_epoch = train_size // cfg.data.batch_size

    # Get positional encoding config
    pos_encoding_kwargs = OmegaConf.to_container(cfg.data.positional_encoding_kwargs)
    pos_encoding_dim = pos_encoding_kwargs.get('encoding_dim', 0) if pos_encoding_kwargs else 0
    pos_combine_method = pos_encoding_kwargs.get('combine_method', 'concat') if pos_encoding_kwargs else 'concat'

    # Create model config (use computed feature_dim from data_module)
    model_config = NodeClassifierConfig(
        feature_dim=data_module.feature_dim,
        d_model=cfg.model.d_model,
        nhead=cfg.model.nhead,
        num_layers=cfg.model.num_layers,
        dim_feedforward=cfg.model.dim_feedforward,
        dropout=cfg.model.dropout,
        activation=cfg.model.activation,
        positional_encoding=cfg.data.positional_encoding,
        board_size=cfg.data.board_size,
        pos_encoding_dim=pos_encoding_dim,
        pos_combine_method=pos_combine_method,
    )

    # Create Lightning module
    lightning_module = NodeClassifierLightningModule(
        model_config=model_config,
        max_lr=cfg.training.max_lr,
        weight_decay=cfg.training.weight_decay,
        warmup_epochs=cfg.training.warmup_epochs,
        total_epochs=cfg.trainer.epochs,
        steps_per_epoch=steps_per_epoch,
        pos_weight=cfg.training.pos_weight,
        log_predictions=cfg.training.log_predictions,
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
        auto_insert_metric_name=cfg.checkpoint.auto_insert_metric_name,
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
    lr_monitor = LearningRateMonitor(logging_interval='step')
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
    print(f"  - total_examples: {total_examples}")
    print(f"  - train_size: {train_size}")
    print(f"  - val_size: {cfg.data.val_size}")
    print(f"  - test_size: {cfg.data.test_size}")
    print(f"  - batch_size: {cfg.data.batch_size}")
    print(f"  - board_size: {cfg.data.board_size}")
    print(f"  - positional_encoding: {cfg.data.positional_encoding}")
    if pos_encoding_dim > 0:
        print(f"  - pos_encoding_dim: {pos_encoding_dim}")
        print(f"  - pos_combine_method: {pos_combine_method}")
    print(f"\nTraining:")
    print(f"  - max_lr: {cfg.training.max_lr}")
    print(f"  - epochs: {cfg.trainer.epochs}")
    print(f"  - warmup_epochs: {cfg.training.warmup_epochs}")
    print(f"  - steps_per_epoch: {steps_per_epoch}")
    print(f"  - total_steps: {steps_per_epoch * cfg.trainer.epochs}")
    print(f"  - warmup_steps: {steps_per_epoch * cfg.training.warmup_epochs}")
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

    # Evaluate on held-out test set and save predictions
    if cfg.test_evaluation.enabled:
        # Get best checkpoint path
        best_checkpoint = checkpoint_callback.best_model_path
        if best_checkpoint and Path(best_checkpoint).exists():
            evaluate_on_test_set(
                checkpoint_path=best_checkpoint,
                test_data_path=cfg.test_evaluation.test_data_path,
                output_path=cfg.test_evaluation.output_path,
                board_size=cfg.data.board_size,
                threshold=cfg.test_evaluation.threshold,
                positional_encoding=cfg.data.positional_encoding,
                positional_encoding_kwargs=OmegaConf.to_container(cfg.data.positional_encoding_kwargs)
            )
        else:
            print(f"WARNING: Best checkpoint not found. Skipping test set evaluation.")

    # Finish wandb run
    wandb.finish()

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
