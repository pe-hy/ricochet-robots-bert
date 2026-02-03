"""
PyTorch Lightning module for multi-task training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import Accuracy, Precision, Recall, F1Score, AUROC
from typing import Dict, Any, Optional
import wandb
import math

from model.models import BaseMultiTaskTransformer, MultiTaskConfig


class MultiTaskLightningModule(pl.LightningModule):
    """
    PyTorch Lightning module for multi-task node classification.

    Handles training, validation, testing with multi-task outputs.
    Computes per-task losses and metrics.
    """

    TASK_NAMES = ['subgoal_label', 'helper_aggregate', 'target_pos', 'chosen_helper']

    def __init__(
        self,
        model: BaseMultiTaskTransformer,
        model_config: MultiTaskConfig,
        max_lr: float = 1e-3,
        weight_decay: float = 0.01,
        warmup_epochs: int = 5,
        total_epochs: int = 100,
        steps_per_epoch: int = 100,
        task_weights: Optional[Dict[str, float]] = None,
        log_predictions: bool = True,
    ):
        """
        Args:
            model: Multi-task model instance
            model_config: Configuration for the model
            max_lr: Maximum learning rate (peak after warmup)
            weight_decay: Weight decay for optimizer
            warmup_epochs: Number of epochs for linear warmup
            total_epochs: Total number of training epochs (for cosine scheduler)
            steps_per_epoch: Number of training steps per epoch
            task_weights: Optional weights for each task (default: equal weights)
            log_predictions: Whether to log example predictions to wandb
        """
        super().__init__()

        # Save hyperparameters (convert model_config to dict for serialization)
        self.save_hyperparameters(ignore=['model'])

        # Store model and config
        self.model = model
        if isinstance(model_config, dict):
            self.model_config = MultiTaskConfig(**model_config)
        else:
            self.model_config = model_config
            self.hparams['model_config'] = model_config.to_dict()

        # Task weights (default: equal weights)
        if task_weights is None:
            task_weights = {task_name: 1.0 for task_name in self.TASK_NAMES}
        self.task_weights = task_weights

        # Loss function (per-task BCE with logits)
        self.criterion = nn.BCEWithLogitsLoss()

        # Metrics for each task (train)
        self.train_metrics = nn.ModuleDict({
            task_name: nn.ModuleDict({
                'accuracy': Accuracy(task='binary'),
            })
            for task_name in self.TASK_NAMES
        })

        # Metrics for each task (val)
        self.val_metrics = nn.ModuleDict({
            task_name: nn.ModuleDict({
                'accuracy': Accuracy(task='binary'),
                'precision': Precision(task='binary'),
                'recall': Recall(task='binary'),
                'f1': F1Score(task='binary'),
                'auroc': AUROC(task='binary'),
            })
            for task_name in self.TASK_NAMES
        })

        # Metrics for each task (test)
        self.test_metrics = nn.ModuleDict({
            task_name: nn.ModuleDict({
                'accuracy': Accuracy(task='binary'),
                'precision': Precision(task='binary'),
                'recall': Recall(task='binary'),
                'f1': F1Score(task='binary'),
                'auroc': AUROC(task='binary'),
            })
            for task_name in self.TASK_NAMES
        })

        # For logging
        self.log_predictions = log_predictions

    def forward(self, features: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        """Forward pass"""
        return self.model(features, attention_mask)

    def _compute_loss(
        self,
        logits_dict: Dict[str, torch.Tensor],
        labels_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute per-task losses.

        Args:
            logits_dict: Dict with task logits, each [batch, num_nodes, 1]
            labels_dict: Dict with task labels, each [batch, num_nodes]

        Returns:
            Dict with 'total' loss and per-task losses
        """
        losses = {}
        total_loss = 0.0

        for task_name in self.TASK_NAMES:
            logits = logits_dict[task_name].squeeze(-1)  # [batch, num_nodes]
            labels = labels_dict[task_name].float()       # [batch, num_nodes]

            task_loss = self.criterion(logits, labels)
            losses[task_name] = task_loss

            # Weighted sum
            total_loss += self.task_weights[task_name] * task_loss

        losses['total'] = total_loss
        return losses

    def _compute_metrics(
        self,
        logits_dict: Dict[str, torch.Tensor],
        labels_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Compute per-task metrics.

        Args:
            logits_dict: Dict with task logits, each [batch, num_nodes, 1]
            labels_dict: Dict with task labels, each [batch, num_nodes]

        Returns:
            Dict of dicts with per-task metrics
        """
        metrics = {}

        for task_name in self.TASK_NAMES:
            logits = logits_dict[task_name].squeeze(-1)  # [batch, num_nodes]
            labels = labels_dict[task_name]              # [batch, num_nodes]

            # Get probabilities and predictions
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).long()

            # Compute exact match per example
            exact_matches = (preds == labels).all(dim=1).float()

            # Flatten for node-level metrics
            probs_flat = probs.flatten()
            preds_flat = preds.flatten()
            labels_flat = labels.flatten()

            metrics[task_name] = {
                'probs': probs_flat,
                'preds': preds_flat,
                'labels': labels_flat,
                'exact_matches': exact_matches,
            }

        return metrics

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step"""
        features = batch['features']  # [batch, num_nodes, feature_dim]
        labels = batch['labels']      # Dict with 4 tasks, each [batch, num_nodes]

        # Forward pass
        logits = self(features)       # Dict with 4 tasks, each [batch, num_nodes, 1]

        # Compute losses
        losses = self._compute_loss(logits, labels)
        total_loss = losses['total']

        # Compute metrics
        metrics = self._compute_metrics(logits, labels)

        # Update and log per-task metrics
        for task_name in self.TASK_NAMES:
            task_metrics = metrics[task_name]

            # Update accuracy
            accuracy = self.train_metrics[task_name]['accuracy'](
                task_metrics['preds'],
                task_metrics['labels']
            )

            # Log metrics
            self.log(f'train/{task_name}/loss', losses[task_name], on_step=True, on_epoch=True, sync_dist=True)
            self.log(f'train/{task_name}/accuracy', accuracy, on_step=True, on_epoch=True, sync_dist=True)
            self.log(f'train/{task_name}/exact_match', task_metrics['exact_matches'].mean(),
                    on_step=True, on_epoch=True, sync_dist=True)

        # Log total loss
        self.log('train/loss', total_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        # Log average accuracy across tasks
        avg_accuracy = torch.stack([
            self.train_metrics[task_name]['accuracy'].compute()
            for task_name in self.TASK_NAMES
        ]).mean()
        self.log('train/avg_accuracy', avg_accuracy, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        # Compute combined exact match (all 4 tasks correct simultaneously)
        all_tasks_exact_match = torch.stack([
            metrics[task_name]['exact_matches']
            for task_name in self.TASK_NAMES
        ], dim=1)  # [batch, 4]

        # Check if ALL 4 tasks are correct for each example
        combined_exact_match = all_tasks_exact_match.all(dim=1).float().mean()  # [batch] -> scalar
        self.log('train/exact_match_all_tasks', combined_exact_match, on_step=True, on_epoch=True, sync_dist=True)

        return total_loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Validation step"""
        features = batch['features']
        labels = batch['labels']

        # Forward pass
        logits = self(features)

        # Compute losses
        losses = self._compute_loss(logits, labels)
        total_loss = losses['total']

        # Compute metrics
        metrics = self._compute_metrics(logits, labels)

        # Update and log per-task metrics
        for task_name in self.TASK_NAMES:
            task_metrics = metrics[task_name]

            # Update metrics
            self.val_metrics[task_name]['accuracy'](task_metrics['preds'], task_metrics['labels'])
            self.val_metrics[task_name]['precision'](task_metrics['preds'], task_metrics['labels'])
            self.val_metrics[task_name]['recall'](task_metrics['preds'], task_metrics['labels'])
            self.val_metrics[task_name]['f1'](task_metrics['preds'], task_metrics['labels'])
            self.val_metrics[task_name]['auroc'](task_metrics['probs'], task_metrics['labels'])

            # Log metrics
            self.log(f'val/{task_name}/loss', losses[task_name], on_step=False, on_epoch=True, sync_dist=True)
            self.log(f'val/{task_name}/accuracy', self.val_metrics[task_name]['accuracy'],
                    on_step=False, on_epoch=True, sync_dist=True)
            self.log(f'val/{task_name}/precision', self.val_metrics[task_name]['precision'],
                    on_step=False, on_epoch=True, sync_dist=True)
            self.log(f'val/{task_name}/recall', self.val_metrics[task_name]['recall'],
                    on_step=False, on_epoch=True, sync_dist=True)
            self.log(f'val/{task_name}/f1', self.val_metrics[task_name]['f1'],
                    on_step=False, on_epoch=True, sync_dist=True)
            self.log(f'val/{task_name}/auroc', self.val_metrics[task_name]['auroc'],
                    on_step=False, on_epoch=True, sync_dist=True)
            self.log(f'val/{task_name}/exact_match', task_metrics['exact_matches'].mean(),
                    on_step=False, on_epoch=True, sync_dist=True)

        # Log total loss
        self.log('val/loss', total_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        # Log average accuracy across tasks
        avg_accuracy = torch.stack([
            self.val_metrics[task_name]['accuracy'].compute()
            for task_name in self.TASK_NAMES
        ]).mean()
        self.log('val/avg_accuracy', avg_accuracy, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        # Also log with underscore for compatibility with checkpointing
        self.log('val_avg_accuracy', avg_accuracy, on_step=False, on_epoch=True, sync_dist=True)

        # Compute combined exact match (all 4 tasks correct simultaneously)
        all_tasks_exact_match = torch.stack([
            metrics[task_name]['exact_matches']
            for task_name in self.TASK_NAMES
        ], dim=1)  # [batch, 4]

        # Check if ALL 4 tasks are correct for each example
        combined_exact_match = all_tasks_exact_match.all(dim=1).float().mean()  # [batch] -> scalar
        self.log('val/exact_match_all_tasks', combined_exact_match, on_step=False, on_epoch=True,
                prog_bar=True, sync_dist=True)

        # Log example predictions to wandb
        if self.log_predictions and batch_idx == 0:
            self._log_predictions(batch, logits)

        return {'val_loss': total_loss}

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Test step"""
        features = batch['features']
        labels = batch['labels']

        # Forward pass
        logits = self(features)

        # Compute losses
        losses = self._compute_loss(logits, labels)
        total_loss = losses['total']

        # Compute metrics
        metrics = self._compute_metrics(logits, labels)

        # Update and log per-task metrics
        for task_name in self.TASK_NAMES:
            task_metrics = metrics[task_name]

            # Update metrics
            self.test_metrics[task_name]['accuracy'](task_metrics['preds'], task_metrics['labels'])
            self.test_metrics[task_name]['precision'](task_metrics['preds'], task_metrics['labels'])
            self.test_metrics[task_name]['recall'](task_metrics['preds'], task_metrics['labels'])
            self.test_metrics[task_name]['f1'](task_metrics['preds'], task_metrics['labels'])
            self.test_metrics[task_name]['auroc'](task_metrics['probs'], task_metrics['labels'])

            # Log metrics
            self.log(f'test/{task_name}/loss', losses[task_name], on_step=False, on_epoch=True, sync_dist=True)
            self.log(f'test/{task_name}/accuracy', self.test_metrics[task_name]['accuracy'],
                    on_step=False, on_epoch=True, sync_dist=True)
            self.log(f'test/{task_name}/precision', self.test_metrics[task_name]['precision'],
                    on_step=False, on_epoch=True, sync_dist=True)
            self.log(f'test/{task_name}/recall', self.test_metrics[task_name]['recall'],
                    on_step=False, on_epoch=True, sync_dist=True)
            self.log(f'test/{task_name}/f1', self.test_metrics[task_name]['f1'],
                    on_step=False, on_epoch=True, sync_dist=True)
            self.log(f'test/{task_name}/auroc', self.test_metrics[task_name]['auroc'],
                    on_step=False, on_epoch=True, sync_dist=True)
            self.log(f'test/{task_name}/exact_match', task_metrics['exact_matches'].mean(),
                    on_step=False, on_epoch=True, sync_dist=True)

        # Log total loss
        self.log('test/loss', total_loss, on_step=False, on_epoch=True, sync_dist=True)

        # Log average accuracy across tasks
        avg_accuracy = torch.stack([
            self.test_metrics[task_name]['accuracy'].compute()
            for task_name in self.TASK_NAMES
        ]).mean()
        self.log('test/avg_accuracy', avg_accuracy, on_step=False, on_epoch=True, sync_dist=True)

        # Compute combined exact match (all 4 tasks correct simultaneously)
        all_tasks_exact_match = torch.stack([
            metrics[task_name]['exact_matches']
            for task_name in self.TASK_NAMES
        ], dim=1)  # [batch, 4]

        # Check if ALL 4 tasks are correct for each example
        combined_exact_match = all_tasks_exact_match.all(dim=1).float().mean()  # [batch] -> scalar
        self.log('test/exact_match_all_tasks', combined_exact_match, on_step=False, on_epoch=True, sync_dist=True)

        return {'test_loss': total_loss}

    def _log_predictions(self, batch: Dict[str, torch.Tensor], logits_dict: Dict[str, torch.Tensor]):
        """Log example predictions to wandb"""
        if not isinstance(self.logger, pl.loggers.WandbLogger):
            return

        features = batch['features']
        labels_dict = batch['labels']

        # Convert logits to probabilities
        probs_dict = {}
        for task_name, logits in logits_dict.items():
            probs_dict[task_name] = torch.sigmoid(logits.squeeze(-1))

        # Log first example in batch for each task
        for task_name in self.TASK_NAMES:
            example_labels = labels_dict[task_name][0].detach().to(torch.float32).cpu().numpy()
            example_probs = probs_dict[task_name][0].detach().to(torch.float32).cpu().numpy()

            # Create table
            columns = ['node_id', 'label', 'prediction', 'probability']
            data = []
            for i in range(min(10, len(example_labels))):
                data.append([
                    i,
                    int(example_labels[i]),
                    int(example_probs[i] > 0.5),
                    float(example_probs[i])
                ])

            self.logger.experiment.log({
                f'predictions_{task_name}': wandb.Table(columns=columns, data=data),
                'global_step': self.global_step
            })

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.max_lr,
            weight_decay=self.hparams.weight_decay
        )

        # Calculate exact training steps from config
        total_steps = self.hparams.steps_per_epoch * self.hparams.total_epochs
        warmup_steps = self.hparams.steps_per_epoch * self.hparams.warmup_epochs

        def lr_lambda(current_step):
            if current_step < warmup_steps:
                # Linear warmup
                return float(current_step) / float(max(1, warmup_steps))
            else:
                # Cosine annealing
                progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
                return 0.5 * (1.0 + math.cos(progress * math.pi))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1
            }
        }
