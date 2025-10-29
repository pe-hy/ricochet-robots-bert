"""
PyTorch Lightning module for training the node classifier.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import Accuracy, Precision, Recall, F1Score, AUROC
from typing import Dict, Any, Optional
import wandb

from model.node_classifier import NodeClassifierTransformer, NodeClassifierConfig


class NodeClassifierLightningModule(pl.LightningModule):
    """
    PyTorch Lightning module for node classification.
    Handles training, validation, testing, and logging.
    """

    def __init__(
        self,
        model_config: NodeClassifierConfig,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 1000,
        max_steps: int = 100000,
        pos_weight: Optional[float] = None,
        log_predictions: bool = True,
        log_every_n_steps: int = 100,
    ):
        """
        Args:
            model_config: Configuration for the model
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
            warmup_steps: Number of warmup steps for learning rate scheduler
            max_steps: Maximum number of training steps
            pos_weight: Positive class weight for imbalanced datasets (None = auto-compute)
            log_predictions: Whether to log example predictions to wandb
            log_every_n_steps: Log predictions every N steps
        """
        super().__init__()

        # Save hyperparameters
        self.save_hyperparameters(ignore=['model_config'])
        self.model_config = model_config

        # Create model
        self.model = NodeClassifierTransformer(**model_config.to_dict())

        # Loss function (will set pos_weight after first batch if None)
        self.pos_weight = pos_weight
        self.criterion = None  # Initialize in first forward pass

        # Metrics for binary classification (node-level)
        self.train_accuracy = Accuracy(task='binary')
        self.val_accuracy = Accuracy(task='binary')
        self.test_accuracy = Accuracy(task='binary')

        self.val_precision = Precision(task='binary')
        self.val_recall = Recall(task='binary')
        self.val_f1 = F1Score(task='binary')
        self.val_auroc = AUROC(task='binary')

        self.test_precision = Precision(task='binary')
        self.test_recall = Recall(task='binary')
        self.test_f1 = F1Score(task='binary')
        self.test_auroc = AUROC(task='binary')

        # Exact match metrics (example-level)
        # Tracks: for each example, are ALL nodes classified correctly?
        self.train_exact_matches = []
        self.val_exact_matches = []
        self.test_exact_matches = []

        # For logging
        self.log_predictions = log_predictions
        self.log_every_n_steps = log_every_n_steps

    def _init_criterion(self, labels: torch.Tensor):
        """Initialize loss function with appropriate pos_weight"""
        if self.criterion is None:
            if self.pos_weight is None:
                # Compute pos_weight from labels
                num_positive = labels.sum()
                num_negative = labels.numel() - num_positive
                pos_weight = num_negative / num_positive if num_positive > 0 else 1.0
                self.pos_weight = pos_weight.item()
                print(f"Auto-computed pos_weight: {self.pos_weight:.4f}")

            self.criterion = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor([self.pos_weight], device=self.device)
            )

    def forward(self, features: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        """Forward pass"""
        return self.model(features, attention_mask)

    def _compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute binary cross-entropy loss.

        Args:
            logits: [batch_size, num_nodes, 1]
            labels: [batch_size, num_nodes]

        Returns:
            loss: scalar
        """
        # Initialize criterion if needed
        self._init_criterion(labels)

        # Reshape for loss computation
        logits = logits.squeeze(-1)  # [batch_size, num_nodes]
        labels = labels.float()      # [batch_size, num_nodes]

        # Compute loss
        loss = self.criterion(logits, labels)
        return loss

    def _compute_metrics(self, logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute metrics for binary classification.

        Args:
            logits: [batch_size, num_nodes, 1]
            labels: [batch_size, num_nodes]

        Returns:
            Dictionary of metrics including exact_match
        """
        # Get probabilities and predictions
        probs = torch.sigmoid(logits.squeeze(-1))  # [batch_size, num_nodes]
        preds = (probs > 0.5).long()  # [batch_size, num_nodes]

        # Compute exact match per example
        # An example matches exactly if ALL nodes are classified correctly
        exact_matches = (preds == labels).all(dim=1).float()  # [batch_size]

        # Flatten for node-level metrics
        probs_flat = probs.flatten()
        preds_flat = preds.flatten()
        labels_flat = labels.flatten()

        return {
            'probs': probs_flat,
            'preds': preds_flat,
            'labels': labels_flat,
            'exact_matches': exact_matches,  # [batch_size]
            'preds_per_example': preds,  # [batch_size, num_nodes] for logging
            'labels_per_example': labels  # [batch_size, num_nodes] for logging
        }

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step"""
        features = batch['features']  # [batch_size, num_nodes, feature_dim]
        labels = batch['labels']      # [batch_size, num_nodes]

        # Forward pass
        logits = self(features)       # [batch_size, num_nodes, 1]

        # Compute loss
        loss = self._compute_loss(logits, labels)

        # Compute metrics
        metrics = self._compute_metrics(logits, labels)
        accuracy = self.train_accuracy(metrics['preds'], metrics['labels'])

        # Track exact matches
        self.train_exact_matches.extend(metrics['exact_matches'].cpu().tolist())
        exact_match_mean = metrics['exact_matches'].mean()

        # Log metrics
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train/accuracy', accuracy, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train/exact_match', exact_match_mean, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        # Log learning rate
        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('train/lr', lr, on_step=True, on_epoch=False, prog_bar=False, sync_dist=True)

        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Validation step"""
        features = batch['features']  # [batch_size, num_nodes, feature_dim]
        labels = batch['labels']      # [batch_size, num_nodes]

        # Forward pass
        logits = self(features)       # [batch_size, num_nodes, 1]

        # Compute loss
        loss = self._compute_loss(logits, labels)

        # Compute metrics
        metrics = self._compute_metrics(logits, labels)

        # Update node-level metrics
        self.val_accuracy(metrics['preds'], metrics['labels'])
        self.val_precision(metrics['preds'], metrics['labels'])
        self.val_recall(metrics['preds'], metrics['labels'])
        self.val_f1(metrics['preds'], metrics['labels'])
        self.val_auroc(metrics['probs'], metrics['labels'])

        # Track exact matches
        self.val_exact_matches.extend(metrics['exact_matches'].cpu().tolist())
        exact_match_mean = metrics['exact_matches'].mean()

        # Log metrics
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val/accuracy', self.val_accuracy, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log('val/precision', self.val_precision, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val/recall', self.val_recall, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val/f1', self.val_f1, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log('val/auroc', self.val_auroc, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val/exact_match', exact_match_mean, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        # Log example predictions to wandb
        if self.log_predictions and batch_idx == 0:
            self._log_predictions(batch, logits)

        return {'val_loss': loss}

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Test step"""
        features = batch['features']
        labels = batch['labels']

        # Forward pass
        logits = self(features)

        # Compute loss
        loss = self._compute_loss(logits, labels)

        # Compute metrics
        metrics = self._compute_metrics(logits, labels)

        # Update node-level metrics
        self.test_accuracy(metrics['preds'], metrics['labels'])
        self.test_precision(metrics['preds'], metrics['labels'])
        self.test_recall(metrics['preds'], metrics['labels'])
        self.test_f1(metrics['preds'], metrics['labels'])
        self.test_auroc(metrics['probs'], metrics['labels'])

        # Track exact matches
        self.test_exact_matches.extend(metrics['exact_matches'].cpu().tolist())
        exact_match_mean = metrics['exact_matches'].mean()

        # Log metrics
        self.log('test/loss', loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log('test/accuracy', self.test_accuracy, on_step=False, on_epoch=True, sync_dist=True)
        self.log('test/precision', self.test_precision, on_step=False, on_epoch=True, sync_dist=True)
        self.log('test/recall', self.test_recall, on_step=False, on_epoch=True, sync_dist=True)
        self.log('test/f1', self.test_f1, on_step=False, on_epoch=True, sync_dist=True)
        self.log('test/auroc', self.test_auroc, on_step=False, on_epoch=True, sync_dist=True)
        self.log('test/exact_match', exact_match_mean, on_step=False, on_epoch=True, sync_dist=True)

        return {'test_loss': loss}

    def _log_predictions(self, batch: Dict[str, torch.Tensor], logits: torch.Tensor):
        """Log example predictions to wandb"""
        if not isinstance(self.logger, pl.loggers.WandbLogger):
            return

        features = batch['features']  # [batch_size, num_nodes, feature_dim]
        labels = batch['labels']      # [batch_size, num_nodes]
        probs = torch.sigmoid(logits.squeeze(-1))  # [batch_size, num_nodes]

        # Log first example in batch
        example_features = features[0].cpu().numpy()  # [num_nodes, feature_dim]
        example_labels = labels[0].cpu().numpy()      # [num_nodes]
        example_probs = probs[0].cpu().numpy()        # [num_nodes]

        # Create table
        columns = ['node_id', 'label', 'prediction', 'probability']
        data = []
        for i in range(min(10, len(example_labels))):  # Log first 10 nodes
            data.append([
                i,
                int(example_labels[i]),
                int(example_probs[i] > 0.5),
                float(example_probs[i])
            ])

        self.logger.experiment.log({
            'predictions_table': wandb.Table(columns=columns, data=data),
            'global_step': self.global_step
        })

    def on_train_epoch_end(self):
        """Reset exact matches at end of training epoch"""
        self.train_exact_matches = []

    def on_validation_epoch_end(self):
        """Reset exact matches at end of validation epoch"""
        self.val_exact_matches = []

    def on_test_epoch_end(self):
        """Reset exact matches at end of test epoch"""
        self.test_exact_matches = []

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler"""
        # AdamW optimizer
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )

        # Learning rate scheduler with warmup
        def lr_lambda(step):
            if step < self.hparams.warmup_steps:
                # Linear warmup
                return step / max(1, self.hparams.warmup_steps)
            else:
                # Cosine decay
                progress = (step - self.hparams.warmup_steps) / max(1, self.hparams.max_steps - self.hparams.warmup_steps)
                return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.141592653589793)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1
            }
        }
