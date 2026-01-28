"""
Validation script to check all parameters before running sweep.
Run this before submitting: python sweep/validate.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from omegaconf import OmegaConf
from model.node_classifier import NodeClassifierConfig
from model.lightning_module import NodeClassifierLightningModule
from utils.data_module import RicochetRobotsDataModule
import inspect

print("="*80)
print("VALIDATING ALL FUNCTION SIGNATURES")
print("="*80)

# Load config
base_config = OmegaConf.load('config/node_classifier.yaml')
task_config = OmegaConf.load('config/task/subgoal_label.yaml')
base_config.task = task_config

# Test 1: RicochetRobotsDataModule
print("\n1. RicochetRobotsDataModule")
print("-" * 40)
sig = inspect.signature(RicochetRobotsDataModule.__init__)
print(f"Expected parameters: {list(sig.parameters.keys())[1:]}")  # Skip 'self'

try:
    data_module = RicochetRobotsDataModule(
        train_path=base_config.data.train_path,
        board_size=base_config.data.board_size,
        batch_size=base_config.data.batch_size,
        num_workers=base_config.data.num_workers,
        val_size=base_config.data.val_size,
        test_size=base_config.data.test_size,
        positional_encoding=base_config.data.positional_encoding,
        positional_encoding_kwargs=OmegaConf.to_container(
            base_config.data.positional_encoding_kwargs, resolve=True
        ),
        task_config=OmegaConf.to_container(base_config.task, resolve=True)
    )
    data_module.setup()
    print("✅ RicochetRobotsDataModule - OK")
except Exception as e:
    print(f"❌ RicochetRobotsDataModule - FAILED: {e}")
    sys.exit(1)

# Test 2: NodeClassifierConfig
print("\n2. NodeClassifierConfig")
print("-" * 40)
sig = inspect.signature(NodeClassifierConfig.__init__)
print(f"Expected parameters: {list(sig.parameters.keys())[1:]}")

try:
    model_config = NodeClassifierConfig(
        feature_dim=data_module.feature_dim,
        d_model=base_config.model.d_model,
        nhead=base_config.model.nhead,
        num_layers=base_config.model.num_layers,
        dim_feedforward=base_config.model.dim_feedforward,
        dropout=base_config.model.dropout,
        activation=base_config.model.activation
    )
    print("✅ NodeClassifierConfig - OK")
except Exception as e:
    print(f"❌ NodeClassifierConfig - FAILED: {e}")
    sys.exit(1)

# Test 3: NodeClassifierLightningModule
print("\n3. NodeClassifierLightningModule")
print("-" * 40)
sig = inspect.signature(NodeClassifierLightningModule.__init__)
print(f"Expected parameters: {list(sig.parameters.keys())[1:]}")

try:
    lightning_module = NodeClassifierLightningModule(
        model_config=model_config,
        max_lr=base_config.training.max_lr,
        weight_decay=base_config.training.weight_decay,
        warmup_epochs=base_config.training.warmup_epochs,
        total_epochs=base_config.trainer.epochs,
        steps_per_epoch=len(data_module.train_dataloader()),
        pos_weight=base_config.training.pos_weight,
        log_predictions=False
    )
    print("✅ NodeClassifierLightningModule - OK")
except Exception as e:
    print(f"❌ NodeClassifierLightningModule - FAILED: {e}")
    sys.exit(1)

print("\n" + "="*80)
print("ALL VALIDATIONS PASSED ✅")
print("="*80)
print("\nThe sweep is ready to run!")
print("\nNext steps:")
print("1. wandb sweep sweep/config_onehot.yaml")
print("2. wandb sweep sweep/config_learned.yaml")
print("3. sbatch --array=1-10 sweep/submit.sh <sweep-id>")
