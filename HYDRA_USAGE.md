# Hydra Configuration Guide

The training script now uses **Hydra** for configuration management, providing powerful CLI overrides and config composition.

## Basic Usage

### 1. Train with Default Config

```bash
python train_node_classifier.py
```

This loads `config/node_classifier.yaml` and trains with default settings.

### 2. Override Parameters from CLI

```bash
# Change epochs and learning rate
python train_node_classifier.py trainer.epochs=50 training.max_lr=2e-3

# Change data split
python train_node_classifier.py data.val_size=32 data.test_size=32

# Change model architecture
python train_node_classifier.py model.d_model=512 model.num_layers=12

# Multiple overrides
python train_node_classifier.py \
    trainer.epochs=200 \
    training.max_lr=1e-3 \
    training.warmup_epochs=10 \
    data.batch_size=64
```

### 3. View Help and Current Config

```bash
# Show help and current configuration
python train_node_classifier.py --help

# Show config without training
python train_node_classifier.py --cfg job
```

## Configuration Structure

The config is organized into logical groups:

```yaml
# config/node_classifier.yaml

seed: 42                    # Random seed

wandb:                      # Weights & Biases
  project: ...
  name: ...

data:                       # Data loading
  train_path: ...
  batch_size: ...
  val_size: ...
  test_size: ...

model:                      # Model architecture
  d_model: ...
  nhead: ...
  num_layers: ...

training:                   # Training parameters
  max_lr: ...
  warmup_epochs: ...

trainer:                    # PyTorch Lightning
  epochs: ...
  accelerator: ...

checkpoint:                 # Checkpointing
  monitor: ...
  save_top_k: ...

early_stopping:             # Early stopping
  enabled: ...
  patience: ...
```

## Common Use Cases

### Change Model Size

```bash
# Smaller model (faster training)
python train_node_classifier.py \
    model.d_model=128 \
    model.num_layers=4 \
    model.nhead=4

# Larger model (more capacity)
python train_node_classifier.py \
    model.d_model=512 \
    model.num_layers=12 \
    model.nhead=16
```

### Adjust Learning Rate Schedule

```bash
# Higher learning rate, longer warmup
python train_node_classifier.py \
    training.max_lr=5e-3 \
    training.warmup_epochs=10

# Lower learning rate, shorter warmup
python train_node_classifier.py \
    training.max_lr=5e-4 \
    training.warmup_epochs=2
```

### Change Data Split

```bash
# More validation data
python train_node_classifier.py data.val_size=32

# No test set (all for train/val)
python train_node_classifier.py data.test_size=0

# Balanced split
python train_node_classifier.py data.val_size=30 data.test_size=30
```

### Change Positional Encoding

```bash
# Use sinusoidal encoding
python train_node_classifier.py \
    data.positional_encoding=sinusoidal \
    'data.positional_encoding_kwargs={encoding_dim: 64}'

# Use normalized coordinates
python train_node_classifier.py data.positional_encoding=normalized
```

### Training Duration

```bash
# Quick training (testing)
python train_node_classifier.py trainer.epochs=10

# Long training
python train_node_classifier.py trainer.epochs=500

# Disable early stopping
python train_node_classifier.py early_stopping.enabled=false
```

### GPU Configuration

```bash
# Specific GPU
python train_node_classifier.py trainer.devices=1

# Multiple GPUs
python train_node_classifier.py trainer.devices=2 trainer.strategy=ddp

# Mixed precision training
python train_node_classifier.py trainer.precision=16-mixed
```

## Advanced Features

### 1. Config Composition

Create multiple config files and compose them:

```yaml
# config/model/small.yaml
d_model: 128
nhead: 4
num_layers: 4

# config/model/large.yaml
d_model: 512
nhead: 16
num_layers: 12
```

Then use:
```bash
python train_node_classifier.py model=small
python train_node_classifier.py model=large
```

### 2. Hyperparameter Sweeps

Run multiple experiments with different parameters:

```bash
python train_node_classifier.py --multirun \
    training.max_lr=1e-4,5e-4,1e-3 \
    model.dropout=0.1,0.2,0.3
```

This runs 9 experiments (3 LR × 3 dropout values).

### 3. Output Directory

Hydra automatically creates timestamped output directories:

```
outputs/
  2024-10-29/
    14-30-15/        # Timestamp
      .hydra/        # Config and logs
      checkpoints/   # Model checkpoints
```

Override output directory:
```bash
python train_node_classifier.py hydra.run.dir=my_experiment
```

### 4. Config Validation

Hydra validates types and catches errors:

```bash
# This will error (epochs must be int)
python train_node_classifier.py trainer.epochs=foo

# This will error (unknown config key)
python train_node_classifier.py trainer.unknown_param=123
```

## Tips and Tricks

### 1. Quick Debugging

```bash
# Small model, few epochs, no wandb
python train_node_classifier.py \
    trainer.epochs=5 \
    model.d_model=64 \
    model.num_layers=2 \
    wandb.log_model=false
```

### 2. Resume from Checkpoint

```bash
# Add checkpoint path to config or CLI
python train_node_classifier.py \
    checkpoint.resume_from=checkpoints/last.ckpt
```

### 3. Experiment Tracking

Use meaningful wandb names:
```bash
python train_node_classifier.py \
    wandb.name=exp_001_baseline
```

### 4. Disable Wandb

```bash
# Set WANDB_MODE=disabled
WANDB_MODE=disabled python train_node_classifier.py
```

## Hydra vs. Argparse

**Advantages of Hydra:**

✅ **Hierarchical config**: Nested structure (e.g., `trainer.epochs`)
✅ **Type validation**: Catches errors early
✅ **Config composition**: Mix and match configs
✅ **Multirun sweeps**: Easy hyperparameter search
✅ **Output management**: Automatic timestamped directories
✅ **No boilerplate**: No need to write argparse code

**Migration from old script:**

| Old (argparse) | New (Hydra) |
|----------------|-------------|
| `--max_epochs 50` | `trainer.epochs=50` |
| `--learning_rate 2e-3` | `training.max_lr=2e-3` |
| `--batch_size 64` | `data.batch_size=64` |
| `--config my_config.yaml` | `--config-name=my_config` |

## Reference

- **Hydra Documentation**: https://hydra.cc/
- **OmegaConf**: https://omegaconf.readthedocs.io/
- **PyTorch Lightning + Hydra**: https://lightning.ai/docs/pytorch/stable/cli/lightning_cli.html

## Example Commands

```bash
# Baseline experiment
python train_node_classifier.py

# Quick test run
python train_node_classifier.py trainer.epochs=5 data.batch_size=8

# Large model, long training
python train_node_classifier.py \
    model.d_model=512 \
    model.num_layers=12 \
    trainer.epochs=500

# Sweep over learning rates
python train_node_classifier.py --multirun \
    training.max_lr=1e-4,5e-4,1e-3,5e-3

# Different positional encoding
python train_node_classifier.py \
    data.positional_encoding=sinusoidal \
    'data.positional_encoding_kwargs={encoding_dim: 64}'

# Production run with good settings
python train_node_classifier.py \
    trainer.epochs=200 \
    training.max_lr=1e-3 \
    training.warmup_epochs=10 \
    data.batch_size=64 \
    early_stopping.patience=20 \
    wandb.name=production_run_v1
```
