# Ricochet Robots Node Classifier

A BERT-like Transformer model for predicting subgoals in Ricochet Robots board states using PyTorch Lightning and Weights & Biases.

## Overview

This project implements a **Transformer encoder** for binary node classification on Ricochet Robots boards. The model predicts which board positions are subgoals (intermediate waypoints) for solving puzzles.

### Key Features

- **BERT-like Architecture**: Bidirectional self-attention (not autoregressive)
- **Modular Positional Encoding**: Easy to swap between one-hot, sinusoidal, normalized, or learned encodings
- **Exact Match Metric**: Primary metric that requires ALL nodes to be classified correctly
- **PyTorch Lightning**: Clean training loop with automatic checkpointing
- **Weights & Biases Integration**: Full experiment tracking and visualization

## Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Login to Weights & Biases
wandb login
```

### 2. Verify Setup

```bash
# Run tests
python test_model.py
```

### 3. Train

```bash
python train_node_classifier.py --config config/node_classifier.yaml
```

### 4. Monitor

Training logs to Weights & Biases automatically. Monitor:
- `val/exact_match` - **Main metric** (per-example accuracy)
- `val/accuracy` - Node-level accuracy
- `val/f1`, `val/precision`, `val/recall`, `val/auroc`

## Project Structure

```
ricochet-robots-bert/
├── config/
│   └── node_classifier.yaml      # All configuration (model, training, data)
│
├── data/
│   ├── mock.py                    # Generate mock training data
│   └── ricochet_data/
│       └── dataset.json           # Training dataset (100 examples)
│
├── model/
│   ├── node_classifier.py         # Transformer architecture
│   └── lightning_module.py        # Training loop, metrics, logging
│
├── utils/
│   ├── data_module.py             # Data loading with positional encoding
│   └── positional_encoding.py    # Modular positional encoding strategies
│
├── train_node_classifier.py      # Training script
├── test_model.py                  # Verification tests
├── test_positional_encodings.py  # Test different encodings
├── visualize_node_processing.py  # Visualize data flow
└── README.md                      # This file
```

## Architecture

### Model Flow

```
Input: Board state (256 nodes × 43 features)
  ↓
Input Projection (43 → 256)
  ↓
Transformer Encoder (6 layers, 8 heads)
  - Bidirectional self-attention
  - Each node attends to ALL other nodes
  ↓
Classification Head (256 → 1)
  ↓
Output: Binary prediction for each node
```

### Feature Vector (43 dimensions)

```
Position 0-4:   Robot type (one-hot: none/target/helper1/helper2/helper3)
Position 5-6:   Has goal (one-hot: no/yes)
Position 7-10:  Walls (one-hot: none/top/left/both)
Position 11-26: X position (one-hot: 0-15)
Position 27-42: Y position (one-hot: 0-15)
```

See [NODE_FLOW_DIAGRAM.md](NODE_FLOW_DIAGRAM.md) for detailed visualization.

## Configuration

All settings are in [`config/node_classifier.yaml`](config/node_classifier.yaml).

### Key Parameters

```yaml
# Data
data:
  board_size: 16
  batch_size: 32
  positional_encoding: "onehot"  # Options: onehot, sinusoidal, normalized

# Model
model:
  d_model: 256          # Embedding dimension
  nhead: 8              # Attention heads
  num_layers: 6         # Transformer layers
  dim_feedforward: 1024
  dropout: 0.1

# Training
training:
  learning_rate: 1.0e-4
  weight_decay: 0.01
  warmup_steps: 1000

# Checkpoints monitor val/exact_match (main metric)
checkpoint:
  monitor: "val/exact_match"
```

## Positional Encoding

The model supports multiple positional encoding strategies. Change in config:

### One-Hot (Default)
```yaml
data:
  positional_encoding: "onehot"
  # Feature dim: 11 + 32 = 43
```

### Sinusoidal
```yaml
data:
  positional_encoding: "sinusoidal"
  positional_encoding_kwargs: {encoding_dim: 64}
  # Feature dim: 11 + 64 = 75
```

### Normalized
```yaml
data:
  positional_encoding: "normalized"
  # Feature dim: 11 + 2 = 13
```

Test all encodings:
```bash
python test_positional_encodings.py
```

## Metrics

### Main Metric: Exact Match
- **Per-example accuracy**: An example is correct only if ALL 256 nodes are classified correctly
- This is the primary metric for checkpointing and early stopping
- Logged as `train/exact_match`, `val/exact_match`, `test/exact_match`

### Secondary Metrics
- **Node-level accuracy**: Percentage of correctly classified nodes
- **Precision/Recall/F1**: Standard classification metrics
- **AUROC**: Area under ROC curve

## Visualization

### Trace a Single Node
```bash
python visualize_node_processing.py
```

Shows:
1. Raw data from JSON
2. Positional encoding applied
3. Feature concatenation
4. Model processing step-by-step
5. Final prediction

### Architecture Diagram
See [NODE_FLOW_DIAGRAM.md](NODE_FLOW_DIAGRAM.md) for ASCII diagrams.

## Data Format

Training data is JSON with board states:

```json
{
  "metadata": {
    "num_examples": 100,
    "board_size": 16,
    "nodes_per_example": 256,
    "features_per_node": 14
  },
  "examples": [
    {
      "example_id": 0,
      "nodes": [
        [x, y, robot(5), goal(2), walls(4), label],
        ...
      ]
    }
  ]
}
```

Generate more data:
```bash
# Edit data/mock.py to change num_examples
python data/mock.py
```

## Training Tips

1. **Class Imbalance**: Dataset is highly imbalanced (~1% positive). Model auto-computes `pos_weight`.

2. **Learning Rate**: Start with `1e-4`. Reduce to `5e-5` if unstable.

3. **Overfitting**: If validation plateaus:
   - Increase dropout (try 0.2 or 0.3)
   - Add more training data
   - Reduce model size

4. **GPU Memory**: If OOM, reduce `batch_size` or `d_model`.

5. **Exact Match**: This metric is harder than accuracy. Expected range: 0.0-0.5 early, 0.7-0.9+ when trained well.

## Model Size

Default configuration (~4.8M parameters):
- **Memory**: ~19MB (FP32), ~10MB (FP16)
- **Training speed**: ~1-2 min/epoch on GPU (100 examples)

## Example Code

```python
from model import NodeClassifierLightningModule, RicochetRobotsDataset
import torch

# Load trained model
model = NodeClassifierLightningModule.load_from_checkpoint('checkpoints/best.ckpt')
model.eval()

# Load data
dataset = RicochetRobotsDataset('data/ricochet_data/dataset.json')
sample = dataset[0]

# Predict
features = sample['features'].unsqueeze(0)  # [1, 256, 43]
with torch.no_grad():
    probs = model.model.predict_proba(features)  # [1, 256]
    predictions = (probs > 0.5).long()

print(f"Predicted {predictions.sum().item()} subgoals")
```

## Advanced Usage

### Custom Positional Encoding

Add new encoding to [`utils/positional_encoding.py`](utils/positional_encoding.py):

```python
class MyCustomEncoding(PositionalEncoding):
    def encode(self, x, y, board_size):
        # Your encoding logic
        return encoding_vector

    def get_encoding_dim(self, board_size):
        return my_dim
```

Register in `create_positional_encoding()` factory.

### Custom Model Architecture

Edit [`model/node_classifier.py`](model/node_classifier.py):
- Change `d_model`, `nhead`, `num_layers` in config
- Modify classification head
- Add custom layers

### Multi-GPU Training

```yaml
trainer:
  devices: 2  # Use 2 GPUs
  strategy: "ddp"  # Distributed data parallel
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Import errors | Run from project root |
| Data not found | Generate with `python data/mock.py` |
| CUDA OOM | Reduce `batch_size` or `d_model` |
| Wandb not initialized | Run `wandb login` |
| Low exact_match | Normal initially, should improve with training |

## References

- **Transformer Architecture**: "Attention Is All You Need" (Vaswani et al., 2017)
- **PyTorch Lightning**: https://lightning.ai/
- **Weights & Biases**: https://wandb.ai/

## License

MIT License - see LICENSE file for details.
