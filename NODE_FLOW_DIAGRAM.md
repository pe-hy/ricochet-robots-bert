# Node Processing Flow

## Single Node Transformation

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ RAW DATA (from JSON)                                                        │
│ Node at position (x=0, y=0)                                                 │
│ [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]                                  │
│  │  │  └──────┬──────┘  └──┬──┘  └───┬───┘ │                                │
│  x  y   robot(5)     goal(2)    walls(4)  label                            │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ POSITIONAL ENCODING (configurable)                                         │
│                                                                             │
│ Input: x=0, y=0                                                             │
│                                                                             │
│ One-hot (default):                                                          │
│   x_onehot = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  (16 dims)   │
│   y_onehot = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  (16 dims)   │
│   → 32 dimensions total                                                     │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ FEATURE CONCATENATION                                                       │
│                                                                             │
│ [robot(5) | goal(2) | walls(4) | x_onehot(16) | y_onehot(16)]             │
│ [1,0,0,0,0 | 1,0 | 0,0,0,1 | 1,0,0,...,0 | 1,0,0,...,0]                    │
│                                                                             │
│ → 43 dimensions (11 base + 32 positional)                                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ MODEL: INPUT PROJECTION (Linear Layer)                                     │
│                                                                             │
│ W @ [43-dim vector] + b → [256-dim embedding]                              │
│                                                                             │
│ This is where positional encoding gets "mixed" with other features!        │
│ The linear layer learns to combine:                                        │
│   - Robot type features                                                     │
│   - Goal features                                                           │
│   - Wall features                                                           │
│   - Position features (x, y)                                                │
│                                                                             │
│ → 256 dimensions                                                            │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ MODEL: TRANSFORMER ENCODER (6 layers)                                      │
│                                                                             │
│ For each of 6 layers:                                                       │
│                                                                             │
│   ┌──────────────────────────────────────────────────────────┐            │
│   │ 1. Multi-Head Self-Attention (8 heads)                   │            │
│   │    - Each node attends to ALL 256 nodes (bidirectional)  │            │
│   │    - Learns spatial relationships                        │            │
│   │    - No causal masking (BERT-like)                       │            │
│   └──────────────────────────────────────────────────────────┘            │
│                          │                                                  │
│                          ▼                                                  │
│   ┌──────────────────────────────────────────────────────────┐            │
│   │ 2. Feedforward Network                                   │            │
│   │    256 → 1024 → 256                                      │            │
│   │    GELU activation                                       │            │
│   └──────────────────────────────────────────────────────────┘            │
│                          │                                                  │
│                          ▼                                                  │
│   ┌──────────────────────────────────────────────────────────┐            │
│   │ 3. Layer Norm + Residual Connection                      │            │
│   └──────────────────────────────────────────────────────────┘            │
│                                                                             │
│ → 256 dimensions (contextualized embedding)                                │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ MODEL: CLASSIFICATION HEAD                                                 │
│                                                                             │
│ Linear(256 → 128) → GELU → Dropout → Linear(128 → 1)                      │
│                                                                             │
│ Logit: 1.2612                                                               │
│                                                                             │
│ → 1 dimension (logit)                                                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ SIGMOID ACTIVATION                                                          │
│                                                                             │
│ sigmoid(1.2612) = 0.7792                                                    │
│                                                                             │
│ → Probability of being a subgoal: 77.92%                                   │
│ → Binary prediction (threshold 0.5): SUBGOAL ✓                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Key Points

### 1. Feature Vector Composition (43 dims)
```
Position 0-4:   Robot type (one-hot: none/target/helper1/helper2/helper3)
Position 5-6:   Has goal (one-hot: no/yes)
Position 7-10:  Walls (one-hot: none/top/left/both)
Position 11-26: X position (one-hot: 0-15)
Position 27-42: Y position (one-hot: 0-15)
```

### 2. Where Positional Encoding is Combined
The **Input Projection** layer is where the magic happens:
```python
# Input: [43 dims]  = [base_features(11) | x_onehot(16) | y_onehot(16)]
# Weight matrix: [256, 43]
# Output: [256 dims]

output = W @ input + b
```

This linear transformation learns to:
- Combine positional information with semantic features
- Create a rich 256-dim embedding
- The network learns which positions correlate with which features

### 3. Parallel Processing
```
Input:  [batch, 256 nodes, 43 features]
          ↓
Model:  All 256 nodes processed in parallel
          ↓
Output: [batch, 256 nodes, 1 probability]
```

**Not sequential!** All nodes are processed simultaneously through the same network.

### 4. Self-Attention Mechanism
```
Node at (5, 10) can attend to:
  - Node at (4, 10)  ← left neighbor
  - Node at (6, 10)  → right neighbor
  - Node at (5, 9)   ↑ top neighbor
  - Node at (5, 11)  ↓ bottom neighbor
  - Node at (15, 0)  ↗ far corner
  - ANY other node on the board!
```

The attention weights are learned and determine:
- Which nodes are relevant for predicting subgoals
- Spatial patterns (e.g., "subgoals near goals")
- Long-range dependencies

## Comparison: Different Positional Encodings

| Encoding     | Dims | Feature Vector Size | Description |
|--------------|------|---------------------|-------------|
| `onehot`     | 32   | 43 dims             | Sparse, explicit position |
| `sinusoidal` | 64   | 75 dims             | Dense, continuous, fixed |
| `normalized` | 2    | 13 dims             | Minimal, just (x_norm, y_norm) |
| `learned`    | 2    | 13 dims             | Placeholder for embedding table |

### Example: Changing to Sinusoidal

```yaml
# config/node_classifier.yaml
data:
  positional_encoding: "sinusoidal"
  positional_encoding_kwargs: {encoding_dim: 64}
```

**Effect:**
- Feature vector: 11 + 64 = **75 dimensions**
- Input projection: 75 → 256
- Rest of model stays the same

## All 256 Nodes Together

```
Board (16×16 = 256 nodes)

     0   1   2   3  ...  15
   ┌───┬───┬───┬───┬───┬───┐
 0 │ N │ N │ N │ N │...│ N │  ← Each node: [batch, 43] features
   ├───┼───┼───┼───┼───┼───┤
 1 │ N │ N │ N │ N │...│ N │
   ├───┼───┼───┼───┼───┼───┤
 2 │ N │ N │ S │ N │...│ N │  S = Subgoal (label=1)
   ├───┼───┼───┼───┼───┼───┤
 . │ . │ . │ . │ . │...│ . │
   .   .   .   .   .   .   .

Input to model:   [batch=1, 256 nodes, 43 features]
Output from model: [batch=1, 256 nodes, 1 probability]
```

Each node gets its own prediction, but predictions are **contextualized** by all other nodes through self-attention.

## Example: What the Model Learns

After training, the model might learn patterns like:
- "Nodes near the goal are more likely to be subgoals"
- "Empty cells (no robot, no wall) are more likely to be subgoals"
- "Subgoals tend to form a path from robot to goal"
- "Cells with walls blocking certain directions are less likely"

These patterns emerge from the training data through the self-attention mechanism!
