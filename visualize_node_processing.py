"""
Visualize what happens to a single node's input vector through the model.

This script traces a node from raw data -> feature encoding -> model processing.
"""

import torch
import numpy as np
import json
from model import RicochetRobotsDataset, NodeClassifierConfig, create_model
from utils.positional_encoding import create_positional_encoding


def print_section(title):
    """Print a section header"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def visualize_raw_data():
    """Show raw data from JSON"""
    print_section("STEP 1: RAW DATA FROM JSON")

    # Load raw JSON
    with open('data/ricochet_data/dataset.json', 'r') as f:
        data = json.load(f)

    # Get first node of first example
    first_example = data['examples'][0]
    first_node = first_example['nodes'][0]

    print(f"\nRaw node data (list of 14 values):")
    print(f"  {first_node}")

    # Parse the node
    x = first_node[0]
    y = first_node[1]
    robot_type = first_node[2:7]
    has_goal = first_node[7:9]
    walls = first_node[9:13]
    label = first_node[13]

    print(f"\nParsed components:")
    print(f"  Position: x={x}, y={y}")
    print(f"  Robot type (5-dim one-hot): {robot_type}")
    print(f"    → {['none', 'target', 'helper1', 'helper2', 'helper3'][np.argmax(robot_type)]}")
    print(f"  Has goal (2-dim one-hot): {has_goal}")
    print(f"    → {'yes' if has_goal[1] == 1 else 'no'}")
    print(f"  Walls (4-dim one-hot): {walls}")
    print(f"    → {['none', 'top', 'left', 'both'][np.argmax(walls)]}")
    print(f"  Label (subgoal): {label}")

    return x, y, robot_type, has_goal, walls, label


def visualize_positional_encoding(x, y, board_size=16):
    """Show different positional encoding strategies"""
    print_section("STEP 2: POSITIONAL ENCODING")

    print(f"\nInput coordinates: x={x}, y={y} (on {board_size}x{board_size} board)")
    print()

    # One-hot encoding
    print("=" * 60)
    print("Option 1: One-Hot Encoding (DEFAULT)")
    print("=" * 60)
    encoder = create_positional_encoding('onehot')
    pos_enc = encoder.encode(x, y, board_size)
    print(f"  Encoding dimension: {encoder.get_encoding_dim(board_size)}")
    print(f"  Output shape: {pos_enc.shape}")
    print(f"\n  X one-hot (16 dims): {pos_enc[:16]}")
    print(f"    → Position {x} is 1.0, rest are 0.0")
    print(f"\n  Y one-hot (16 dims): {pos_enc[16:]}")
    print(f"    → Position {y} is 1.0, rest are 0.0")
    print(f"\n  Total: {len(pos_enc)} dimensions")

    # Sinusoidal encoding
    print("\n" + "=" * 60)
    print("Option 2: Sinusoidal Encoding")
    print("=" * 60)
    encoder = create_positional_encoding('sinusoidal', encoding_dim=64)
    pos_enc = encoder.encode(x, y, board_size)
    print(f"  Encoding dimension: {encoder.get_encoding_dim(board_size)}")
    print(f"  Output shape: {pos_enc.shape}")
    print(f"  First 8 values: {pos_enc[:8]}")
    print(f"  → Smooth continuous values based on sin/cos functions")
    print(f"  Total: {len(pos_enc)} dimensions")

    # Normalized encoding
    print("\n" + "=" * 60)
    print("Option 3: Normalized Coordinates")
    print("=" * 60)
    encoder = create_positional_encoding('normalized')
    pos_enc = encoder.encode(x, y, board_size)
    print(f"  Encoding dimension: {encoder.get_encoding_dim(board_size)}")
    print(f"  Output shape: {pos_enc.shape}")
    print(f"  Values: {pos_enc}")
    print(f"  → x_norm = {x}/(16-1) = {pos_enc[0]:.4f}")
    print(f"  → y_norm = {y}/(16-1) = {pos_enc[1]:.4f}")
    print(f"  Total: {len(pos_enc)} dimensions")

    # Return one-hot for next steps
    encoder = create_positional_encoding('onehot')
    return encoder.encode(x, y, board_size)


def visualize_feature_concatenation(robot_type, has_goal, walls, pos_encoding):
    """Show how features are concatenated"""
    print_section("STEP 3: FEATURE CONCATENATION")

    print("\nCombining all features into a single vector:")
    print()

    # Convert to numpy arrays
    robot_type = np.array(robot_type)
    has_goal = np.array(has_goal)
    walls = np.array(walls)

    # Show dimensions
    print(f"  Robot type:     {robot_type.shape} → {robot_type}")
    print(f"  Has goal:       {has_goal.shape} → {has_goal}")
    print(f"  Walls:          {walls.shape} → {walls}")
    print(f"  Pos encoding:   {pos_encoding.shape} → {pos_encoding[:5]}... (showing first 5)")
    print()

    # Concatenate
    features = np.concatenate([robot_type, has_goal, walls, pos_encoding])

    print(f"  ↓ CONCATENATE ↓")
    print()
    print(f"  Final feature vector: {features.shape}")
    print(f"  → Total dimensions: {len(features)}")
    print()
    print(f"  Structure:")
    print(f"    [0:5]    = robot_type  = {features[0:5]}")
    print(f"    [5:7]    = has_goal    = {features[5:7]}")
    print(f"    [7:11]   = walls       = {features[7:11]}")
    print(f"    [11:27]  = x_onehot    = {features[11:27]}")
    print(f"    [27:43]  = y_onehot    = {features[27:43]}")

    return features


def visualize_model_processing(features):
    """Show what happens inside the model"""
    print_section("STEP 4: MODEL PROCESSING")

    # Create model
    config = NodeClassifierConfig(
        feature_dim=43,
        d_model=256,
        nhead=8,
        num_layers=6,
        dim_feedforward=1024,
        dropout=0.1
    )
    model = create_model(config)
    model.eval()

    print(f"\nModel architecture:")
    print(f"  Input dimension:  {config.feature_dim}")
    print(f"  Embedding dimension (d_model): {config.d_model}")
    print(f"  Transformer layers: {config.num_layers}")
    print(f"  Attention heads: {config.nhead}")
    print()

    # Convert to tensor and add batch dimension
    x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    # Shape: [batch=1, num_nodes=1, feature_dim=43]

    print(f"Input tensor shape: {x.shape}")
    print(f"  → [batch_size=1, num_nodes=1, feature_dim={config.feature_dim}]")
    print()

    # Trace through model
    with torch.no_grad():
        # Step 1: Input projection
        print("─" * 60)
        print("Layer 1: Input Projection (Linear)")
        print("─" * 60)
        projected = model.input_projection(x)
        print(f"  Input:  {x.shape} → values in range [{x.min():.2f}, {x.max():.2f}]")
        print(f"  Output: {projected.shape} → values in range [{projected.min():.2f}, {projected.max():.2f}]")
        print(f"  → Projects 43-dim features to {config.d_model}-dim embeddings")
        print(f"  → This is where positional encoding is 'mixed' with other features")
        print()

        # Step 2: Transformer encoder
        print("─" * 60)
        print("Layer 2: Transformer Encoder (6 layers)")
        print("─" * 60)
        encoded = model.transformer_encoder(projected)
        print(f"  Input:  {projected.shape}")
        print(f"  Output: {encoded.shape}")
        print(f"  → Each layer applies:")
        print(f"    1. Multi-head self-attention (8 heads)")
        print(f"       - Node can attend to ALL other nodes (bidirectional)")
        print(f"    2. Feedforward network (256 → 1024 → 256)")
        print(f"    3. Layer normalization and residual connections")
        print(f"  → Output: contextualized representation of each node")
        print()

        # Step 3: Classification head
        print("─" * 60)
        print("Layer 3: Classification Head")
        print("─" * 60)
        logits = model.classifier(encoded)
        probs = torch.sigmoid(logits)
        print(f"  Input:  {encoded.shape}")
        print(f"  After Linear(256 → 128): [1, 1, 128]")
        print(f"  After GELU activation: [1, 1, 128]")
        print(f"  After Linear(128 → 1): [1, 1, 1]")
        print(f"  Logit: {logits.item():.4f}")
        print(f"  After Sigmoid: {probs.item():.4f}")
        print(f"  → Final prediction: {probs.item():.4f} (probability of being a subgoal)")
        print()

    return probs.item()


def visualize_complete_example():
    """Show complete example with real data"""
    print_section("COMPLETE EXAMPLE: Processing Multiple Nodes")

    # Load real data
    dataset = RicochetRobotsDataset(
        data_path='data/ricochet_data/dataset.json',
        board_size=16,
        positional_encoding='onehot'
    )

    sample = dataset[0]
    features = sample['features']  # [256, 43]
    labels = sample['labels']      # [256]

    print(f"\nComplete board state:")
    print(f"  Number of nodes: {len(features)} (16x16 grid)")
    print(f"  Features per node: {features.shape[1]}")
    print(f"  Number of subgoals: {labels.sum().item()}")
    print()

    # Show a few nodes
    print("Example nodes:")
    for i in range(3):
        x = i % 16
        y = i // 16
        print(f"\n  Node {i} (position {x},{y}):")
        print(f"    Feature vector: {features[i].shape}")
        print(f"    First 11 dims (base features): {features[i][:11].numpy()}")
        print(f"    Remaining 32 dims: positional encoding")
        print(f"    Label (subgoal): {labels[i].item()}")

    # Process through model
    print("\n" + "─" * 60)
    print("Processing all 256 nodes through model...")
    print("─" * 60)

    config = NodeClassifierConfig(feature_dim=43, d_model=256, nhead=8, num_layers=6)
    model = create_model(config)
    model.eval()

    with torch.no_grad():
        batch = features.unsqueeze(0)  # [1, 256, 43]
        probs = model.predict_proba(batch)  # [1, 256]
        preds = (probs > 0.5).long()

    print(f"\nModel output:")
    print(f"  Predictions shape: {probs.shape}")
    print(f"  Predicted subgoals: {preds.sum().item()}")
    print(f"  Actual subgoals: {labels.sum().item()}")
    print(f"  Probability range: [{probs.min():.4f}, {probs.max():.4f}]")
    print()

    # Show predictions for nodes that are actually subgoals
    subgoal_indices = (labels == 1).nonzero(as_tuple=True)[0]
    print(f"Predictions for actual subgoal nodes:")
    for idx in subgoal_indices:
        x = idx.item() % 16
        y = idx.item() // 16
        prob = probs[0, idx].item()
        pred = preds[0, idx].item()
        print(f"  Node {idx.item()} at ({x},{y}): prob={prob:.4f}, pred={pred}, label=1")


def main():
    print("\n" + "=" * 80)
    print("  VISUALIZING NODE PROCESSING PIPELINE")
    print("  From Raw Data → Feature Vector → Model → Prediction")
    print("=" * 80)

    # Step 1: Raw data
    x, y, robot_type, has_goal, walls, label = visualize_raw_data()

    # Step 2: Positional encoding
    pos_encoding = visualize_positional_encoding(x, y)

    # Step 3: Feature concatenation
    features = visualize_feature_concatenation(robot_type, has_goal, walls, pos_encoding)

    # Step 4: Model processing
    prob = visualize_model_processing(features)

    print_section("SUMMARY")
    print(f"""
Input node at position ({x}, {y}):
  • 43-dimensional feature vector
    - 11 dims: base features (robot, goal, walls)
    - 32 dims: positional encoding (one-hot x,y)

  • Input projection: 43 → 256 dimensions
    - Linear transformation combines all features
    - Positional info is "mixed in" here

  • Transformer encoder: 256 → 256 dimensions
    - 6 layers of bidirectional self-attention
    - Each node attends to ALL other nodes
    - Learns spatial relationships and patterns

  • Classification head: 256 → 1 dimension
    - Final binary prediction
    - Sigmoid activation → probability

  • Final prediction: {prob:.4f}
    - Threshold at 0.5 for binary classification
    - {'SUBGOAL' if prob > 0.5 else 'NOT SUBGOAL'} (actual label: {label})
""")

    # Complete example
    visualize_complete_example()

    print("\n" + "=" * 80)
    print("  KEY TAKEAWAYS")
    print("=" * 80)
    print("""
1. Each node is a 43-dim vector (11 base + 32 positional)
2. Input projection mixes all features together into 256 dims
3. Transformer encoder processes ALL nodes simultaneously
4. Self-attention allows nodes to "see" the entire board
5. Classification head outputs probability for each node independently
6. All 256 nodes are processed in parallel through the same model

This is different from sequence models:
  ✗ Not autoregressive (doesn't predict one token at a time)
  ✓ All nodes processed in parallel
  ✓ Bidirectional attention (BERT-like, not GPT-like)
  ✓ Each node's prediction is independent given the context
""")


if __name__ == '__main__':
    main()
