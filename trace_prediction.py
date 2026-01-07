"""Trace a single prediction to understand the discrepancy"""
import torch
import json
import numpy as np
from model.lightning_module import NodeClassifierLightningModule
from utils.data_module import RicochetRobotsDataset

# Load the model
checkpoint_path = "./tmp/checkpoints/d256_l24_h16_n50000_lr0.001_bs256/epoch=45-0.8828.ckpt"
print(f"Loading checkpoint: {checkpoint_path}")
model = NodeClassifierLightningModule.load_from_checkpoint(checkpoint_path)
model.eval()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

# Load validation split from dataset.json
print("\nLoading training dataset...")
full_dataset = RicochetRobotsDataset(
    data_path='data/ricochet_data/dataset.json',
    board_size=16,
    positional_encoding='onehot',
    positional_encoding_kwargs={}
)

# Split to get validation set (same way as training)
total_size = len(full_dataset)
val_size = 256
test_size = 256
train_size = total_size - val_size - test_size

train_dataset, val_dataset, test_split_dataset = torch.utils.data.random_split(
    full_dataset,
    [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42)
)

# Load test.json
print("Loading test.json...")
test_dataset = RicochetRobotsDataset(
    data_path='data/ricochet_data/test.json',
    board_size=16,
    positional_encoding='onehot',
    positional_encoding_kwargs={}
)

print(f"\nDataset sizes:")
print(f"  Validation split: {len(val_dataset)}")
print(f"  Test.json: {len(test_dataset)}")

def analyze_prediction(dataset, idx, name):
    """Analyze a single prediction"""
    print(f"\n{'='*80}")
    print(f"{name} - Example {idx}")
    print(f"{'='*80}")

    sample = dataset[idx]
    features = sample['features']  # [256, 43]
    labels = sample['labels']      # [256]

    print(f"Feature shape: {features.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Labels sum (number of positives): {labels.sum()}")

    # Get prediction
    with torch.no_grad():
        features_batch = features.unsqueeze(0).to(device)
        logits = model(features_batch)
        probs = torch.sigmoid(logits.squeeze(-1)).squeeze(0).cpu().numpy()
        preds = (probs > 0.5).astype(int)

    print(f"Predictions sum: {preds.sum()}")
    print(f"Max probability: {probs.max():.4f} at node {probs.argmax()}")

    # Check exact match
    exact_match = np.array_equal(preds, labels.numpy())
    print(f"Exact match: {exact_match}")

    # Show ground truth subgoals
    gt_indices = np.where(labels.numpy() == 1)[0]
    print(f"\nGround truth subgoals (indices): {gt_indices.tolist()}")
    for gt_idx in gt_indices:
        x = gt_idx % 16
        y = gt_idx // 16
        print(f"  Node {gt_idx} ({x}, {y}): prob={probs[gt_idx]:.6f}")

    # Show predicted subgoals
    pred_indices = np.where(preds == 1)[0]
    print(f"\nPredicted subgoals (indices): {pred_indices.tolist()}")
    for pred_idx in pred_indices[:5]:  # Show first 5
        x = pred_idx % 16
        y = pred_idx // 16
        correct = "✓" if pred_idx in gt_indices else "✗"
        print(f"  Node {pred_idx} ({x}, {y}): prob={probs[pred_idx]:.6f} {correct}")

    # Check feature statistics
    print(f"\nFeature statistics:")
    print(f"  Mean: {features.mean().item():.6f}")
    print(f"  Std: {features.std().item():.6f}")
    print(f"  Min: {features.min().item():.6f}")
    print(f"  Max: {features.max().item():.6f}")

    # Check first few features (robot, goal, wall encoding)
    print(f"\nFirst 11 features (robot+goal+wall) at node 0:")
    print(f"  {features[0, :11].numpy()}")

    return exact_match, features, probs

# Test on validation split
print("\n" + "#"*80)
print("VALIDATION SPLIT FROM dataset.json")
print("#"*80)
val_matches = []
for i in range(3):
    match, _, _ = analyze_prediction(val_dataset, i, "VALIDATION")
    val_matches.append(match)

# Test on test.json
print("\n" + "#"*80)
print("TEST.JSON")
print("#"*80)
test_matches = []
for i in range(3):
    match, _, _ = analyze_prediction(test_dataset, i, "TEST")
    test_matches.append(match)

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"Validation exact match rate: {np.mean(val_matches):.2%}")
print(f"Test exact match rate: {np.mean(test_matches):.2%}")
