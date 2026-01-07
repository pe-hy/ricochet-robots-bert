"""Check if features are identical for same (x,y) position"""
import torch
import numpy as np
from utils.data_module import RicochetRobotsDataset

# Load validation split
full_dataset = RicochetRobotsDataset(
    data_path='data/ricochet_data/dataset.json',
    board_size=16,
    positional_encoding='onehot',
    positional_encoding_kwargs={}
)

total_size = len(full_dataset)
val_size = 256
test_size = 256
train_size = total_size - val_size - test_size

train_dataset, val_dataset, _ = torch.utils.data.random_split(
    full_dataset,
    [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42)
)

# Load test dataset
test_dataset = RicochetRobotsDataset(
    data_path='data/ricochet_data/test.json',
    board_size=16,
    positional_encoding='onehot',
    positional_encoding_kwargs={}
)

print("Comparing positional encodings between val and test datasets")
print("="*80)

# Get first example from each
val_sample = val_dataset[0]
test_sample = test_dataset[0]

val_features = val_sample['features'].numpy()  # [256, 43]
test_features = test_sample['features'].numpy()  # [256, 43]

print(f"\nFeature dimensions:")
print(f"  Val: {val_features.shape}")
print(f"  Test: {test_features.shape}")

# The first 11 dimensions are robot+goal+wall (can differ per example)
# The last 32 dimensions are positional encoding (should be IDENTICAL for same node index)

print(f"\nChecking positional encoding (last 32 dims) for each node...")
print(f"These should be IDENTICAL between val and test for the same node index!")

# Check a few node indices
test_nodes = [0, 10, 100, 149, 255]

for node_idx in test_nodes:
    val_pos_enc = val_features[node_idx, 11:]  # Last 32 dims
    test_pos_enc = test_features[node_idx, 11:]  # Last 32 dims

    x = node_idx % 16
    y = node_idx // 16

    identical = np.array_equal(val_pos_enc, test_pos_enc)

    print(f"\nNode {node_idx} ({x}, {y}):")
    print(f"  Identical positional encoding: {identical}")

    if not identical:
        print(f"  Val pos enc: {val_pos_enc}")
        print(f"  Test pos enc: {test_pos_enc}")
        print(f"  Difference: {np.abs(val_pos_enc - test_pos_enc).sum()}")

print(f"\n" + "="*80)
print("Checking ALL nodes...")
all_identical = np.all(val_features[:, 11:] == test_features[:, 11:])
print(f"All positional encodings identical: {all_identical}")

if not all_identical:
    print("\n❌ PROBLEM FOUND: Positional encodings differ!")
    diff_mask = ~np.all(val_features[:, 11:] == test_features[:, 11:], axis=1)
    diff_indices = np.where(diff_mask)[0]
    print(f"Nodes with different pos encodings: {diff_indices[:20].tolist()}")  # Show first 20
else:
    print("\n✓ Positional encodings are identical (as expected)")

# Now check if node 149 (ground truth in test example 0) has the right positional encoding
print(f"\n" + "="*80)
print("Detailed check of node 149 (ground truth in test example 0)")
print(f"="*80)

node_idx = 149
x_expected = 149 % 16  # Should be 5
y_expected = 149 // 16  # Should be 9

print(f"Expected coordinates: ({x_expected}, {y_expected})")
print(f"\nPositional encoding (last 32 dims):")
print(f"  First 16 values (x one-hot): {test_features[149, 11:27]}")
print(f"  Last 16 values (y one-hot): {test_features[149, 27:43]}")

# Check if one-hot is correct
x_onehot = test_features[149, 11:27]
y_onehot = test_features[149, 27:43]

x_actual = np.where(x_onehot == 1)[0]
y_actual = np.where(y_onehot == 1)[0]

print(f"\nDecoded coordinates from one-hot:")
print(f"  x: {x_actual[0] if len(x_actual) > 0 else 'NONE'}")
print(f"  y: {y_actual[0] if len(y_actual) > 0 else 'NONE'}")

if len(x_actual) > 0 and len(y_actual) > 0:
    if x_actual[0] == x_expected and y_actual[0] == y_expected:
        print(f"✓ Positional encoding is CORRECT for node 149")
    else:
        print(f"❌ Positional encoding is WRONG! Expected ({x_expected}, {y_expected}), got ({x_actual[0]}, {y_actual[0]})")
