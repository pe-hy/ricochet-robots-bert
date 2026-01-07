"""Deep comparison of all features between datasets"""
import json
import numpy as np
from utils.data_module import RicochetRobotsDataset

# Load datasets
train_dataset = RicochetRobotsDataset(
    data_path='data/ricochet_data/dataset.json',
    board_size=16,
    positional_encoding='onehot',
    positional_encoding_kwargs={}
)

test_dataset = RicochetRobotsDataset(
    data_path='data/ricochet_data/test.json',
    board_size=16,
    positional_encoding='onehot',
    positional_encoding_kwargs={}
)

# Load raw JSON
with open('data/ricochet_data/dataset.json', 'r') as f:
    train_raw = json.load(f)

with open('data/ricochet_data/test.json', 'r') as f:
    test_raw = json.load(f)

print("="*80)
print("DEEP FEATURE COMPARISON")
print("="*80)

# Get first examples
train_sample = train_dataset[0]
test_sample = test_dataset[0]

train_features = train_sample['features'].numpy()
test_features = test_sample['features'].numpy()

# Get raw data
train_raw_nodes = np.array(train_raw['examples'][0]['nodes'])
test_raw_nodes = np.array(test_raw['examples'][0]['nodes'])

print(f"\nTRAINING example 0:")
print(f"  Target robot: node {np.where(train_raw_nodes[:, 3] == 1)[0][0]}")
print(f"  Helper1: node {np.where(train_raw_nodes[:, 4] == 1)[0][0]}")
print(f"  Helper2: node {np.where(train_raw_nodes[:, 5] == 1)[0][0]}")
print(f"  Helper3: node {np.where(train_raw_nodes[:, 6] == 1)[0][0]}")
print(f"  Goal: node {np.where(train_raw_nodes[:, 8] == 1)[0][0]}")

print(f"\nTEST example 0:")
print(f"  Target robot: node {np.where(test_raw_nodes[:, 3] == 1)[0][0]}")
print(f"  Helper1: node {np.where(test_raw_nodes[:, 4] == 1)[0][0]}")
print(f"  Helper2: node {np.where(test_raw_nodes[:, 5] == 1)[0][0]}")
print(f"  Helper3: node {np.where(test_raw_nodes[:, 6] == 1)[0][0]}")
print(f"  Goal: node {np.where(test_raw_nodes[:, 8] == 1)[0][0]}")

# Now let's check the processed features for the robot positions
print(f"\n" + "="*80)
print("CHECKING PROCESSED FEATURES AT ROBOT POSITIONS")
print("="*80)

def check_robot_features(features, raw_nodes, name):
    print(f"\n{name}:")
    target_idx = np.where(raw_nodes[:, 3] == 1)[0][0]
    h1_idx = np.where(raw_nodes[:, 4] == 1)[0][0]
    h2_idx = np.where(raw_nodes[:, 5] == 1)[0][0]
    h3_idx = np.where(raw_nodes[:, 6] == 1)[0][0]

    print(f"  Target robot at node {target_idx}:")
    print(f"    Raw: {raw_nodes[target_idx]}")
    print(f"    Processed (first 11): {features[target_idx, :11]}")

    print(f"  Helper1 at node {h1_idx}:")
    print(f"    Raw: {raw_nodes[h1_idx]}")
    print(f"    Processed (first 11): {features[h1_idx, :11]}")

    print(f"  Helper2 at node {h2_idx}:")
    print(f"    Raw: {raw_nodes[h2_idx]}")
    print(f"    Processed (first 11): {features[h2_idx, :11]}")

    print(f"  Helper3 at node {h3_idx}:")
    print(f"    Raw: {raw_nodes[h3_idx]}")
    print(f"    Processed (first 11): {features[h3_idx, :11]}")

check_robot_features(train_features, train_raw_nodes, "TRAINING")
check_robot_features(test_features, test_raw_nodes, "TEST")

# Check feature statistics by type
print(f"\n" + "="*80)
print("FEATURE STATISTICS BY TYPE")
print("="*80)

feature_names = [
    "robot_none", "robot_target", "robot_helper1", "robot_helper2", "robot_helper3",
    "goal_no", "goal_yes",
    "wall_none", "wall_top", "wall_left", "wall_both"
]

print(f"\n{'Feature':<20} {'Train Sum':>12} {'Test Sum':>12} {'Match'}")
print("-"*60)
for i, name in enumerate(feature_names):
    train_sum = train_features[:, i].sum()
    test_sum = test_features[:, i].sum()
    match = "✓" if train_sum == test_sum else "✗"
    print(f"{name:<20} {train_sum:>12.0f} {test_sum:>12.0f} {match}")

# Walls should be identical
print(f"\n" + "="*80)
print("WALL FEATURES (should be identical)")
print("="*80)
walls_identical = np.array_equal(train_features[:, 7:11], test_features[:, 7:11])
print(f"Wall features identical: {walls_identical}")

# Check if there's any systematic difference in how features are ordered/processed
print(f"\n" + "="*80)
print("CHECKING RAW vs PROCESSED MAPPING")
print("="*80)

def verify_processing(raw_nodes, processed_features, name):
    """Verify that processed features match raw data"""
    print(f"\n{name}:")
    errors = []

    for node_idx in range(256):
        raw = raw_nodes[node_idx]
        proc = processed_features[node_idx, :11]

        # Expected processed features:
        # 0-4: robot_type (raw[2:7])
        # 5-6: has_goal (raw[7:9])
        # 7-10: walls (raw[9:13])

        expected = np.concatenate([raw[2:7], raw[7:9], raw[9:13]]).astype(np.float32)

        if not np.array_equal(proc, expected):
            errors.append(node_idx)

    if len(errors) == 0:
        print(f"  ✓ All nodes processed correctly")
    else:
        print(f"  ❌ {len(errors)} nodes have processing errors!")
        print(f"  First 5 error nodes: {errors[:5]}")

verify_processing(train_raw_nodes, train_features, "TRAINING")
verify_processing(test_raw_nodes, test_features, "TEST")
