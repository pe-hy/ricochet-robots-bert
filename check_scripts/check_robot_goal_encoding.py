"""Check how robot/goal features are encoded"""
import json
import numpy as np
from utils.data_module import RicochetRobotsDataset

# Load raw JSON
with open('data/ricochet_data/dataset.json', 'r') as f:
    train_raw = json.load(f)

with open('data/ricochet_data/test.json', 'r') as f:
    test_raw = json.load(f)

# Load processed datasets
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

print("="*80)
print("CHECKING ROBOT/GOAL FEATURE ENCODING")
print("="*80)

def check_example(raw_data, dataset, idx, name):
    print(f"\n{name} - Example {idx}")
    print("-"*80)

    # Get raw data
    raw_example = raw_data['examples'][idx]
    raw_nodes = np.array(raw_example['nodes'])

    # Get processed data
    sample = dataset[idx]
    features = sample['features'].numpy()  # [256, 43]

    # Find robot and goal positions from raw data
    target_robot_nodes = np.where(raw_nodes[:, 3] == 1)[0]  # robot_target feature
    goal_nodes = np.where(raw_nodes[:, 8] == 1)[0]  # goal_yes feature

    if len(target_robot_nodes) > 0:
        robot_idx = target_robot_nodes[0]
        robot_x, robot_y = raw_nodes[robot_idx, 0], raw_nodes[robot_idx, 1]
        print(f"Target robot at node {robot_idx}: ({robot_x}, {robot_y})")
    else:
        print("WARNING: No target robot found!")
        robot_idx = None

    if len(goal_nodes) > 0:
        goal_idx = goal_nodes[0]
        goal_x, goal_y = raw_nodes[goal_idx, 0], raw_nodes[goal_idx, 1]
        print(f"Goal at node {goal_idx}: ({goal_x}, {goal_y})")
    else:
        print("WARNING: No goal found!")
        goal_idx = None

    # Now check the processed features
    print(f"\nChecking processed features:")

    # Find where robot_target=1 in processed features (index 1 of first 11 features)
    robot_target_feature = features[:, 1]  # Second feature (index 1)
    processed_robot_nodes = np.where(robot_target_feature == 1)[0]

    if len(processed_robot_nodes) > 0:
        proc_robot_idx = processed_robot_nodes[0]
        proc_robot_x = proc_robot_idx % 16
        proc_robot_y = proc_robot_idx // 16
        print(f"  Processed robot_target=1 at node {proc_robot_idx}: ({proc_robot_x}, {proc_robot_y})")
    else:
        print(f"  WARNING: No robot_target=1 in processed features!")

    # Find where goal_yes=1 in processed features (index 6 of first 11 features)
    goal_yes_feature = features[:, 6]  # 7th feature (index 6)
    processed_goal_nodes = np.where(goal_yes_feature == 1)[0]

    if len(processed_goal_nodes) > 0:
        proc_goal_idx = processed_goal_nodes[0]
        proc_goal_x = proc_goal_idx % 16
        proc_goal_y = proc_goal_idx // 16
        print(f"  Processed goal_yes=1 at node {proc_goal_idx}: ({proc_goal_x}, {proc_goal_y})")
    else:
        print(f"  WARNING: No goal_yes=1 in processed features!")

    # Check ground truth subgoal
    labels = sample['labels'].numpy()
    subgoal_nodes = np.where(labels == 1)[0]
    print(f"\nGround truth subgoals: {subgoal_nodes.tolist()}")

    # Verify consistency
    if robot_idx is not None and len(processed_robot_nodes) > 0:
        if robot_idx == processed_robot_nodes[0]:
            print(f"✓ Robot encoding consistent")
        else:
            print(f"❌ Robot encoding INCONSISTENT! Raw: {robot_idx}, Processed: {processed_robot_nodes[0]}")

    if goal_idx is not None and len(processed_goal_nodes) > 0:
        if goal_idx == processed_goal_nodes[0]:
            print(f"✓ Goal encoding consistent")
        else:
            print(f"❌ Goal encoding INCONSISTENT! Raw: {goal_idx}, Processed: {processed_goal_nodes[0]}")

    return raw_nodes, features

# Check first example from each dataset
train_raw_nodes, train_features = check_example(train_raw, train_dataset, 0, "TRAINING")
test_raw_nodes, test_features = check_example(test_raw, test_dataset, 0, "TEST")

# Now let's compare the actual feature format
print(f"\n" + "="*80)
print("COMPARING FEATURE STRUCTURE")
print("="*80)

print(f"\nFeature indices in processed data (first 11 features):")
print(f"  0: robot_none")
print(f"  1: robot_target")
print(f"  2: robot_helper1")
print(f"  3: robot_helper2")
print(f"  4: robot_helper3")
print(f"  5: goal_no")
print(f"  6: goal_yes")
print(f"  7: wall_none")
print(f"  8: wall_top")
print(f"  9: wall_left")
print(f"  10: wall_both")

print(f"\nFeature indices in RAW JSON data:")
print(f"  0: x")
print(f"  1: y")
print(f"  2: robot_none")
print(f"  3: robot_target")
print(f"  4: robot_helper1")
print(f"  5: robot_helper2")
print(f"  6: robot_helper3")
print(f"  7: goal_no")
print(f"  8: goal_yes")
print(f"  9: wall_none")
print(f"  10: wall_top")
print(f"  11: wall_left")
print(f"  12: wall_both")
print(f"  13: label")

# Check node 0 from both
print(f"\n" + "="*80)
print("Node 0 comparison:")
print("="*80)
print(f"Training raw: {train_raw_nodes[0]}")
print(f"Training processed (first 11): {train_features[0, :11]}")
print(f"\nTest raw: {test_raw_nodes[0]}")
print(f"Test processed (first 11): {test_features[0, :11]}")
