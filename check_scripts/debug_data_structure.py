"""Compare data structure between dataset.json and test.json"""
import json
import numpy as np

print("="*80)
print("COMPARING DATA STRUCTURE")
print("="*80)

# Load both files
with open('data/ricochet_data/dataset.json', 'r') as f:
    train_data = json.load(f)

with open('data/ricochet_data/test.json', 'r') as f:
    test_data = json.load(f)

print("\n1. FILE STRUCTURE")
print("-"*80)
print(f"dataset.json keys: {train_data.keys()}")
print(f"test.json keys: {test_data.keys()}")
print(f"\ndataset.json metadata: {train_data['metadata']}")
print(f"test.json metadata: {test_data['metadata']}")

print("\n2. FIRST EXAMPLE STRUCTURE")
print("-"*80)
train_ex = train_data['examples'][0]
test_ex = test_data['examples'][0]

print(f"dataset.json example keys: {train_ex.keys()}")
print(f"test.json example keys: {test_ex.keys()}")

print(f"\ndataset.json nodes count: {len(train_ex['nodes'])}")
print(f"test.json nodes count: {len(test_ex['nodes'])}")

print("\n3. FIRST NODE STRUCTURE")
print("-"*80)
train_node = train_ex['nodes'][0]
test_node = test_ex['nodes'][0]

print(f"dataset.json first node: {train_node}")
print(f"test.json first node: {test_node}")
print(f"\ndataset.json node length: {len(train_node)}")
print(f"test.json node length: {len(test_node)}")

print("\n4. LABEL DISTRIBUTION IN FIRST EXAMPLE")
print("-"*80)
train_labels = [node[13] for node in train_ex['nodes']]
test_labels = [node[13] for node in test_ex['nodes']]

print(f"dataset.json labels: sum={sum(train_labels)}, indices with 1: {[i for i, l in enumerate(train_labels) if l == 1]}")
print(f"test.json labels: sum={sum(test_labels)}, indices with 1: {[i for i, l in enumerate(test_labels) if l == 1]}")

print("\n5. ROBOT AND GOAL POSITIONS")
print("-"*80)
# Find positions where robot features are 1
train_nodes_np = np.array(train_ex['nodes'])
test_nodes_np = np.array(test_ex['nodes'])

# Features are: [x, y, robot_none, robot_target, robot_helper1, robot_helper2, robot_helper3, goal_no, goal_yes, wall_none, wall_top, wall_left, wall_both, label]
# Indices: 0:x, 1:y, 2:robot_none, 3:robot_target, 4-6:helpers, 7:goal_no, 8:goal_yes, 9-12:walls, 13:label

print("dataset.json:")
robot_target_idx = np.where(train_nodes_np[:, 3] == 1)[0]
goal_idx = np.where(train_nodes_np[:, 8] == 1)[0]
print(f"  Target robot at node {robot_target_idx[0] if len(robot_target_idx) > 0 else 'NOT FOUND'}: coords ({train_nodes_np[robot_target_idx[0], 0]}, {train_nodes_np[robot_target_idx[0], 1]})")
print(f"  Goal at node {goal_idx[0] if len(goal_idx) > 0 else 'NOT FOUND'}: coords ({train_nodes_np[goal_idx[0], 0]}, {train_nodes_np[goal_idx[0], 1]})")

print("\ntest.json:")
robot_target_idx = np.where(test_nodes_np[:, 3] == 1)[0]
goal_idx = np.where(test_nodes_np[:, 8] == 1)[0]
print(f"  Target robot at node {robot_target_idx[0] if len(robot_target_idx) > 0 else 'NOT FOUND'}: coords ({test_nodes_np[robot_target_idx[0], 0]}, {test_nodes_np[robot_target_idx[0], 1]})")
print(f"  Goal at node {goal_idx[0] if len(goal_idx) > 0 else 'NOT FOUND'}: coords ({test_nodes_np[goal_idx[0], 0]}, {test_nodes_np[goal_idx[0], 1]})")

print("\n6. CHECK FOR DATA TYPE DIFFERENCES")
print("-"*80)
print(f"dataset.json node[0] types: {[type(x).__name__ for x in train_node]}")
print(f"test.json node[0] types: {[type(x).__name__ for x in test_node]}")

print("\n7. WALL ENCODING CHECK")
print("-"*80)
# Check if walls are encoded differently
train_walls = train_nodes_np[:, 9:13]
test_walls = test_nodes_np[:, 9:13]
print(f"dataset.json wall features sum per type: {train_walls.sum(axis=0)}")
print(f"test.json wall features sum per type: {test_walls.sum(axis=0)}")

print("\n" + "="*80)
