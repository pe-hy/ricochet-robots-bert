"""Check if robot_none feature is encoded correctly in both datasets"""
import json
import numpy as np

# Load both datasets
with open('data/ricochet_data/dataset.json', 'r') as f:
    train_data = json.load(f)

with open('data/ricochet_data/test.json', 'r') as f:
    test_data = json.load(f)

print("="*80)
print("CHECKING robot_none ENCODING")
print("="*80)
print("\nCorrect encoding: robot_none=1 if NO robot at position, 0 if there IS a robot")

def check_robot_none_encoding(data, name, example_idx=0):
    print(f"\n{name} - Example {example_idx}")
    print("-"*80)

    example = data['examples'][example_idx]
    nodes = np.array(example['nodes'])

    # Find where robots are located
    robot_target_idx = np.where(nodes[:, 3] == 1)[0]
    robot_h1_idx = np.where(nodes[:, 4] == 1)[0]
    robot_h2_idx = np.where(nodes[:, 5] == 1)[0]
    robot_h3_idx = np.where(nodes[:, 6] == 1)[0]

    all_robot_indices = set()
    if len(robot_target_idx) > 0:
        all_robot_indices.add(robot_target_idx[0])
        print(f"Target robot at node {robot_target_idx[0]}")
    if len(robot_h1_idx) > 0:
        all_robot_indices.add(robot_h1_idx[0])
        print(f"Helper1 robot at node {robot_h1_idx[0]}")
    if len(robot_h2_idx) > 0:
        all_robot_indices.add(robot_h2_idx[0])
        print(f"Helper2 robot at node {robot_h2_idx[0]}")
    if len(robot_h3_idx) > 0:
        all_robot_indices.add(robot_h3_idx[0])
        print(f"Helper3 robot at node {robot_h3_idx[0]}")

    print(f"\nAll robot positions: {sorted(all_robot_indices)}")

    # Check robot_none encoding
    robot_none = nodes[:, 2]  # Index 2 is robot_none

    errors = []
    for node_idx in range(256):
        robot_none_val = robot_none[node_idx]
        has_robot = node_idx in all_robot_indices

        # Correct: robot_none=1 if no robot, 0 if robot present
        expected = 0 if has_robot else 1

        if robot_none_val != expected:
            errors.append({
                'node_idx': node_idx,
                'has_robot': has_robot,
                'robot_none_val': robot_none_val,
                'expected': expected
            })

    if len(errors) == 0:
        print(f"\n✓ robot_none encoding is CORRECT")
    else:
        print(f"\n❌ robot_none encoding has {len(errors)} ERRORS!")
        print(f"\nFirst 10 errors:")
        for err in errors[:10]:
            print(f"  Node {err['node_idx']}: has_robot={err['has_robot']}, robot_none={err['robot_none_val']}, expected={err['expected']}")

    # Also check if robot_none=0 at all robot positions
    print(f"\nChecking robot_none at robot positions:")
    for robot_idx in sorted(all_robot_indices):
        rn_val = robot_none[robot_idx]
        print(f"  Node {robot_idx}: robot_none={rn_val} (should be 0)")

    # And check some positions without robots
    non_robot_samples = [0, 1, 2, 100, 200] if 0 not in all_robot_indices else [10, 11, 12, 100, 200]
    print(f"\nChecking robot_none at NON-robot positions:")
    for idx in non_robot_samples:
        if idx not in all_robot_indices:
            rn_val = robot_none[idx]
            print(f"  Node {idx}: robot_none={rn_val} (should be 1)")

    return len(errors)

# Check both datasets
train_errors = check_robot_none_encoding(train_data, "TRAINING dataset.json", 0)
test_errors = check_robot_none_encoding(test_data, "TEST test.json", 0)

# Check a few more examples
print(f"\n" + "="*80)
print("CHECKING MULTIPLE EXAMPLES")
print("="*80)

train_error_counts = []
test_error_counts = []

for i in range(10):
    train_errors = check_robot_none_encoding(train_data, f"TRAIN {i}", i)
    train_error_counts.append(train_errors)

    test_errors = check_robot_none_encoding(test_data, f"TEST {i}", i)
    test_error_counts.append(test_errors)

print(f"\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"Training examples with robot_none errors: {sum(1 for e in train_error_counts if e > 0)}/10")
print(f"Test examples with robot_none errors: {sum(1 for e in test_error_counts if e > 0)}/10")

if sum(train_error_counts) > 0:
    print(f"\n❌ TRAINING DATA HAS robot_none BUG")
if sum(test_error_counts) > 0:
    print(f"❌ TEST DATA HAS robot_none BUG")

if sum(train_error_counts) == 0 and sum(test_error_counts) == 0:
    print(f"\n✓ Both datasets have correct robot_none encoding")
elif sum(train_error_counts) != sum(test_error_counts):
    print(f"\n❌❌❌ DATASETS HAVE DIFFERENT robot_none ENCODINGS!")
    print(f"This could explain the model failure on test.json!")
