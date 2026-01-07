"""Confirm that walls are in different positions"""
import json
import numpy as np

# Load raw JSON
with open('data/ricochet_data/dataset.json', 'r') as f:
    train_data = json.load(f)

with open('data/ricochet_data/test.json', 'r') as f:
    test_data = json.load(f)

# Check walls from first example
train_walls = np.array(train_data['examples'][0]['nodes'])[:, 9:13]
test_walls = np.array(test_data['examples'][0]['nodes'])[:, 9:13]

print("="*80)
print("WALL CONFIGURATION COMPARISON")
print("="*80)

walls_identical = np.array_equal(train_walls, test_walls)
print(f"\nWall configurations identical: {walls_identical}")

if not walls_identical:
    print("\n❌❌❌ THE BOARDS HAVE DIFFERENT WALL CONFIGURATIONS! ❌❌❌")
    print("\nThis is why the model fails on test.json!")
    print("The model was trained on ONE board layout (dataset.json)")
    print("But test.json uses a DIFFERENT board layout!")

    # Find differences
    diff_mask = ~np.all(train_walls == test_walls, axis=1)
    diff_indices = np.where(diff_mask)[0]

    print(f"\n{len(diff_indices)} out of 256 nodes have different wall configurations")
    print(f"That's {len(diff_indices)/256*100:.1f}% of the board!")

    print(f"\nFirst 10 nodes with different walls:")
    for idx in diff_indices[:10]:
        x = idx % 16
        y = idx // 16
        print(f"  Node {idx} ({x:2d}, {y:2d}): train={train_walls[idx]}, test={test_walls[idx]}")

    # Verify with later examples
    print(f"\n" + "="*80)
    print("CHECKING OTHER EXAMPLES")
    print("="*80)

    for ex_idx in range(min(5, len(train_data['examples']), len(test_data['examples']))):
        train_ex_walls = np.array(train_data['examples'][ex_idx]['nodes'])[:, 9:13]
        test_ex_walls = np.array(test_data['examples'][ex_idx]['nodes'])[:, 9:13]

        identical = np.array_equal(train_ex_walls, test_ex_walls)
        print(f"Example {ex_idx}: walls identical = {identical}")

        if ex_idx == 0:
            # Check if dataset.json examples share the same walls
            train_ex0_walls = train_ex_walls
        else:
            same_as_ex0 = np.array_equal(train_ex_walls, train_ex0_walls)
            print(f"  dataset.json ex{ex_idx} vs ex0: {same_as_ex0}")

    print(f"\nCONCLUSION:")
    print(f"dataset.json and test.json were generated with DIFFERENT board/wall configurations!")
    print(f"This is a COMPLETELY DIFFERENT PROBLEM than distribution shift.")
    print(f"The model cannot generalize to a different board layout!")
