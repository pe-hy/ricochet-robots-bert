"""Check if the current pkl board matches dataset.json or test.json"""
import pickle
import json
import numpy as np

# Load the current board pkl
with open('robots/temp/grid_graph_0.pkl', 'rb') as f:
    current_board = pickle.load(f)

print("="*80)
print("CHECKING WHICH DATASET MATCHES THE CURRENT BOARD PKL")
print("="*80)

print("\nPKL file timestamp: Oct 30 22:15")
print("dataset.json timestamp: Oct 30 15:09")
print("test.json timestamp: Oct 30 11:21")
print("test_256.json timestamp: Oct 30 22:22")

# Extract wall configuration from pkl
def get_walls_from_pkl(board):
    """Extract wall configuration from board graph"""
    walls = np.zeros((256, 4), dtype=int)

    for node in board['graph'].nodes():
        x, y = node
        node_idx = y * 16 + x

        # Check for walls
        wall_top = False
        wall_left = False

        # Top wall (edge to (x, y-1))
        if not board['graph'].has_edge((x, y), (x, y-1)):
            wall_top = True

        # Left wall (edge to (x-1, y))
        if not board['graph'].has_edge((x, y), (x-1, y)):
            wall_left = True

        # Encode as one-hot
        # wall_none, wall_top, wall_left, wall_both
        if not wall_top and not wall_left:
            walls[node_idx] = [1, 0, 0, 0]
        elif wall_top and not wall_left:
            walls[node_idx] = [0, 1, 0, 0]
        elif not wall_top and wall_left:
            walls[node_idx] = [0, 0, 1, 0]
        else:  # both
            walls[node_idx] = [0, 0, 0, 1]

    return walls

pkl_walls = get_walls_from_pkl(current_board)

# Load datasets
with open('data/ricochet_data/dataset.json', 'r') as f:
    dataset_data = json.load(f)

with open('data/ricochet_data/test.json', 'r') as f:
    test_data = json.load(f)

# Extract walls from first example of each
dataset_walls = np.array(dataset_data['examples'][0]['nodes'])[:, 9:13]
test_walls = np.array(test_data['examples'][0]['nodes'])[:, 9:13]

# Compare
dataset_match = np.array_equal(pkl_walls, dataset_walls)
test_match = np.array_equal(pkl_walls, test_walls)

print(f"\n" + "="*80)
print("COMPARISON RESULTS")
print("="*80)

print(f"\nCurrent PKL matches dataset.json: {dataset_match}")
print(f"Current PKL matches test.json: {test_match}")

if dataset_match:
    print("\n✓ The current board PKL (22:15) matches dataset.json (15:09)")
    print("This is IMPOSSIBLE - the PKL was created AFTER dataset.json!")
    print("This means the board was regenerated and happened to be the same.")
elif test_match:
    print("\n✓ The current board PKL (22:15) matches test.json (11:21)")
    print("This means test.json was regenerated or used the same seed.")
else:
    print("\n❌ The current board PKL matches NEITHER dataset!")
    print("The board has been regenerated multiple times.")

# Check if test_256.json matches the current pkl
try:
    with open('data/ricochet_data/test_256.json', 'r') as f:
        test_256_data = json.load(f)
    test_256_walls = np.array(test_256_data['examples'][0]['nodes'])[:, 9:13]
    test_256_match = np.array_equal(pkl_walls, test_256_walls)

    print(f"Current PKL matches test_256.json: {test_256_match}")

    if test_256_match:
        print("\n✓ test_256.json (22:22) was generated using the current board PKL (22:15)")
except:
    print("\nCouldn't check test_256.json")

# Show differences
if not dataset_match and not test_match:
    print(f"\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("\nThere have been at least 3 different boards generated:")
    print("  1. Board used for test.json (Oct 30 11:21)")
    print("  2. Board used for dataset.json (Oct 30 15:09)")
    print("  3. Current board in PKL (Oct 30 22:15)")
