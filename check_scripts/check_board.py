import json

# Check training data
with open('data/ricochet_data/dataset.json', 'r') as f:
    train = json.load(f)
    
# Check test data  
with open('data/ricochet_data/test.json', 'r') as f:
    test = json.load(f)

print("TRAINING DATA:")
print(f"  Examples: {train['metadata']['num_examples']}")
print(f"  First example, first node: {train['examples'][0]['nodes'][0]}")

print("\nTEST DATA:")
print(f"  Examples: {test['metadata']['num_examples']}")
print(f"  First example, first node: {test['examples'][0]['nodes'][0]}")

# Check if walls are same (comparing node features at same position)
train_walls = train['examples'][0]['nodes'][0][9:13]
test_walls = test['examples'][0]['nodes'][0][9:13]
print(f"\nSame wall configuration? {train_walls == test_walls}")