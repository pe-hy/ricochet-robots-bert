"""
Test script to verify different positional encoding strategies work correctly.
"""

import torch
from model import RicochetRobotsDataset, NodeClassifierConfig, create_model

def test_positional_encoding(encoding_type, encoding_kwargs=None):
    """Test a specific positional encoding strategy"""
    print(f"Testing {encoding_type} positional encoding...")

    # Create dataset with specific encoding
    dataset = RicochetRobotsDataset(
        data_path='data/ricochet_data/dataset.json',
        board_size=16,
        positional_encoding=encoding_type,
        positional_encoding_kwargs=encoding_kwargs or {}
    )

    # Get sample
    sample = dataset[0]
    features = sample['features']
    labels = sample['labels']

    print(f"  Feature shape: {features.shape}")
    print(f"  Feature dim: {features.shape[1]}")

    # Create model with appropriate feature_dim
    config = NodeClassifierConfig(
        feature_dim=features.shape[1],
        d_model=128,  # Smaller for faster testing
        nhead=4,
        num_layers=2
    )

    model = create_model(config)
    model.eval()

    # Test forward pass
    batch = features.unsqueeze(0)  # [1, 256, feature_dim]
    with torch.no_grad():
        logits = model(batch)
        probs = model.predict_proba(batch)

    print(f"  Output shape: {logits.shape}")
    print(f"  Probs range: [{probs.min():.4f}, {probs.max():.4f}]")
    print(f"  ✓ {encoding_type} encoding works!\n")

    return True


def main():
    """Test all positional encoding strategies"""
    print("=" * 80)
    print("TESTING POSITIONAL ENCODING STRATEGIES")
    print("=" * 80 + "\n")

    # Test one-hot encoding
    test_positional_encoding('onehot')

    # Test sinusoidal encoding
    test_positional_encoding('sinusoidal', {'encoding_dim': 64})

    # Test normalized encoding
    test_positional_encoding('normalized')

    # Test learned encoding (placeholder)
    test_positional_encoding('learned', {'embedding_dim': 64})

    print("=" * 80)
    print("ALL POSITIONAL ENCODINGS WORK!")
    print("=" * 80)
    print("\nTo use different encodings, edit config/node_classifier.yaml:")
    print('  positional_encoding: "onehot"     # Default')
    print('  positional_encoding: "sinusoidal" # Fixed sinusoidal')
    print('  positional_encoding: "normalized" # Simple normalized coords')
    print()
    print("For sinusoidal, add kwargs:")
    print('  positional_encoding_kwargs: {encoding_dim: 64}')
    print()

if __name__ == '__main__':
    main()
