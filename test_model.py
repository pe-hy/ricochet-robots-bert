"""
Test script to verify the node classifier model setup.

This script:
1. Loads the dataset
2. Creates the model
3. Runs a forward pass
4. Checks dimensions and outputs
"""

import torch
from model import NodeClassifierConfig, create_model, RicochetRobotsDataset

def test_data_loading():
    """Test data loading and preprocessing"""
    print("=" * 80)
    print("TEST 1: Data Loading")
    print("=" * 80)

    dataset = RicochetRobotsDataset(
        data_path='data/ricochet_data/dataset.json',
        board_size=16
    )

    print(f"Dataset size: {len(dataset)}")

    # Get first sample
    sample = dataset[0]
    features = sample['features']
    labels = sample['labels']

    print(f"Features shape: {features.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Feature dim: {features.shape[1]}")
    print(f"Number of nodes: {features.shape[0]}")
    print(f"Number of positive labels: {labels.sum().item()}")
    print(f"Number of negative labels: {(labels == 0).sum().item()}")
    print(f"Positive ratio: {labels.float().mean().item():.4f}")

    # Check feature values
    print(f"\nFirst node features (first 20 dims): {features[0, :20]}")
    print(f"First node label: {labels[0].item()}")

    print("✓ Data loading test passed!\n")
    return dataset

def test_model_forward():
    """Test model forward pass"""
    print("=" * 80)
    print("TEST 2: Model Forward Pass")
    print("=" * 80)

    # Create model
    config = NodeClassifierConfig(
        feature_dim=43,  # 11 + 2*16
        d_model=256,
        nhead=8,
        num_layers=6,
        dim_feedforward=1024,
        dropout=0.1,
    )

    model = create_model(config)
    print(f"Model created with config:")
    print(f"  - feature_dim: {config.feature_dim}")
    print(f"  - d_model: {config.d_model}")
    print(f"  - nhead: {config.nhead}")
    print(f"  - num_layers: {config.num_layers}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters:")
    print(f"  - Total: {total_params:,}")
    print(f"  - Trainable: {trainable_params:,}")

    # Create dummy input
    batch_size = 4
    num_nodes = 256
    feature_dim = 43

    features = torch.randn(batch_size, num_nodes, feature_dim)

    print(f"\nInput shape: {features.shape}")

    # Forward pass
    model.eval()
    with torch.no_grad():
        logits = model(features)
        probs = model.predict_proba(features)

    print(f"Output logits shape: {logits.shape}")
    print(f"Output probs shape: {probs.shape}")
    print(f"Probs range: [{probs.min().item():.4f}, {probs.max().item():.4f}]")
    print(f"Mean prob: {probs.mean().item():.4f}")

    # Check output dimensions
    assert logits.shape == (batch_size, num_nodes, 1), f"Expected shape {(batch_size, num_nodes, 1)}, got {logits.shape}"
    assert probs.shape == (batch_size, num_nodes), f"Expected shape {(batch_size, num_nodes)}, got {probs.shape}"
    assert (probs >= 0).all() and (probs <= 1).all(), "Probabilities should be in [0, 1]"

    print("✓ Model forward pass test passed!\n")
    return model

def test_model_with_real_data():
    """Test model with real dataset"""
    print("=" * 80)
    print("TEST 3: Model with Real Data")
    print("=" * 80)

    # Load dataset
    dataset = RicochetRobotsDataset(
        data_path='data/ricochet_data/dataset.json',
        board_size=16
    )

    # Create model
    config = NodeClassifierConfig(feature_dim=43)
    model = create_model(config)
    model.eval()

    # Get batch
    batch_size = 4
    samples = [dataset[i] for i in range(batch_size)]

    # Stack into batch
    features = torch.stack([s['features'] for s in samples])
    labels = torch.stack([s['labels'] for s in samples])

    print(f"Batch features shape: {features.shape}")
    print(f"Batch labels shape: {labels.shape}")

    # Forward pass
    with torch.no_grad():
        logits = model(features)
        probs = model.predict_proba(features)

    print(f"Output probs shape: {probs.shape}")

    # Compute simple accuracy
    preds = (probs > 0.5).long()
    accuracy = (preds == labels).float().mean()

    print(f"Random initialization accuracy: {accuracy.item():.4f}")
    print(f"Expected random accuracy (class balance): {labels.float().mean().item():.4f}")

    print("✓ Model with real data test passed!\n")

def test_loss_computation():
    """Test loss computation"""
    print("=" * 80)
    print("TEST 4: Loss Computation")
    print("=" * 80)

    import torch.nn.functional as F

    # Create dummy data
    batch_size = 4
    num_nodes = 256

    logits = torch.randn(batch_size, num_nodes, 1)
    labels = torch.randint(0, 2, (batch_size, num_nodes)).float()

    print(f"Logits shape: {logits.shape}")
    print(f"Labels shape: {labels.shape}")

    # Compute loss
    loss = F.binary_cross_entropy_with_logits(logits.squeeze(-1), labels)

    print(f"Loss value: {loss.item():.4f}")
    print(f"Expected loss range: [0, ∞), typical: [0.1, 1.0]")

    assert not torch.isnan(loss), "Loss is NaN!"
    assert not torch.isinf(loss), "Loss is infinite!"

    print("✓ Loss computation test passed!\n")

def main():
    """Run all tests"""
    print("\n" + "=" * 80)
    print("RUNNING MODEL TESTS")
    print("=" * 80 + "\n")

    try:
        # Test 1: Data loading
        dataset = test_data_loading()

        # Test 2: Model forward pass
        model = test_model_forward()

        # Test 3: Model with real data
        test_model_with_real_data()

        # Test 4: Loss computation
        test_loss_computation()

        print("=" * 80)
        print("ALL TESTS PASSED!")
        print("=" * 80)
        print("\nYou can now run training with:")
        print("  python train_node_classifier.py --config config/node_classifier.yaml")
        print()

    except Exception as e:
        print("\n" + "=" * 80)
        print("TEST FAILED!")
        print("=" * 80)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == '__main__':
    exit(main())
