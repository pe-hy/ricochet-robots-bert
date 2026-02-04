"""
Test script to verify combined exact match metric works correctly.
"""

import torch
from model.models import MultiTaskConfig, create_multitask_model
from model.multitask_lightning_module import MultiTaskLightningModule


def test_combined_exact_match():
    """Test that combined exact match metric is computed correctly."""
    print("\n" + "="*80)
    print("TEST: Combined Exact Match Metric")
    print("="*80)

    # Create a small model
    config = MultiTaskConfig(feature_dim=44, d_model=64, num_layers=2, architecture='basic')
    model = create_multitask_model(config)

    # Create lightning module
    lightning_module = MultiTaskLightningModule(
        model=model,
        model_config=config,
        max_lr=0.001,
        weight_decay=0.01,
        warmup_epochs=5,
        total_epochs=10,
        steps_per_epoch=10,
        log_predictions=False,
    )

    # Create synthetic batch
    batch_size = 4
    num_nodes = 256

    features = torch.randn(batch_size, num_nodes, 44)

    # Create labels where:
    # - Example 0: All tasks have all correct predictions (all zeros)
    # - Example 1: Task 1 has one error
    # - Example 2: Task 2 has one error
    # - Example 3: All tasks have errors
    labels = {}
    for task_name in ['subgoal_label', 'helper_aggregate', 'target_pos', 'chosen_helper']:
        labels[task_name] = torch.zeros(batch_size, num_nodes, dtype=torch.long)

    # Add errors
    labels['subgoal_label'][1, 0] = 1      # Example 1: error in task 1
    labels['helper_aggregate'][2, 0] = 1   # Example 2: error in task 2
    labels['target_pos'][3, 0] = 1         # Example 3: error in task 3
    labels['chosen_helper'][3, 1] = 1      # Example 3: another error

    batch = {'features': features, 'labels': labels}

    # Set model to eval mode
    lightning_module.eval()

    # Forward pass
    with torch.no_grad():
        # Get logits (all zeros since model is random and untrained)
        logits = lightning_module(features)

        # Manually set logits to predict all zeros (for testing)
        for task_name in logits.keys():
            logits[task_name] = torch.full_like(logits[task_name], -10.0)  # Sigmoid -> ~0

        # Compute metrics
        metrics = lightning_module._compute_metrics(logits, labels)

    # Check per-task exact matches
    print("\nPer-task exact matches:")
    for task_name in ['subgoal_label', 'helper_aggregate', 'target_pos', 'chosen_helper']:
        exact_match = metrics[task_name]['exact_matches']
        print(f"  {task_name}: {exact_match.tolist()}")

    # Compute combined exact match
    all_tasks_exact_match = torch.stack([
        metrics[task_name]['exact_matches']
        for task_name in ['subgoal_label', 'helper_aggregate', 'target_pos', 'chosen_helper']
    ], dim=1)  # [batch, 4]

    combined_exact_match = all_tasks_exact_match.all(dim=1).float()
    print(f"\nCombined exact match (all 4 tasks): {combined_exact_match.tolist()}")
    print(f"Average: {combined_exact_match.mean():.4f}")

    # Expected results:
    # Example 0: All tasks correct -> combined = 1.0
    # Example 1: Task 1 wrong -> combined = 0.0
    # Example 2: Task 2 wrong -> combined = 0.0
    # Example 3: Multiple tasks wrong -> combined = 0.0
    expected = torch.tensor([1.0, 0.0, 0.0, 0.0])

    assert torch.allclose(combined_exact_match, expected), \
        f"Expected {expected.tolist()}, got {combined_exact_match.tolist()}"

    print("\n✓ Combined exact match metric works correctly!")
    print("\nInterpretation:")
    print("  - Example 0: All 4 tasks correct -> combined exact match = 1.0")
    print("  - Example 1-3: At least one task wrong -> combined exact match = 0.0")
    print("  - This metric is VERY strict - all 256 nodes × 4 tasks must be correct!")


if __name__ == "__main__":
    test_combined_exact_match()
    print("\n" + "="*80)
    print("TEST PASSED ✓")
    print("="*80)
