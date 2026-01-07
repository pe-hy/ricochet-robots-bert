"""
Evaluation script for the trained node classifier on held-out test set.

Loads the best checkpoint and evaluates on test.json, saving predictions in a specific format:
- Pickle file with key "examples" containing list of (node_features, predicted_coords) tuples
- node_features: raw feature vectors (256 x feature_dim for 16x16 grid)
- predicted_coords: list of (x, y) tuples where model predicted label=1

Usage:
    python evaluate_test_set.py \
        --checkpoint ./tmp/checkpoints/d256_l6_h8_n10000_lr0.001_bs256/best.ckpt \
        --test_data data/ricochet_data/test.json \
        --output predictions.pkl \
        --board_size 16
"""

import argparse
import json
import pickle
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

from model.lightning_module import NodeClassifierLightningModule
from utils.data_module import RicochetRobotsDataset


def load_model(checkpoint_path: str, device: str = 'cuda') -> NodeClassifierLightningModule:
    """Load model from checkpoint"""
    print(f"Loading checkpoint from: {checkpoint_path}")
    model = NodeClassifierLightningModule.load_from_checkpoint(checkpoint_path)
    model = model.to(device)
    model.eval()
    print(f"Model loaded successfully on {device}")
    return model


def evaluate_test_set(
    model: NodeClassifierLightningModule,
    test_data_path: str,
    board_size: int = 16,
    threshold: float = 0.5,
    device: str = 'cuda'
) -> list:
    """
    Evaluate model on test set and return predictions in required format.

    Returns:
        List of tuples: [(node_features, predicted_coords), ...]
        - node_features: np.ndarray of shape [num_nodes, feature_dim]
        - predicted_coords: list of (x, y) tuples where prediction == 1
    """
    # Load test dataset
    print(f"Loading test dataset from: {test_data_path}")

    # Read the config from the model to get positional encoding settings
    # We'll assume default settings here - adjust if needed
    test_dataset = RicochetRobotsDataset(
        data_path=test_data_path,
        board_size=board_size,
        positional_encoding='onehot',  # Match your training config
        positional_encoding_kwargs={}
    )

    print(f"Test dataset size: {len(test_dataset)} examples")

    examples = []

    with torch.no_grad():
        for idx in tqdm(range(len(test_dataset)), desc="Evaluating test set"):
            sample = test_dataset[idx]
            features = sample['features']  # [num_nodes, feature_dim]

            # Convert to batch format for model
            features_batch = features.unsqueeze(0).to(device)  # [1, num_nodes, feature_dim]

            # Get predictions
            logits = model(features_batch)  # [1, num_nodes, 1]
            probs = torch.sigmoid(logits.squeeze(-1))  # [1, num_nodes]
            preds = (probs > threshold).long()  # [1, num_nodes]

            # Convert back to CPU and extract
            preds = preds.squeeze(0).cpu().numpy()  # [num_nodes]
            features_np = features.cpu().numpy()  # [num_nodes, feature_dim]

            # Extract (x, y) coordinates where prediction == 1
            predicted_coords = []
            for node_idx in range(len(preds)):
                if preds[node_idx] == 1:
                    # Convert node index to (x, y) coordinates
                    # Assuming row-major ordering: node_idx = y * board_size + x
                    x = node_idx % board_size
                    y = node_idx // board_size
                    predicted_coords.append((x, y))

            # Store in required format
            examples.append((features_np, predicted_coords))

    return examples


def compute_metrics(examples: list, test_data_path: str, board_size: int = 16):
    """Compute evaluation metrics by comparing with ground truth"""
    # Load ground truth labels
    with open(test_data_path, 'r') as f:
        data = json.load(f)

    total_nodes = 0
    correct_nodes = 0
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    exact_matches = 0

    for idx, (node_features, pred_coords) in enumerate(examples):
        # Get ground truth from dataset
        gt_nodes = data['examples'][idx]['nodes']
        gt_labels = [int(node[13]) for node in gt_nodes]  # Label is at index 13

        # Convert predicted coords to binary array
        pred_labels = [0] * len(gt_labels)
        for x, y in pred_coords:
            node_idx = y * board_size + x
            pred_labels[node_idx] = 1

        # Compute metrics
        for gt_label, pred_label in zip(gt_labels, pred_labels):
            total_nodes += 1
            if gt_label == pred_label:
                correct_nodes += 1
            if gt_label == 1 and pred_label == 1:
                true_positives += 1
            if gt_label == 0 and pred_label == 1:
                false_positives += 1
            if gt_label == 1 and pred_label == 0:
                false_negatives += 1

        # Check exact match (all nodes correct)
        if gt_labels == pred_labels:
            exact_matches += 1

    # Calculate metrics
    accuracy = correct_nodes / total_nodes if total_nodes > 0 else 0
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    exact_match_rate = exact_matches / len(examples) if len(examples) > 0 else 0

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'exact_match_rate': exact_match_rate,
        'total_examples': len(examples),
        'exact_matches': exact_matches
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained model on test set')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--test_data', type=str, default='data/ricochet_data/test.json',
                        help='Path to test data JSON')
    parser.add_argument('--output', type=str, default='test_predictions.pkl',
                        help='Output pickle file path')
    parser.add_argument('--board_size', type=int, default=16,
                        help='Board size (default: 16)')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Prediction threshold (default: 0.5)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--compute_metrics', action='store_true',
                        help='Compute and display evaluation metrics')

    args = parser.parse_args()

    # Check if test data exists
    if not Path(args.test_data).exists():
        print(f"ERROR: Test data not found at {args.test_data}")
        return

    # Check if checkpoint exists
    if not Path(args.checkpoint).exists():
        print(f"ERROR: Checkpoint not found at {args.checkpoint}")
        return

    # Set device
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'

    print("=" * 80)
    print("EVALUATION CONFIGURATION")
    print("=" * 80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Test data: {args.test_data}")
    print(f"Output file: {args.output}")
    print(f"Board size: {args.board_size}")
    print(f"Threshold: {args.threshold}")
    print(f"Device: {device}")
    print("=" * 80)

    # Load model
    model = load_model(args.checkpoint, device=device)

    # Evaluate
    examples = evaluate_test_set(
        model=model,
        test_data_path=args.test_data,
        board_size=args.board_size,
        threshold=args.threshold,
        device=device
    )

    # Save predictions in required format
    print(f"\nSaving predictions to: {args.output}")
    with open(args.output, 'wb') as f:
        pickle.dump({'examples': examples}, f)

    print(f"✓ Saved {len(examples)} examples")

    # Compute metrics if requested
    if args.compute_metrics:
        print("\n" + "=" * 80)
        print("EVALUATION METRICS")
        print("=" * 80)
        metrics = compute_metrics(examples, args.test_data, args.board_size)
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")
        print(f"Exact Match Rate: {metrics['exact_match_rate']:.4f} ({metrics['exact_matches']}/{metrics['total_examples']})")
        print("=" * 80)

    print("\n✓ Evaluation complete!")


if __name__ == '__main__':
    main()
