#!/usr/bin/env python3
"""
Evaluate all 4 best checkpoints on IID and OOD test sets with branching logic.

This script:
1. Loads all 4 checkpoints from best_ckpts/
2. Evaluates them in pipeline order with branching for multiple predictions
3. Uses model predictions (not ground truth) as input for subsequent models
4. Saves per-example branched predictions to predictions.pkl
5. Saves statistics (exact match accuracy, etc.) to CSV

Pipeline order: subgoal_label → helper_aggregate → target_pos → chosen_helper

When a model predicts multiple outputs (multiple nodes > threshold), the example
branches into separate paths, each continuing through the pipeline independently.

Usage:
    python evaluate_best_checkpoints.py
    python evaluate_best_checkpoints.py --checkpoint-dir best_ckpts --output-dir results
"""

import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import torch
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
from omegaconf import OmegaConf
from copy import deepcopy

from model.lightning_module import NodeClassifierLightningModule
from utils.data_module import RicochetRobotsDataset
from utils.positional_encoding import create_positional_encoding


# Checkpoint to task mapping (based on validation EM in filename)
CHECKPOINT_TO_TASK = {
    'epoch=26-val_em=1.0000.ckpt': 'chosen_helper',
    'epoch=74-val_em=0.9023.ckpt': 'helper_aggregate',
    'epoch=54-val_em=0.9805.ckpt': 'target_pos',
    'epoch=72-val_em=0.8750.ckpt': 'subgoal_label',
}

# Task order in pipeline
TASK_ORDER = ['subgoal_label', 'helper_aggregate', 'target_pos', 'chosen_helper']

# Task to node index mapping for goal features
TASK_TO_NODE_INDEX = {
    'subgoal_label': 20,      # Index 20 in node data
    'chosen_helper': 19,      # Index 19 in node data
    'target_pos': 18,         # Index 18 in node data
    'helper_aggregate': 17,   # Index 17 in node data
}


@dataclass
class Branch:
    """Represents a single prediction branch through the pipeline."""
    original_example_id: int
    raw_example: Dict  # Original example with nodes
    predictions: Dict[str, np.ndarray]  # task -> node-level binary predictions [num_nodes]
    predicted_coords: Dict[str, List[Tuple[int, int]]]  # task -> list of predicted (x,y)
    probabilities: Dict[str, np.ndarray]  # task -> node probabilities [num_nodes]
    board_size: int = 16

    def copy_with_new_prediction(self, task: str, node_idx: int, prob: float, board_size: int):
        """Create a new branch with a single prediction for the given task."""
        new_branch = Branch(
            original_example_id=self.original_example_id,
            raw_example=self.raw_example,
            predictions=deepcopy(self.predictions),
            predicted_coords=deepcopy(self.predicted_coords),
            probabilities=deepcopy(self.probabilities),
            board_size=board_size
        )

        # Create binary prediction array with only this node set to 1
        num_nodes = board_size * board_size
        binary_pred = np.zeros(num_nodes, dtype=np.int64)
        binary_pred[node_idx] = 1
        new_branch.predictions[task] = binary_pred

        # Store the coordinate
        x = node_idx % board_size
        y = node_idx // board_size
        new_branch.predicted_coords[task] = [(x, y)]

        # Store probability array (for this branch, only this node is relevant)
        prob_array = np.zeros(num_nodes, dtype=np.float32)
        prob_array[node_idx] = prob
        new_branch.probabilities[task] = prob_array

        return new_branch


def load_checkpoint_with_task(checkpoint_path: str, task_name: str):
    """Load a checkpoint and its task configuration."""
    # Load task config
    task_config_path = f'config/task/{task_name}.yaml'
    task_config = OmegaConf.load(task_config_path)

    # Load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = NodeClassifierLightningModule.load_from_checkpoint(checkpoint_path)
    model = model.to(device)
    model.eval()

    return model, OmegaConf.to_container(task_config, resolve=True), device


def build_features_from_branch(
    branch: Branch,
    task_config: Dict,
    positional_encoding: str = 'onehot',
    positional_encoding_kwargs: Optional[Dict] = None
) -> torch.Tensor:
    """
    Build node features for a branch using predicted labels from previous tasks.

    Args:
        branch: Branch containing predictions from previous tasks
        task_config: Task configuration with include_goal_features
        positional_encoding: Type of positional encoding
        positional_encoding_kwargs: Additional kwargs for positional encoding

    Returns:
        features: [num_nodes, feature_dim] tensor
    """
    board_size = branch.board_size
    num_nodes = board_size * board_size

    # Create positional encoder
    encoder_kwargs = positional_encoding_kwargs or {}
    encoder_kwargs = {k: v for k, v in encoder_kwargs.items() if k not in ('combine_method',)}
    pos_encoder = create_positional_encoding(positional_encoding, **encoder_kwargs)

    # Get nodes from raw example
    nodes = branch.raw_example['nodes']

    features_list = []
    for node_idx, node in enumerate(nodes):
        # Extract coordinates
        x = int(node[0])
        y = int(node[1])

        # Extract base features (these don't change)
        robot_type = node[2:7]      # 5 dims
        has_goal = node[7:9]         # 2 dims
        walls = node[9:14]           # 5 dims

        # Build goal features using PREDICTED labels from this branch
        goal_features = []
        include_indices = task_config.get('include_goal_features', [])
        for idx in include_indices:
            # Map node index to task name
            if idx == 14:  # helper1_goal_pos
                # This is a helper - need to figure out which one
                # For simplicity, we'll use ground truth for individual helpers
                # since they're not in our task pipeline
                goal_features.append(node[idx])
            elif idx == 15:  # helper2_goal_pos
                goal_features.append(node[idx])
            elif idx == 16:  # helper3_goal_pos
                goal_features.append(node[idx])
            elif idx == 17:  # helper_aggregate_goal_pos
                # Use predicted helper_aggregate if available
                if 'helper_aggregate' in branch.predictions:
                    goal_features.append(float(branch.predictions['helper_aggregate'][node_idx]))
                else:
                    goal_features.append(node[idx])  # Fallback to ground truth if not yet predicted
            elif idx == 18:  # target_pos
                # Use predicted target_pos if available
                if 'target_pos' in branch.predictions:
                    goal_features.append(float(branch.predictions['target_pos'][node_idx]))
                else:
                    goal_features.append(node[idx])
            elif idx == 19:  # chosen_helper
                # Use predicted chosen_helper if available
                if 'chosen_helper' in branch.predictions:
                    goal_features.append(float(branch.predictions['chosen_helper'][node_idx]))
                else:
                    goal_features.append(node[idx])
            elif idx == 20:  # subgoal_label
                # Use predicted subgoal_label if available
                if 'subgoal_label' in branch.predictions:
                    goal_features.append(float(branch.predictions['subgoal_label'][node_idx]))
                else:
                    goal_features.append(node[idx])
            else:
                # Unknown index, use ground truth
                goal_features.append(node[idx])

        # Encode coordinates
        pos_encoding = pos_encoder.encode(x, y, board_size)

        # Concatenate all features
        feature_parts = [robot_type, has_goal, walls]
        if len(goal_features) > 0:
            feature_parts.append(goal_features)
        feature_parts.append(pos_encoding)

        features = np.concatenate(feature_parts).astype(np.float32)
        features_list.append(features)

    # Convert to tensor
    features_tensor = torch.tensor(np.stack(features_list), dtype=torch.float32)  # [num_nodes, feature_dim]

    return features_tensor


def evaluate_all_checkpoints(
    checkpoint_dir: str,
    test_data_paths: dict,
    board_size: int = 16,
    threshold: float = 0.5,
    positional_encoding: str = 'onehot'
):
    """
    Evaluate all 4 checkpoints on all test sets with branching pipeline logic.

    Args:
        checkpoint_dir: Directory containing the 4 checkpoint files
        test_data_paths: Dict mapping test set names to paths
        board_size: Board size
        threshold: Prediction threshold
        positional_encoding: Type of positional encoding

    Returns:
        Tuple of (statistics DataFrame, predictions dict)
    """
    checkpoint_dir = Path(checkpoint_dir)

    # Load all checkpoints
    print("=" * 80)
    print("LOADING CHECKPOINTS")
    print("=" * 80)

    models = {}
    device = None
    for ckpt_filename, task_name in CHECKPOINT_TO_TASK.items():
        ckpt_path = checkpoint_dir / ckpt_filename
        if not ckpt_path.exists():
            print(f"WARNING: Checkpoint not found: {ckpt_path}")
            continue

        print(f"Loading {task_name}: {ckpt_filename}")
        model, task_config, device = load_checkpoint_with_task(str(ckpt_path), task_name)
        models[task_name] = {
            'model': model,
            'task_config': task_config,
            'checkpoint_path': str(ckpt_path)
        }

    print(f"\nLoaded {len(models)} models")
    print("=" * 80)

    # Evaluate on each test set
    all_statistics = []
    all_predictions = {}

    for test_name, test_path in test_data_paths.items():
        print(f"\n{'=' * 80}")
        print(f"EVALUATING ON {test_name.upper()} TEST SET WITH BRANCHING PIPELINE")
        print(f"{'=' * 80}")
        print(f"Test data: {test_path}")

        if not Path(test_path).exists():
            print(f"WARNING: Test data not found at {test_path}. Skipping.")
            continue

        # Evaluate with branching pipeline
        statistics, test_predictions = evaluate_pipeline_with_branching(
            models=models,
            test_path=test_path,
            board_size=board_size,
            threshold=threshold,
            positional_encoding=positional_encoding,
            device=device
        )

        # Add test set name to statistics
        for stats in statistics:
            stats['test_set'] = test_name
            all_statistics.append(stats)

        # Store predictions for this test set (organized by task)
        all_predictions[test_name] = test_predictions

        # Print summary
        print(f"\n{'=' * 80}")
        print(f"{test_name.upper()} TEST SET SUMMARY")
        print(f"{'=' * 80}")
        print(f"{'Task':<20} {'Exact Match':<15} {'Accuracy':<10} {'Branches':<10}")
        print("-" * 80)
        for stats in statistics:
            branches_str = f"{stats.get('num_branches', 'N/A')}"
            print(f"{stats['task']:<20} {stats['exact_match']:<15.4f} {stats['accuracy']:<10.4f} {branches_str:<10}")

    return pd.DataFrame(all_statistics), all_predictions


def evaluate_pipeline_with_branching(
    models: Dict,
    test_path: str,
    board_size: int,
    threshold: float,
    positional_encoding: str,
    device: str
):
    """
    Evaluate the full pipeline with branching logic.

    When a model predicts multiple nodes, creates separate branches for each prediction.
    Uses model predictions (not ground truth) as input to subsequent models.

    Args:
        models: Dict of {task_name: {'model': model, 'task_config': config, ...}}
        test_path: Path to test data
        board_size: Board size
        threshold: Prediction threshold (0.5)
        positional_encoding: Type of positional encoding
        device: Device to run on

    Returns:
        Tuple of (statistics list, predictions dict organized by task)
    """
    # Load test data to get raw examples
    import json
    if test_path.endswith('.jsonl'):
        with open(test_path, 'r') as f:
            lines = f.readlines()
        raw_examples = [json.loads(line) for line in lines[1:]]  # Skip metadata
    else:
        with open(test_path, 'r') as f:
            data = json.load(f)
        raw_examples = data['examples']

    print(f"\nStarting with {len(raw_examples)} test examples")

    # Initialize branches: one branch per test example
    branches = []
    for idx, raw_example in enumerate(raw_examples):
        branch = Branch(
            original_example_id=idx,
            raw_example=raw_example,
            predictions={},
            predicted_coords={},
            probabilities={},
            board_size=board_size
        )
        branches.append(branch)

    # Statistics for each task
    all_statistics = []

    # Store predictions in old format: task -> list of examples
    task_predictions = {task: [] for task in TASK_ORDER}

    # Process each task in pipeline order
    for task_name in TASK_ORDER:
        if task_name not in models:
            print(f"\n  WARNING: Skipping {task_name} (model not loaded)")
            continue

        print(f"\n  Processing {task_name}...")
        print(f"    Input branches: {len(branches)}")

        model_info = models[task_name]
        model = model_info['model']
        task_config = model_info['task_config']

        # Process all branches and collect new branches
        new_branches = []
        # Per-branch metrics (current behavior)
        task_metrics_per_branch = {
            'total_nodes': 0,
            'correct_nodes': 0,
            'true_positives': 0,
            'false_positives': 0,
            'false_negatives': 0,
            'exact_matches': 0,
            'total_branches': 0,
            'examples_with_no_predictions': 0
        }
        # Per-original-example metrics (aggregated branches)
        task_metrics_aggregated = {
            'total_nodes': 0,
            'correct_nodes': 0,
            'true_positives': 0,
            'false_positives': 0,
            'false_negatives': 0,
            'exact_matches': 0,
            'total_examples': 0,
            'examples_with_no_predictions': 0
        }

        with torch.no_grad():
            for branch in tqdm(branches, desc=f"  {task_name}", leave=False):
                # Build features using predictions from previous tasks
                features = build_features_from_branch(
                    branch=branch,
                    task_config=task_config,
                    positional_encoding=positional_encoding,
                    positional_encoding_kwargs={}
                )

                # Get ground truth labels for this task
                target_index = task_config.get('target_index', -1)
                labels = []
                for node in branch.raw_example['nodes']:
                    labels.append(float(node[target_index]))
                labels = np.array(labels, dtype=np.int64)

                # Run model prediction
                features_batch = features.unsqueeze(0).to(device)  # [1, num_nodes, feature_dim]
                logits = model(features_batch)  # [1, num_nodes, 1]
                probs = torch.sigmoid(logits.squeeze(-1))  # [1, num_nodes]

                # Convert to numpy
                probs_np = probs.squeeze(0).cpu().numpy()  # [num_nodes]

                # Find all nodes above threshold
                predicted_node_indices = np.where(probs_np > threshold)[0]

                # Branch creation logic
                if len(predicted_node_indices) == 0:
                    # No prediction - create empty branch to track this
                    task_metrics_per_branch['examples_with_no_predictions'] += 1
                    empty_branch = branch.copy_with_new_prediction(
                        task=task_name,
                        node_idx=0,  # Dummy, will be overridden
                        prob=0.0,
                        board_size=board_size
                    )
                    # Override to all zeros
                    num_nodes = board_size * board_size
                    empty_branch.predictions[task_name] = np.zeros(num_nodes, dtype=np.int64)
                    empty_branch.predicted_coords[task_name] = []
                    empty_branch.probabilities[task_name] = np.zeros(num_nodes, dtype=np.float32)
                    new_branches.append(empty_branch)
                    continue
                elif len(predicted_node_indices) == 1:
                    # Single prediction - continue with existing branch
                    node_idx = predicted_node_indices[0]
                    new_branch = branch.copy_with_new_prediction(
                        task=task_name,
                        node_idx=node_idx,
                        prob=probs_np[node_idx],
                        board_size=board_size
                    )
                    new_branches.append(new_branch)

                    # Update PER-BRANCH metrics for this branch
                    task_metrics_per_branch['total_branches'] += 1
                    task_metrics_per_branch['total_nodes'] += len(labels)
                    task_metrics_per_branch['correct_nodes'] += (new_branch.predictions[task_name] == labels).sum()

                    for pred, label in zip(new_branch.predictions[task_name], labels):
                        if pred == 1 and label == 1:
                            task_metrics_per_branch['true_positives'] += 1
                        elif pred == 1 and label == 0:
                            task_metrics_per_branch['false_positives'] += 1
                        elif pred == 0 and label == 1:
                            task_metrics_per_branch['false_negatives'] += 1

                    if (new_branch.predictions[task_name] == labels).all():
                        task_metrics_per_branch['exact_matches'] += 1

                else:
                    # Multiple predictions - create separate branch for EACH prediction
                    for node_idx in predicted_node_indices:
                        new_branch = branch.copy_with_new_prediction(
                            task=task_name,
                            node_idx=node_idx,
                            prob=probs_np[node_idx],
                            board_size=board_size
                        )
                        new_branches.append(new_branch)

                        # Update PER-BRANCH metrics for this branch
                        task_metrics_per_branch['total_branches'] += 1
                        task_metrics_per_branch['total_nodes'] += len(labels)
                        task_metrics_per_branch['correct_nodes'] += (new_branch.predictions[task_name] == labels).sum()

                        for pred, label in zip(new_branch.predictions[task_name], labels):
                            if pred == 1 and label == 1:
                                task_metrics_per_branch['true_positives'] += 1
                            elif pred == 1 and label == 0:
                                task_metrics_per_branch['false_positives'] += 1
                            elif pred == 0 and label == 1:
                                task_metrics_per_branch['false_negatives'] += 1

                        if (new_branch.predictions[task_name] == labels).all():
                            task_metrics_per_branch['exact_matches'] += 1

        # Update branches for next task
        branches = new_branches
        print(f"    Output branches: {len(branches)}")

        # COMPUTE AGGREGATED METRICS (per-original-example)
        # Group branches by original example ID
        branches_by_original_id = {}
        for branch in branches:
            orig_id = branch.original_example_id
            if orig_id not in branches_by_original_id:
                branches_by_original_id[orig_id] = []
            branches_by_original_id[orig_id].append(branch)

        # Compute metrics for each original example (aggregate all its branches)
        for orig_id, example_branches in branches_by_original_id.items():
            # Union all predictions from branches of this example
            num_nodes = board_size * board_size
            aggregated_predictions = np.zeros(num_nodes, dtype=np.int64)

            for branch in example_branches:
                if task_name in branch.predictions:
                    aggregated_predictions = np.logical_or(
                        aggregated_predictions,
                        branch.predictions[task_name]
                    ).astype(np.int64)

            # Get ground truth (same for all branches of this example)
            target_index = task_config.get('target_index', -1)
            labels = []
            for node in example_branches[0].raw_example['nodes']:
                labels.append(float(node[target_index]))
            labels = np.array(labels, dtype=np.int64)

            # Check if this example had no predictions
            if aggregated_predictions.sum() == 0:
                task_metrics_aggregated['examples_with_no_predictions'] += 1

            # Compute metrics for this ORIGINAL EXAMPLE
            task_metrics_aggregated['total_examples'] += 1
            task_metrics_aggregated['total_nodes'] += len(labels)
            task_metrics_aggregated['correct_nodes'] += (aggregated_predictions == labels).sum()

            for pred, label in zip(aggregated_predictions, labels):
                if pred == 1 and label == 1:
                    task_metrics_aggregated['true_positives'] += 1
                elif pred == 1 and label == 0:
                    task_metrics_aggregated['false_positives'] += 1
                elif pred == 0 and label == 1:
                    task_metrics_aggregated['false_negatives'] += 1

            if (aggregated_predictions == labels).all():
                task_metrics_aggregated['exact_matches'] += 1

        # Store predictions for this task
        # Store aggregated predictions (per original example) + branch details
        for orig_id, example_branches in branches_by_original_id.items():
            # Get ground truth
            target_index = task_config.get('target_index', -1)
            labels = []
            for node in example_branches[0].raw_example['nodes']:
                labels.append(float(node[target_index]))
            labels_np = np.array(labels, dtype=np.int64)

            # Extract true coordinates
            true_coords = []
            for node_idx, label in enumerate(labels_np):
                if label == 1:
                    x = node_idx % board_size
                    y = node_idx // board_size
                    true_coords.append((x, y))

            # Aggregate predictions from all branches
            num_nodes = board_size * board_size
            aggregated_predictions = np.zeros(num_nodes, dtype=np.int64)
            aggregated_probabilities = np.zeros(num_nodes, dtype=np.float32)
            all_predicted_coords = []

            for branch in example_branches:
                if task_name in branch.predictions:
                    aggregated_predictions = np.logical_or(
                        aggregated_predictions,
                        branch.predictions[task_name]
                    ).astype(np.int64)
                    # Take max probability across branches
                    aggregated_probabilities = np.maximum(
                        aggregated_probabilities,
                        branch.probabilities[task_name]
                    )
                    all_predicted_coords.extend(branch.predicted_coords[task_name])

            # Remove duplicates from predicted coords
            aggregated_coords = list(set(all_predicted_coords))

            # Create example dict with both aggregated and per-branch info
            example = {
                # Aggregated predictions (for metrics)
                'predicted_coords': aggregated_coords,
                'predictions': aggregated_predictions,
                'probabilities': aggregated_probabilities,
                # Ground truth
                'true_coords': true_coords,
                'labels': labels_np,
                # Metadata
                'original_example_id': orig_id,
                'num_branches': len(example_branches),
                'scored_subgoals': example_branches[0].raw_example.get('scored_subgoals', {}),
                # Per-branch details for analysis
                'branch_details': []
            }

            # Add individual branch details
            for branch in example_branches:
                example['branch_details'].append({
                    'predicted_coords': branch.predicted_coords[task_name],
                    'predictions': branch.predictions[task_name],
                    'probabilities': branch.probabilities[task_name]
                })

            task_predictions[task_name].append(example)

        # Calculate PER-BRANCH metrics (original behavior)
        accuracy_pb = task_metrics_per_branch['correct_nodes'] / task_metrics_per_branch['total_nodes'] if task_metrics_per_branch['total_nodes'] > 0 else 0
        tp_pb = task_metrics_per_branch['true_positives']
        fp_pb = task_metrics_per_branch['false_positives']
        fn_pb = task_metrics_per_branch['false_negatives']
        precision_pb = tp_pb / (tp_pb + fp_pb) if (tp_pb + fp_pb) > 0 else 0
        recall_pb = tp_pb / (tp_pb + fn_pb) if (tp_pb + fn_pb) > 0 else 0
        f1_pb = 2 * precision_pb * recall_pb / (precision_pb + recall_pb) if (precision_pb + recall_pb) > 0 else 0
        exact_match_rate_pb = task_metrics_per_branch['exact_matches'] / task_metrics_per_branch['total_branches'] if task_metrics_per_branch['total_branches'] > 0 else 0

        # Calculate AGGREGATED metrics (per-original-example)
        accuracy_agg = task_metrics_aggregated['correct_nodes'] / task_metrics_aggregated['total_nodes'] if task_metrics_aggregated['total_nodes'] > 0 else 0
        tp_agg = task_metrics_aggregated['true_positives']
        fp_agg = task_metrics_aggregated['false_positives']
        fn_agg = task_metrics_aggregated['false_negatives']
        precision_agg = tp_agg / (tp_agg + fp_agg) if (tp_agg + fp_agg) > 0 else 0
        recall_agg = tp_agg / (tp_agg + fn_agg) if (tp_agg + fn_agg) > 0 else 0
        f1_agg = 2 * precision_agg * recall_agg / (precision_agg + recall_agg) if (precision_agg + recall_agg) > 0 else 0
        exact_match_rate_agg = task_metrics_aggregated['exact_matches'] / task_metrics_aggregated['total_examples'] if task_metrics_aggregated['total_examples'] > 0 else 0

        statistics = {
            'task': task_name,
            'checkpoint': model_info['checkpoint_path'],
            # Per-branch metrics
            'accuracy_per_branch': accuracy_pb,
            'precision_per_branch': precision_pb,
            'recall_per_branch': recall_pb,
            'f1_per_branch': f1_pb,
            'exact_match_per_branch': exact_match_rate_pb,
            'exact_matches_per_branch': task_metrics_per_branch['exact_matches'],
            'total_branches': task_metrics_per_branch['total_branches'],
            # Aggregated metrics (recommended)
            'accuracy': accuracy_agg,
            'precision': precision_agg,
            'recall': recall_agg,
            'f1': f1_agg,
            'exact_match': exact_match_rate_agg,
            'exact_matches': task_metrics_aggregated['exact_matches'],
            'total_examples': task_metrics_aggregated['total_examples'],
            # Additional info
            'num_branches': len(branches),
            'avg_branches_per_example': len(branches) / task_metrics_aggregated['total_examples'] if task_metrics_aggregated['total_examples'] > 0 else 0,
            'examples_with_no_predictions_agg': task_metrics_aggregated['examples_with_no_predictions'],
            'examples_with_no_predictions_pb': task_metrics_per_branch['examples_with_no_predictions']
        }
        all_statistics.append(statistics)

        # Print metrics (both per-branch and aggregated)
        print(f"\n    === PER-BRANCH METRICS (branches treated independently) ===")
        print(f"    Exact Match: {exact_match_rate_pb:.4f} ({task_metrics_per_branch['exact_matches']}/{task_metrics_per_branch['total_branches']})")
        print(f"    Accuracy:    {accuracy_pb:.4f}")
        print(f"    Precision:   {precision_pb:.4f}")
        print(f"    Recall:      {recall_pb:.4f}")
        print(f"    F1:          {f1_pb:.4f}")
        print(f"\n    === AGGREGATED METRICS (branches combined per example) ===")
        print(f"    Exact Match: {exact_match_rate_agg:.4f} ({task_metrics_aggregated['exact_matches']}/{task_metrics_aggregated['total_examples']})")
        print(f"    Accuracy:    {accuracy_agg:.4f}")
        print(f"    Precision:   {precision_agg:.4f}")
        print(f"    Recall:      {recall_agg:.4f}")
        print(f"    F1:          {f1_agg:.4f}")
        print(f"    Avg branches per example: {len(branches) / task_metrics_aggregated['total_examples']:.2f}" if task_metrics_aggregated['total_examples'] > 0 else "")

    print(f"\nFinal branches after all tasks: {len(branches)}")
    print(f"Total predictions per task:")
    for task_name in TASK_ORDER:
        if task_name in task_predictions:
            print(f"  {task_name}: {len(task_predictions[task_name])} examples")

    return all_statistics, task_predictions


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate all 4 best checkpoints on test sets'
    )
    parser.add_argument('--checkpoint-dir', type=str, default='best_ckpts',
                       help='Directory containing checkpoint files (default: best_ckpts)')
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Output directory for results (default: results)')
    parser.add_argument('--test-iid', type=str, default='data/test_multiple_iid.jsonl',
                       help='IID test set path (default: data/test_multiple_iid.jsonl)')
    parser.add_argument('--test-ood', type=str, default='data/test_multiple_ood.jsonl',
                       help='OOD test set path (default: data/test_multiple_ood.jsonl)')
    parser.add_argument('--board-size', type=int, default=16,
                       help='Board size (default: 16)')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Prediction threshold (default: 0.5)')

    args = parser.parse_args()

    # Set up test data paths
    test_data_paths = {
        'iid': args.test_iid,
        'ood': args.test_ood
    }

    # Evaluate all checkpoints
    statistics_df, predictions = evaluate_all_checkpoints(
        checkpoint_dir=args.checkpoint_dir,
        test_data_paths=test_data_paths,
        board_size=args.board_size,
        threshold=args.threshold
    )

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save statistics to CSV
    stats_path = output_dir / 'test_statistics.csv'
    statistics_df.to_csv(stats_path, index=False)
    print(f"\n{'=' * 80}")
    print(f"Statistics saved to: {stats_path}")

    # Save predictions for each test set
    for test_name, test_preds in predictions.items():
        pred_path = output_dir / f'predictions_{test_name}.pkl'
        with open(pred_path, 'wb') as f:
            pickle.dump(test_preds, f)
        print(f"Predictions for {test_name} saved to: {pred_path}")

        # Print summary
        print(f"\n  {test_name.upper()} predictions structure:")
        for task_name, examples in test_preds.items():
            print(f"    {task_name}: {len(examples)} examples")
            if len(examples) > 0:
                # Calculate total branches and average
                total_branches = sum(ex.get('num_branches', 1) for ex in examples)
                avg_branches = total_branches / len(examples) if len(examples) > 0 else 0
                print(f"      (total {total_branches} branches, avg {avg_branches:.2f} branches per example)")

    print(f"\n{'=' * 80}")
    print("EVALUATION COMPLETE")
    print(f"{'=' * 80}")

    # Print final summary
    print("\nFinal Summary (AGGREGATED METRICS - recommended):")
    print(f"{'Test Set':<10} {'Task':<20} {'EM (agg)':<12} {'EM (branch)':<12} {'Branches':<10}")
    print("-" * 80)
    for _, row in statistics_df.iterrows():
        print(f"{row['test_set']:<10} {row['task']:<20} {row['exact_match']:<12.4f} {row['exact_match_per_branch']:<12.4f} {row.get('avg_branches_per_example', 0):<10.2f}")
    print("=" * 80)
    print("\nNote: 'EM (agg)' is exact match on aggregated branch predictions (recommended).")
    print("      'EM (branch)' is exact match treating each branch independently.")


if __name__ == '__main__':
    main()
