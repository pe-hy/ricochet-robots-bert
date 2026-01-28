"""
Data module for Ricochet Robots node classification.

Loads JSON data and applies configurable positional encoding for coordinates.
"""

import json
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import numpy as np
import linecache

from utils.positional_encoding import create_positional_encoding, PositionalEncoding


class RicochetRobotsDataset(Dataset):
    """
    Dataset for Ricochet Robots node classification.

    Features per node:
    - robot_type: 5 dims (one-hot)
    - has_goal: 2 dims (one-hot)
    - walls: 5 dims (one-hot)
    - positional_encoding: variable dims (depends on encoding type)
    Total: 12 + positional_encoding_dim

    For 16x16 board with one-hot encoding: 12 + 32 = 44 features
    """

    def __init__(
        self,
        data_path: str,
        board_size: int = 16,
        positional_encoding: str = 'onehot',
        positional_encoding_kwargs: Optional[Dict] = None,
        task_config: Optional[Dict] = None
    ):
        """
        Args:
            data_path: Path to JSON dataset file
            board_size: Size of the board (default 16 for 16x16)
            positional_encoding: Type of positional encoding ('onehot', 'sinusoidal', 'normalized', 'learned')
            positional_encoding_kwargs: Additional kwargs for positional encoding
            task_config: Task configuration (target_index, include_goal_features)
        """
        self.board_size = board_size
        self.positional_encoding_kwargs = positional_encoding_kwargs or {}
        self.task_config = task_config or {'target_index': 19, 'include_goal_features': []}

        # Filter out model-specific kwargs (combine_method is used by model, not encoder)
        encoder_kwargs = {k: v for k, v in self.positional_encoding_kwargs.items()
                         if k not in ('combine_method',)}

        # Create positional encoding strategy
        self.pos_encoder = create_positional_encoding(
            positional_encoding,
            **encoder_kwargs
        )

        self.data_path = data_path
        self._length = self._count_examples(data_path)

    def _count_examples(self, data_path: str) -> int:
        """Count number of examples without loading into memory"""
        if data_path.endswith('.jsonl'):
            # JSONL format - one example per line (first line is metadata)
            with open(data_path, 'r') as f:
                return sum(1 for _ in f) - 1  # Subtract 1 for metadata line
        else:
            # Original JSON format - load only to count
            with open(data_path, 'r') as f:
                data = json.load(f)
            return len(data['examples'])

    def _process_node(self, node: List) -> Tuple[np.ndarray, float]:
        """
        Process a single node from the dataset.

        Input node format (15 features): [x, y, robot_type(5), has_goal(2), walls(5), label]
        Input node format (20 features): [x, y, robot_type(5), has_goal(2), walls(5),
                                          helper1_goal_pos, helper2_goal_pos, helper3_goal_pos,
                                          helper_aggregate_goal_pos, target_goal_pos, label]
        Input node format (21 features): [x, y, robot_type(5), has_goal(2), walls(5),
                                          helper1_goal_pos, helper2_goal_pos, helper3_goal_pos,
                                          helper_aggregate_goal_pos, target_goal_pos,
                                          chosen_helper, subgoal_label]

        Returns:
            features: [robot_type(5), has_goal(2), walls(5), goal_features(?), positional_encoding(...)]
            target: target value (binary or continuous)
        """
        # Extract coordinates
        x = int(node[0])
        y = int(node[1])

        # Extract base features
        robot_type = node[2:7]      # 5 dims
        has_goal = node[7:9]         # 2 dims
        walls = node[9:14]           # 5 dims

        # Handle goal position features based on task config
        goal_features = []
        if len(node) >= 20:  # New format with goal features (20 or 21 features)
            # Include specified goal features
            include_indices = self.task_config.get('include_goal_features', [])
            for idx in include_indices:
                goal_features.append(node[idx])

        # Extract target based on task config
        target_index = self.task_config.get('target_index', -1)
        if len(node) >= 20:  # New format (20 or 21 features)
            target = float(node[target_index])
        else:
            # Old format (15 features) - only has subgoal_label at index 14
            target = float(node[14])

        # Encode coordinates using the positional encoding strategy
        pos_encoding = self.pos_encoder.encode(x, y, self.board_size)

        # Concatenate all features
        feature_parts = [robot_type, has_goal, walls]
        if len(goal_features) > 0:
            feature_parts.append(goal_features)
        feature_parts.append(pos_encoding)

        features = np.concatenate(feature_parts).astype(np.float32)

        return features, target

    def __len__(self) -> int:
        return self._length

    def _load_example(self, idx: int) -> Dict:
        """Load a single example lazily"""
        if self.data_path.endswith('.jsonl'):
            # JSONL format - use linecache for efficient line reading
            # Line 1 is metadata, examples start at line 2
            line = linecache.getline(self.data_path, idx + 2)
            return json.loads(line)
        else:
            # Original JSON format - fallback to loading full file
            # This is slower but maintains compatibility
            with open(self.data_path, 'r') as f:
                data = json.load(f)
            return data['examples'][idx]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            Dictionary with:
                - 'features': [num_nodes, feature_dim] - node features
                - 'labels': [num_nodes] - binary labels
                - 'example_id': scalar - example identifier
        """
        example = self._load_example(idx)
        nodes = example['nodes']

        # Process all nodes
        features_list = []
        labels_list = []

        for node in nodes:
            features, label = self._process_node(node)
            features_list.append(features)
            labels_list.append(label)

        # Convert to tensors
        features = torch.tensor(np.stack(features_list), dtype=torch.float32)  # [num_nodes, feature_dim]
        labels = torch.tensor(labels_list, dtype=torch.long)  # [num_nodes]

        return {
            'features': features,
            'labels': labels,
            'example_id': example['example_id']
        }


class RicochetRobotsDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for Ricochet Robots.
    """

    def __init__(
        self,
        train_path: str,
        board_size: int = 16,
        batch_size: int = 32,
        num_workers: int = 4,
        val_size: int = 16,
        test_size: int = 0,
        positional_encoding: str = 'onehot',
        positional_encoding_kwargs: Optional[Dict] = None,
        task_config: Optional[Dict] = None
    ):
        """
        Args:
            train_path: Path to training data JSON
            board_size: Board size (default 16)
            batch_size: Batch size for DataLoader
            num_workers: Number of workers for DataLoader
            val_size: Number of examples for validation set
            test_size: Number of examples for test set (0 = no test set)
            positional_encoding: Type of positional encoding ('onehot', 'sinusoidal', 'normalized', 'learned')
            positional_encoding_kwargs: Additional kwargs for positional encoding
            task_config: Task configuration (target_index, include_goal_features)
        """
        super().__init__()
        self.train_path = train_path
        self.board_size = board_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_size = val_size
        self.test_size = test_size
        self.positional_encoding = positional_encoding
        self.positional_encoding_kwargs = positional_encoding_kwargs or {}
        self.task_config = task_config or {'target_index': 19, 'include_goal_features': []}

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        # Filter out model-specific kwargs (combine_method is used by model, not encoder)
        encoder_kwargs = {k: v for k, v in self.positional_encoding_kwargs.items()
                         if k not in ('combine_method',)}

        # Compute feature dimension
        pos_encoder = create_positional_encoding(
            self.positional_encoding,
            **encoder_kwargs
        )
        # Base features: robot_type(5) + has_goal(2) + walls(5) = 12
        # + goal features (configurable) + positional encoding
        num_goal_features = len(self.task_config.get('include_goal_features', []))
        self._feature_dim = 12 + num_goal_features + pos_encoder.get_encoding_dim(board_size)

    def setup(self, stage: Optional[str] = None):
        """Set up datasets for different stages"""

        if stage == 'fit' or stage is None:
            # Load full dataset
            full_dataset = RicochetRobotsDataset(
                self.train_path,
                board_size=self.board_size,
                positional_encoding=self.positional_encoding,
                positional_encoding_kwargs=self.positional_encoding_kwargs,
                task_config=self.task_config
            )

            total_size = len(full_dataset)

            # Calculate split sizes
            test_size = self.test_size if self.test_size > 0 else 0
            val_size = self.val_size
            train_size = total_size - val_size - test_size

            if train_size <= 0:
                raise ValueError(
                    f"Not enough data: total={total_size}, val={val_size}, test={test_size}. "
                    f"Need at least {val_size + test_size + 1} examples."
                )

            # Split dataset
            if test_size > 0:
                self.train_dataset, self.val_dataset, self.test_dataset = torch.utils.data.random_split(
                    full_dataset,
                    [train_size, val_size, test_size],
                    generator=torch.Generator().manual_seed(42)
                )
            else:
                self.train_dataset, self.val_dataset = torch.utils.data.random_split(
                    full_dataset,
                    [train_size, val_size],
                    generator=torch.Generator().manual_seed(42)
                )
                self.test_dataset = None

        if stage == 'test' and self.test_dataset is None:
            # If no test set was created, skip test stage
            pass

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            return None
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

    @property
    def feature_dim(self) -> int:
        """Return the feature dimension"""
        return self._feature_dim
