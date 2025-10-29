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

from model.positional_encoding import create_positional_encoding, PositionalEncoding


class RicochetRobotsDataset(Dataset):
    """
    Dataset for Ricochet Robots node classification.

    Features per node:
    - robot_type: 5 dims (one-hot)
    - has_goal: 2 dims (one-hot)
    - walls: 4 dims (one-hot)
    - positional_encoding: variable dims (depends on encoding type)
    Total: 11 + positional_encoding_dim

    For 16x16 board with one-hot encoding: 11 + 32 = 43 features
    """

    def __init__(
        self,
        data_path: str,
        board_size: int = 16,
        positional_encoding: str = 'onehot',
        positional_encoding_kwargs: Optional[Dict] = None
    ):
        """
        Args:
            data_path: Path to JSON dataset file
            board_size: Size of the board (default 16 for 16x16)
            positional_encoding: Type of positional encoding ('onehot', 'sinusoidal', 'normalized', 'learned')
            positional_encoding_kwargs: Additional kwargs for positional encoding
        """
        self.board_size = board_size
        self.positional_encoding_kwargs = positional_encoding_kwargs or {}

        # Create positional encoding strategy
        self.pos_encoder = create_positional_encoding(
            positional_encoding,
            **self.positional_encoding_kwargs
        )

        self.data = self._load_data(data_path)

    def _load_data(self, data_path: str) -> List[Dict]:
        """Load and parse JSON data"""
        with open(data_path, 'r') as f:
            data = json.load(f)
        return data['examples']

    def _process_node(self, node: List) -> Tuple[np.ndarray, int]:
        """
        Process a single node from the dataset.

        Input node format: [x, y, robot_type(5), has_goal(2), walls(4), label]

        Returns:
            features: [robot_type(5), has_goal(2), walls(4), positional_encoding(...)]
            label: binary subgoal label
        """
        # Extract components
        x = int(node[0])
        y = int(node[1])
        robot_type = node[2:7]      # already one-hot
        has_goal = node[7:9]         # already one-hot
        walls = node[9:13]           # already one-hot
        label = int(node[13])

        # Encode coordinates using the positional encoding strategy
        pos_encoding = self.pos_encoder.encode(x, y, self.board_size)

        # Concatenate all features
        features = np.concatenate([
            robot_type,
            has_goal,
            walls,
            pos_encoding
        ]).astype(np.float32)

        return features, label

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            Dictionary with:
                - 'features': [num_nodes, feature_dim] - node features
                - 'labels': [num_nodes] - binary labels
                - 'example_id': scalar - example identifier
        """
        example = self.data[idx]
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
        val_path: Optional[str] = None,
        test_path: Optional[str] = None,
        board_size: int = 16,
        batch_size: int = 32,
        num_workers: int = 4,
        train_val_split: float = 0.8,
        positional_encoding: str = 'onehot',
        positional_encoding_kwargs: Optional[Dict] = None
    ):
        """
        Args:
            train_path: Path to training data JSON
            val_path: Path to validation data JSON (if None, will split from train)
            test_path: Path to test data JSON
            board_size: Board size (default 16)
            batch_size: Batch size for DataLoader
            num_workers: Number of workers for DataLoader
            train_val_split: Fraction of train data to use for training (rest for val)
            positional_encoding: Type of positional encoding ('onehot', 'sinusoidal', 'normalized', 'learned')
            positional_encoding_kwargs: Additional kwargs for positional encoding
        """
        super().__init__()
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.board_size = board_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_val_split = train_val_split
        self.positional_encoding = positional_encoding
        self.positional_encoding_kwargs = positional_encoding_kwargs or {}

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        # Compute feature dimension
        pos_encoder = create_positional_encoding(
            self.positional_encoding,
            **self.positional_encoding_kwargs
        )
        self._feature_dim = 11 + pos_encoder.get_encoding_dim(board_size)

    def setup(self, stage: Optional[str] = None):
        """Set up datasets for different stages"""

        if stage == 'fit' or stage is None:
            # Load training data
            full_train_dataset = RicochetRobotsDataset(
                self.train_path,
                board_size=self.board_size,
                positional_encoding=self.positional_encoding,
                positional_encoding_kwargs=self.positional_encoding_kwargs
            )

            # Split into train/val if no separate val set
            if self.val_path is None:
                train_size = int(self.train_val_split * len(full_train_dataset))
                val_size = len(full_train_dataset) - train_size
                self.train_dataset, self.val_dataset = torch.utils.data.random_split(
                    full_train_dataset,
                    [train_size, val_size],
                    generator=torch.Generator().manual_seed(42)
                )
            else:
                self.train_dataset = full_train_dataset
                self.val_dataset = RicochetRobotsDataset(
                    self.val_path,
                    board_size=self.board_size,
                    positional_encoding=self.positional_encoding,
                    positional_encoding_kwargs=self.positional_encoding_kwargs
                )

        if stage == 'test' or stage is None:
            if self.test_path is not None:
                self.test_dataset = RicochetRobotsDataset(
                    self.test_path,
                    board_size=self.board_size,
                    positional_encoding=self.positional_encoding,
                    positional_encoding_kwargs=self.positional_encoding_kwargs
                )

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
