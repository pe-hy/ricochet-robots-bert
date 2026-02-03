"""
Multi-task data module for Ricochet Robots.

Loads all 4 task labels simultaneously for joint multi-task learning.
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


class MultiTaskDataset(Dataset):
    """
    Multi-task dataset for Ricochet Robots.

    Loads all 4 task labels simultaneously:
    - subgoal_label (index 20)
    - helper_aggregate (index 17)
    - target_pos (index 18)
    - chosen_helper (index 19)

    Features per node:
    - robot_type: 5 dims (one-hot)
    - has_goal: 2 dims (one-hot)
    - walls: 5 dims (one-hot)
    - positional_encoding: variable dims (depends on encoding type)
    Total: 12 + positional_encoding_dim

    For 16x16 board with one-hot encoding: 12 + 32 = 44 features
    """

    TASK_INDICES = {
        'subgoal_label': 20,
        'helper_aggregate': 17,
        'target_pos': 18,
        'chosen_helper': 19
    }

    def __init__(
        self,
        data_path: str,
        board_size: int = 16,
        positional_encoding: str = 'onehot',
        positional_encoding_kwargs: Optional[Dict] = None,
    ):
        """
        Args:
            data_path: Path to JSONL dataset file
            board_size: Size of the board (default 16 for 16x16)
            positional_encoding: Type of positional encoding ('onehot', 'sinusoidal', 'normalized', 'learned')
            positional_encoding_kwargs: Additional kwargs for positional encoding
        """
        self.board_size = board_size
        self.positional_encoding_kwargs = positional_encoding_kwargs or {}

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

    def _process_node(self, node: List) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Process a single node from the dataset.

        Input node format (21 features): [x, y, robot_type(5), has_goal(2), walls(5),
                                          helper1_goal_pos, helper2_goal_pos, helper3_goal_pos,
                                          helper_aggregate_goal_pos, target_pos_goal_pos,
                                          chosen_helper, subgoal_label]

        Returns:
            features: [robot_type(5), has_goal(2), walls(5), positional_encoding(...)]
            targets: dict with 4 task labels
        """
        # Extract coordinates
        x = int(node[0])
        y = int(node[1])

        # Extract base features (no goal features included for multi-task)
        robot_type = node[2:7]      # 5 dims
        has_goal = node[7:9]         # 2 dims
        walls = node[9:14]           # 5 dims

        # Extract all 4 task labels
        targets = {
            task_name: float(node[idx])
            for task_name, idx in self.TASK_INDICES.items()
        }

        # Encode coordinates using the positional encoding strategy
        pos_encoding = self.pos_encoder.encode(x, y, self.board_size)

        # Concatenate all features (no goal features)
        features = np.concatenate([robot_type, has_goal, walls, pos_encoding]).astype(np.float32)

        return features, targets

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
            with open(self.data_path, 'r') as f:
                data = json.load(f)
            return data['examples'][idx]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            Dictionary with:
                - 'features': [num_nodes, feature_dim] - node features
                - 'labels': dict with 4 tasks, each [num_nodes]
                - 'example_id': scalar - example identifier
        """
        example = self._load_example(idx)
        nodes = example['nodes']

        # Process all nodes
        features_list = []
        labels_dict = {task_name: [] for task_name in self.TASK_INDICES.keys()}

        for node in nodes:
            features, targets = self._process_node(node)
            features_list.append(features)
            for task_name, label in targets.items():
                labels_dict[task_name].append(label)

        # Convert to tensors
        features = torch.tensor(np.stack(features_list), dtype=torch.float32)  # [num_nodes, feature_dim]
        labels = {
            task_name: torch.tensor(label_list, dtype=torch.long)
            for task_name, label_list in labels_dict.items()
        }

        return {
            'features': features,
            'labels': labels,
            'example_id': example['example_id']
        }


class MultiTaskDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for multi-task Ricochet Robots learning.
    """

    def __init__(
        self,
        train_path: str,
        board_size: int = 16,
        batch_size: int = 32,
        num_workers: int = 4,
        val_size: int = 256,
        test_size: int = 1024,
        positional_encoding: str = 'onehot',
        positional_encoding_kwargs: Optional[Dict] = None,
    ):
        """
        Args:
            train_path: Path to training data JSONL
            board_size: Board size (default 16)
            batch_size: Batch size for DataLoader
            num_workers: Number of workers for DataLoader
            val_size: Number of examples for validation set
            test_size: Number of examples for test set (0 = no test set)
            positional_encoding: Type of positional encoding ('onehot', 'sinusoidal', 'normalized', 'learned')
            positional_encoding_kwargs: Additional kwargs for positional encoding
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

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        # Filter out model-specific kwargs
        encoder_kwargs = {k: v for k, v in self.positional_encoding_kwargs.items()
                         if k not in ('combine_method',)}

        # Compute feature dimension
        pos_encoder = create_positional_encoding(
            self.positional_encoding,
            **encoder_kwargs
        )
        # Base features: robot_type(5) + has_goal(2) + walls(5) = 12
        # No goal features for multi-task (model learns dependencies)
        self._feature_dim = 12 + pos_encoder.get_encoding_dim(board_size)

    def setup(self, stage: Optional[str] = None):
        """Set up datasets for different stages"""

        if stage == 'fit' or stage is None:
            # Load full dataset
            full_dataset = MultiTaskDataset(
                self.train_path,
                board_size=self.board_size,
                positional_encoding=self.positional_encoding,
                positional_encoding_kwargs=self.positional_encoding_kwargs,
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
