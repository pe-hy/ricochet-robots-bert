"""
Positional encoding strategies for node features.

This module provides different positional encoding methods that can be easily swapped.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional
from abc import ABC, abstractmethod


class PositionalEncoding(ABC):
    """
    Abstract base class for positional encoding strategies.
    """

    @abstractmethod
    def encode(self, x: int, y: int, board_size: int) -> np.ndarray:
        """
        Encode a single (x, y) position.

        Args:
            x: X coordinate
            y: Y coordinate
            board_size: Size of the board

        Returns:
            Encoded position as numpy array
        """
        pass

    @abstractmethod
    def get_encoding_dim(self, board_size: int) -> int:
        """
        Get the dimension of the positional encoding.

        Args:
            board_size: Size of the board

        Returns:
            Encoding dimension
        """
        pass


class OneHotPositionalEncoding(PositionalEncoding):
    """
    One-hot encoding for x and y coordinates separately.

    For a board of size N:
    - x is encoded as one-hot vector of length N
    - y is encoded as one-hot vector of length N
    - Total encoding dimension: 2*N
    """

    def encode(self, x: int, y: int, board_size: int) -> np.ndarray:
        """
        Encode (x, y) as concatenated one-hot vectors.

        Args:
            x: X coordinate
            y: Y coordinate
            board_size: Size of the board

        Returns:
            One-hot encoded position [x_onehot, y_onehot] of shape (2*board_size,)
        """
        x_onehot = np.zeros(board_size, dtype=np.float32)
        x_onehot[x] = 1.0

        y_onehot = np.zeros(board_size, dtype=np.float32)
        y_onehot[y] = 1.0

        return np.concatenate([x_onehot, y_onehot])

    def get_encoding_dim(self, board_size: int) -> int:
        """Return 2*board_size for one-hot encoding"""
        return 2 * board_size


class LearnedPositionalEncoding(PositionalEncoding):
    """
    Learned embedding for (x, y) coordinates.

    Uses a lookup table to learn position embeddings.
    Requires a PyTorch nn.Embedding module.
    """

    def __init__(self, embedding_dim: int = 64):
        """
        Args:
            embedding_dim: Dimension of learned position embeddings
        """
        self.embedding_dim = embedding_dim

    def encode(self, x: int, y: int, board_size: int) -> np.ndarray:
        """
        For learned embeddings, we return the index instead of the encoding.
        The actual embedding is applied by the model.

        Returns:
            Position index as [x, y]
        """
        return np.array([x, y], dtype=np.float32)

    def get_encoding_dim(self, board_size: int) -> int:
        """Return 2 for (x, y) indices - embedding is done by the model"""
        return 2


class SinusoidalPositionalEncoding(PositionalEncoding):
    """
    Sinusoidal positional encoding (similar to Transformer paper).

    Encodes (x, y) using sine and cosine functions of different frequencies.
    """

    def __init__(self, encoding_dim: int = 64):
        """
        Args:
            encoding_dim: Dimension of the encoding (must be even)
        """
        assert encoding_dim % 2 == 0, "encoding_dim must be even"
        self.encoding_dim = encoding_dim

    def encode(self, x: int, y: int, board_size: int) -> np.ndarray:
        """
        Encode (x, y) using sinusoidal functions.

        Args:
            x: X coordinate
            y: Y coordinate
            board_size: Size of the board

        Returns:
            Sinusoidal encoding of shape (encoding_dim,)
        """
        # Normalize coordinates to [0, 1]
        x_norm = x / board_size
        y_norm = y / board_size

        # Create frequency bands
        half_dim = self.encoding_dim // 4  # Split between x and y
        freqs = np.exp(np.arange(half_dim) * -(np.log(10000.0) / half_dim))

        # Encode x
        x_encoding = np.concatenate([
            np.sin(x_norm * freqs),
            np.cos(x_norm * freqs)
        ])

        # Encode y
        y_encoding = np.concatenate([
            np.sin(y_norm * freqs),
            np.cos(y_norm * freqs)
        ])

        return np.concatenate([x_encoding, y_encoding]).astype(np.float32)

    def get_encoding_dim(self, board_size: int) -> int:
        """Return fixed encoding dimension"""
        return self.encoding_dim


class NormalizedCoordinateEncoding(PositionalEncoding):
    """
    Simply use normalized (x, y) coordinates as features.

    Normalizes coordinates to [0, 1] range.
    """

    def encode(self, x: int, y: int, board_size: int) -> np.ndarray:
        """
        Encode (x, y) as normalized coordinates.

        Args:
            x: X coordinate
            y: Y coordinate
            board_size: Size of the board

        Returns:
            Normalized [x, y] of shape (2,)
        """
        x_norm = x / (board_size - 1) if board_size > 1 else 0.0
        y_norm = y / (board_size - 1) if board_size > 1 else 0.0
        return np.array([x_norm, y_norm], dtype=np.float32)

    def get_encoding_dim(self, board_size: int) -> int:
        """Return 2 for (x, y)"""
        return 2


# Factory function to create positional encoding
def create_positional_encoding(encoding_type: str, **kwargs) -> PositionalEncoding:
    """
    Factory function to create positional encoding strategy.

    Args:
        encoding_type: Type of encoding ('onehot', 'learned', 'sinusoidal', 'normalized')
        **kwargs: Additional arguments for the encoding strategy

    Returns:
        PositionalEncoding instance

    Example:
        >>> encoder = create_positional_encoding('onehot')
        >>> pos_encoding = encoder.encode(5, 10, board_size=16)
    """
    encodings = {
        'onehot': OneHotPositionalEncoding,
        'learned': LearnedPositionalEncoding,
        'sinusoidal': SinusoidalPositionalEncoding,
        'normalized': NormalizedCoordinateEncoding,
    }

    if encoding_type not in encodings:
        raise ValueError(
            f"Unknown encoding type: {encoding_type}. "
            f"Available types: {list(encodings.keys())}"
        )

    return encodings[encoding_type](**kwargs)


# Example usage
if __name__ == "__main__":
    # Test different encodings
    board_size = 16
    x, y = 5, 10

    print("Testing positional encodings:")
    print("=" * 60)

    # One-hot encoding
    encoder = create_positional_encoding('onehot')
    encoding = encoder.encode(x, y, board_size)
    print(f"One-hot encoding:")
    print(f"  Dimension: {encoder.get_encoding_dim(board_size)}")
    print(f"  Shape: {encoding.shape}")
    print(f"  First 10 values: {encoding[:10]}")
    print()

    # Sinusoidal encoding
    encoder = create_positional_encoding('sinusoidal', encoding_dim=64)
    encoding = encoder.encode(x, y, board_size)
    print(f"Sinusoidal encoding:")
    print(f"  Dimension: {encoder.get_encoding_dim(board_size)}")
    print(f"  Shape: {encoding.shape}")
    print(f"  First 10 values: {encoding[:10]}")
    print()

    # Normalized encoding
    encoder = create_positional_encoding('normalized')
    encoding = encoder.encode(x, y, board_size)
    print(f"Normalized encoding:")
    print(f"  Dimension: {encoder.get_encoding_dim(board_size)}")
    print(f"  Shape: {encoding.shape}")
    print(f"  Values: {encoding}")
