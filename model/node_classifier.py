"""
Transformer-based node classifier for Ricochet Robots subgoal prediction.

Architecture:
1. Linear projection: maps [robot, goal, walls, x_onehot, y_onehot] to d_model
2. Transformer encoder: processes all nodes with self-attention
3. Classification head: binary prediction for each node
"""

import torch
import torch.nn as nn
import math
from typing import Optional


class NodeClassifierTransformer(nn.Module):
    """
    BERT-like Transformer encoder for node classification.

    Architecture:
    - Bidirectional self-attention (all nodes can attend to all other nodes)
    - No causal masking (unlike GPT)
    - Pre-normalization for training stability

    The positional encoding is handled through configurable encoding strategies
    (one-hot, sinusoidal, normalized, or learned) applied to x,y coordinates.
    """

    def __init__(
        self,
        feature_dim: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        """
        Args:
            feature_dim: Input feature dimension (11 + 2*board_size for one-hot coords)
            d_model: Transformer embedding dimension
            nhead: Number of attention heads
            num_layers: Number of transformer encoder layers
            dim_feedforward: Dimension of feedforward network
            dropout: Dropout probability
            activation: Activation function ('relu' or 'gelu')
        """
        super().__init__()

        self.feature_dim = feature_dim
        self.d_model = d_model

        # Input projection: converts features (including one-hot positional encoding) to d_model
        # This effectively combines feature embeddings and positional embeddings
        self.input_projection = nn.Linear(feature_dim, d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,  # Input shape: [batch, seq_len, d_model]
            norm_first=True     # Pre-normalization (more stable training)
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model)
        )

        # Classification head: binary classification for each node
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier/Kaiming initialization"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            features: [batch_size, num_nodes, feature_dim] - node features with one-hot positions
            attention_mask: [batch_size, num_nodes] - mask for padding (optional)

        Returns:
            logits: [batch_size, num_nodes, 1] - binary classification logits
        """
        # Project input features to d_model
        # This combines all features including one-hot positional encodings
        x = self.input_projection(features)  # [batch, num_nodes, d_model]

        # Create attention mask if provided
        # PyTorch expects mask where True/1 = ignore, False/0 = attend
        if attention_mask is not None:
            attention_mask = attention_mask.bool()
            attention_mask = ~attention_mask  # Invert: 0 -> ignore, 1 -> attend

        # Pass through transformer encoder
        x = self.transformer_encoder(
            x,
            src_key_padding_mask=attention_mask
        )  # [batch, num_nodes, d_model]

        # Classification head
        logits = self.classifier(x)  # [batch, num_nodes, 1]

        return logits

    def predict_proba(self, features: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Predict probabilities for each node.

        Args:
            features: [batch_size, num_nodes, feature_dim]
            attention_mask: [batch_size, num_nodes]

        Returns:
            probs: [batch_size, num_nodes] - probabilities in [0, 1]
        """
        logits = self.forward(features, attention_mask)  # [batch, num_nodes, 1]
        probs = torch.sigmoid(logits).squeeze(-1)  # [batch, num_nodes]
        return probs


class NodeClassifierConfig:
    """Configuration for NodeClassifierTransformer"""

    def __init__(
        self,
        feature_dim: int = 43,  # 11 + 2*16 for 16x16 board
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        self.feature_dim = feature_dim
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.activation = activation

    def to_dict(self):
        return {
            "feature_dim": self.feature_dim,
            "d_model": self.d_model,
            "nhead": self.nhead,
            "num_layers": self.num_layers,
            "dim_feedforward": self.dim_feedforward,
            "dropout": self.dropout,
            "activation": self.activation,
        }


def create_model(config: NodeClassifierConfig) -> NodeClassifierTransformer:
    """Create model from config"""
    return NodeClassifierTransformer(**config.to_dict())
