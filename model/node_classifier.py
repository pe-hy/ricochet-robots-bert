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

    For learned positional encodings, the model creates nn.Embedding layers
    that learn position representations during training.
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
        positional_encoding: Optional[str] = None,
        board_size: int = 16,
        pos_encoding_dim: int = 0,
        pos_combine_method: str = "concat",
    ):
        """
        Args:
            feature_dim: Input feature dimension (raw tensor size)
            d_model: Transformer embedding dimension
            nhead: Number of attention heads
            num_layers: Number of transformer encoder layers
            dim_feedforward: Dimension of feedforward network
            dropout: Dropout probability
            activation: Activation function ('relu' or 'gelu')
            positional_encoding: Type of encoding ('learned', 'onehot', etc.)
            board_size: Board size for learned embeddings
            pos_encoding_dim: Target dimension for learned embeddings
            pos_combine_method: How to combine x,y embeddings ('concat', 'additive', 'joint')
        """
        super().__init__()

        self.feature_dim = feature_dim
        self.d_model = d_model
        self.positional_encoding = positional_encoding
        self.board_size = board_size
        self.pos_encoding_dim = pos_encoding_dim
        self.pos_combine_method = pos_combine_method

        # For learned positional encoding, create embedding layers
        self.use_learned_pos = positional_encoding == 'learned' and pos_encoding_dim > 0
        if self.use_learned_pos:
            if pos_combine_method == "joint":
                # Single embedding for each (x,y) cell
                num_positions = board_size * board_size
                self.pos_embedding = nn.Embedding(num_positions, pos_encoding_dim)
                self.x_embedding = None
                self.y_embedding = None
            else:
                # Separate x and y embeddings
                if pos_combine_method == "additive":
                    # Both embeddings have full dimension (will be added)
                    embed_dim_per_coord = pos_encoding_dim
                else:  # concat
                    # Split dimension between x and y (will be concatenated)
                    embed_dim_per_coord = pos_encoding_dim // 2
                self.x_embedding = nn.Embedding(board_size, embed_dim_per_coord)
                self.y_embedding = nn.Embedding(board_size, embed_dim_per_coord)
                self.pos_embedding = None

            # Effective feature dim: original features minus 2 indices plus learned embeddings
            effective_feature_dim = feature_dim - 2 + pos_encoding_dim
        else:
            self.x_embedding = None
            self.y_embedding = None
            self.pos_embedding = None
            effective_feature_dim = feature_dim

        # Input projection: converts features to d_model
        self.input_projection = nn.Linear(effective_feature_dim, d_model)

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
            features: [batch_size, num_nodes, feature_dim] - node features
            attention_mask: [batch_size, num_nodes] - mask for padding (optional)

        Returns:
            logits: [batch_size, num_nodes, 1] - binary classification logits
        """
        # Apply learned positional embeddings if configured
        if self.use_learned_pos:
            # Split features: [other_features, x_idx, y_idx]
            other_features = features[..., :-2]  # [batch, nodes, feature_dim-2]
            pos_indices = features[..., -2:].long()  # [batch, nodes, 2]
            x_idx = pos_indices[..., 0]  # [batch, nodes]
            y_idx = pos_indices[..., 1]  # [batch, nodes]

            # Get position embeddings based on combine method
            if self.pos_combine_method == "joint":
                # Flatten (x, y) to single index
                flat_idx = y_idx * self.board_size + x_idx  # [batch, nodes]
                pos_embed = self.pos_embedding(flat_idx)  # [batch, nodes, pos_encoding_dim]
            elif self.pos_combine_method == "additive":
                # Add x and y embeddings
                x_embed = self.x_embedding(x_idx)  # [batch, nodes, pos_encoding_dim]
                y_embed = self.y_embedding(y_idx)  # [batch, nodes, pos_encoding_dim]
                pos_embed = x_embed + y_embed  # [batch, nodes, pos_encoding_dim]
            else:  # concat
                # Concatenate x and y embeddings
                x_embed = self.x_embedding(x_idx)  # [batch, nodes, pos_encoding_dim/2]
                y_embed = self.y_embedding(y_idx)  # [batch, nodes, pos_encoding_dim/2]
                pos_embed = torch.cat([x_embed, y_embed], dim=-1)  # [batch, nodes, pos_encoding_dim]

            # Combine with other features
            features = torch.cat([other_features, pos_embed], dim=-1)

        # Project input features to d_model
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
        positional_encoding: Optional[str] = None,
        board_size: int = 16,
        pos_encoding_dim: int = 0,
        pos_combine_method: str = "concat",
    ):
        self.feature_dim = feature_dim
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.activation = activation
        self.positional_encoding = positional_encoding
        self.board_size = board_size
        self.pos_encoding_dim = pos_encoding_dim
        self.pos_combine_method = pos_combine_method

    def to_dict(self):
        return {
            "feature_dim": self.feature_dim,
            "d_model": self.d_model,
            "nhead": self.nhead,
            "num_layers": self.num_layers,
            "dim_feedforward": self.dim_feedforward,
            "dropout": self.dropout,
            "activation": self.activation,
            "positional_encoding": self.positional_encoding,
            "board_size": self.board_size,
            "pos_encoding_dim": self.pos_encoding_dim,
            "pos_combine_method": self.pos_combine_method,
        }


def create_model(config: NodeClassifierConfig) -> NodeClassifierTransformer:
    """Create model from config"""
    return NodeClassifierTransformer(**config.to_dict())
