"""
Sequential Multi-Task Transformer Models for Ricochet Robots.

This module implements 4 transformer architectures for sequential multi-task learning.
Instead of training 4 separate models (one per task), these architectures learn all 4
tasks jointly by processing label tokens sequentially with causal attention.

Task sequence: subgoal_label → helper_aggregate → target_pos → chosen_helper

Architecture pattern:
- Nodes (0-255): Full bidirectional attention among themselves
- Label tokens (256+): Causal attention (can only attend backward)
- Sequence: [node1...node256, LABEL1, LABEL2, LABEL3, LABEL4]

Classes:
    - BaseMultiTaskTransformer: Abstract base class with shared utilities
    - BasicSequentialMultiTask: Simple sequential with causal masking
    - CachedSequentialMultiTask: Optimized with cached node embeddings
    - ComputationVectorMultiTask: With working memory vectors
    - CachedComputationVectorMultiTask: Cached + computation vectors
    - MultiTaskConfig: Configuration dataclass
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict
from abc import ABC, abstractmethod


class BaseMultiTaskTransformer(nn.Module, ABC):
    """
    Abstract base class for multi-task transformer architectures.

    Provides shared utilities:
    - Hybrid attention mask generation (bidirectional + causal)
    - Label-to-node prediction projection
    - Weight initialization
    - Probability prediction interface
    """

    TASK_NAMES = ['subgoal_label', 'helper_aggregate', 'target_pos', 'chosen_helper']

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
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.activation = activation
        self.positional_encoding = positional_encoding
        self.board_size = board_size
        self.pos_encoding_dim = pos_encoding_dim
        self.pos_combine_method = pos_combine_method

        # For learned positional encoding, create embedding layers (same as node_classifier.py)
        self.use_learned_pos = positional_encoding == 'learned' and pos_encoding_dim > 0
        if self.use_learned_pos:
            if pos_combine_method == "joint":
                num_positions = board_size * board_size
                self.pos_embedding = nn.Embedding(num_positions, pos_encoding_dim)
                self.x_embedding = None
                self.y_embedding = None
            else:
                if pos_combine_method == "additive":
                    embed_dim_per_coord = pos_encoding_dim
                else:  # concat
                    embed_dim_per_coord = pos_encoding_dim // 2
                self.x_embedding = nn.Embedding(board_size, embed_dim_per_coord)
                self.y_embedding = nn.Embedding(board_size, embed_dim_per_coord)
                self.pos_embedding = None

            # FIXED: Input projection expects feature_dim (data module already provides correct dim)
            effective_feature_dim = feature_dim
        else:
            self.x_embedding = None
            self.y_embedding = None
            self.pos_embedding = None
            effective_feature_dim = feature_dim

        # Input projection: converts features to d_model
        self.input_projection = nn.Linear(effective_feature_dim, d_model)

        # Learnable label embeddings (4 tasks)
        self.label_embeds = nn.Parameter(torch.randn(4, d_model) * 0.02)

        # Task-specific classification heads (same architecture as node_classifier.py)
        for i in range(4):
            classifier = nn.Sequential(
                nn.Linear(d_model * 2, d_model),  # Combined: label + node
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, 1)
            )
            setattr(self, f'classifier_task{i+1}', classifier)

    def _apply_learned_positional_encoding(self, features: torch.Tensor) -> torch.Tensor:
        """Apply learned positional encoding if configured (same as node_classifier.py)"""
        if not self.use_learned_pos:
            return features

        # Split features: [other_features, x_idx, y_idx]
        other_features = features[..., :-2]
        pos_indices = features[..., -2:].long()
        x_idx = pos_indices[..., 0]
        y_idx = pos_indices[..., 1]

        # Get position embeddings based on combine method
        if self.pos_combine_method == "joint":
            flat_idx = y_idx * self.board_size + x_idx
            pos_embed = self.pos_embedding(flat_idx)
        elif self.pos_combine_method == "additive":
            x_embed = self.x_embedding(x_idx)
            y_embed = self.y_embedding(y_idx)
            pos_embed = x_embed + y_embed
        else:  # concat
            x_embed = self.x_embedding(x_idx)
            y_embed = self.y_embedding(y_idx)
            pos_embed = torch.cat([x_embed, y_embed], dim=-1)

        # Combine with other features
        return torch.cat([other_features, pos_embed], dim=-1)

    def _create_hybrid_attention_mask(
        self,
        seq_len: int,
        num_nodes: int = 256,
        device: torch.device = None
    ) -> torch.Tensor:
        """
        Create hybrid bidirectional + causal attention mask.

        Args:
            seq_len: Total sequence length (nodes + labels + comp vectors)
            num_nodes: Number of node tokens (default: 256)
            device: Device for tensor creation

        Returns:
            mask: [seq_len, seq_len] boolean mask where True = cannot attend

        Mask structure:
        - [0:num_nodes, 0:num_nodes]: All False (nodes attend bidirectionally)
        - [num_nodes:, 0:num_nodes]: All False (labels/comp can attend to all nodes)
        - [num_nodes:, num_nodes:]: Upper triangular True (causal for labels/comp)
        """
        mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)

        # Nodes (0:num_nodes) already bidirectional (all False)

        # Labels and computation vectors (num_nodes:seq_len) have causal masking
        for i in range(num_nodes, seq_len):
            # Cannot attend to future positions
            mask[i, i+1:] = True

        return mask

    def _label_to_node_predictions(
        self,
        label_embed: torch.Tensor,
        node_embeds: torch.Tensor,
        classifier: nn.Module
    ) -> torch.Tensor:
        """
        Project label token embedding to per-node predictions.

        Uses cross-attention mechanism: label attends to each node, then
        combines label context with each node embedding for classification.

        Args:
            label_embed: [batch, d_model] - single label token embedding
            node_embeds: [batch, num_nodes, d_model] - all node embeddings
            classifier: nn.Module - task-specific classification head

        Returns:
            predictions: [batch, num_nodes, 1] - per-node binary predictions
        """
        batch_size = node_embeds.shape[0]
        num_nodes = node_embeds.shape[1]

        # Compute attention: label attends to each node
        # Q: label, K/V: nodes
        attn_scores = torch.matmul(
            label_embed.unsqueeze(1),  # [batch, 1, d_model]
            node_embeds.transpose(-2, -1)  # [batch, d_model, num_nodes]
        )  # [batch, 1, num_nodes]

        attn_weights = F.softmax(attn_scores / math.sqrt(self.d_model), dim=-1)

        # Weighted combination of node embeddings
        context = torch.matmul(attn_weights, node_embeds)  # [batch, 1, d_model]

        # Broadcast label context to all nodes
        label_broadcast = context.expand(-1, num_nodes, -1)  # [batch, num_nodes, d_model]

        # Combine: concatenate label context with each node embedding
        combined = torch.cat([label_broadcast, node_embeds], dim=-1)  # [batch, num_nodes, d_model*2]

        # Classify each node
        predictions = classifier(combined)  # [batch, num_nodes, 1]

        return predictions

    def _init_weights(self):
        """Initialize weights using Xavier/Kaiming initialization"""
        for name, p in self.named_parameters():
            # Skip custom-initialized embeddings
            if 'label_embeds' in name or 'comp_embeds' in name:
                continue

            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    @abstractmethod
    def forward(
        self,
        features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[Dict[str, torch.Tensor]] = None,
        use_teacher_forcing: bool = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass (to be implemented by subclasses).

        Args:
            features: [batch_size, num_nodes, feature_dim] - node features
            attention_mask: [batch_size, num_nodes] - padding mask (optional)
            labels: Ground truth labels (used by teacher forcing architectures)
            use_teacher_forcing: Whether to use teacher forcing (used by teacher forcing architectures)

        Returns:
            Dict with 4 task predictions, each [batch_size, num_nodes, 1]:
            {
                'subgoal_label': [batch, num_nodes, 1],
                'helper_aggregate': [batch, num_nodes, 1],
                'target_pos': [batch, num_nodes, 1],
                'chosen_helper': [batch, num_nodes, 1]
            }
        """
        pass

    def predict_proba(
        self,
        features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Predict probabilities for all tasks.

        Args:
            features: [batch_size, num_nodes, feature_dim]
            attention_mask: [batch_size, num_nodes]

        Returns:
            Dict with 4 task probabilities, each [batch_size, num_nodes]:
            {
                'subgoal_label': [batch, num_nodes],
                'helper_aggregate': [batch, num_nodes],
                'target_pos': [batch, num_nodes],
                'chosen_helper': [batch, num_nodes]
            }
        """
        logits_dict = self.forward(features, attention_mask)
        probs_dict = {}
        for task_name, logits in logits_dict.items():
            probs_dict[task_name] = F.softmax(logits.squeeze(-1), dim=-1)
        return probs_dict


class BasicSequentialMultiTask(BaseMultiTaskTransformer):
    """
    Basic sequential multi-task transformer with causal masking.

    Sequence: [node1...node256, LABEL1, LABEL2, LABEL3, LABEL4]
    Total sequence length: 260

    Attention pattern:
    - Nodes: Full bidirectional (can attend to all 256 nodes)
    - LABEL1: Causal (attends to all 256 nodes)
    - LABEL2: Causal (attends to 256 nodes + LABEL1)
    - LABEL3: Causal (attends to 256 nodes + LABEL1 + LABEL2)
    - LABEL4: Causal (attends to 256 nodes + LABEL1 + LABEL2 + LABEL3)

    Processes all 260 tokens through transformer in one pass.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            activation=self.activation,
            batch_first=True,
            norm_first=True
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.num_layers,
            norm=nn.LayerNorm(self.d_model)
        )

        # Initialize weights
        self._init_weights()

    def forward(
        self,
        features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[Dict[str, torch.Tensor]] = None,
        use_teacher_forcing: bool = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            features: [batch_size, num_nodes, feature_dim] - node features
            attention_mask: [batch_size, num_nodes] - padding mask (optional)

        Returns:
            Dict with 4 task predictions, each [batch_size, num_nodes, 1]
        """
        batch_size, num_nodes = features.shape[:2]
        device = features.device

        # Apply learned positional encoding if configured
        features = self._apply_learned_positional_encoding(features)

        # 1. Project node features
        node_embeds = self.input_projection(features)  # [batch, num_nodes, d_model]

        # 2. Expand label embeddings for batch
        label_embeds = self.label_embeds.unsqueeze(0).expand(batch_size, -1, -1)  # [batch, 4, d_model]

        # 3. Concatenate sequence: nodes + labels
        sequence = torch.cat([node_embeds, label_embeds], dim=1)  # [batch, num_nodes+4, d_model]

        # 4. Create hybrid attention mask
        seq_len = num_nodes + 4
        attn_mask = self._create_hybrid_attention_mask(seq_len, num_nodes, device)

        # 5. Pass through transformer
        encoded = self.transformer_encoder(sequence, mask=attn_mask)  # [batch, num_nodes+4, d_model]

        # 6. Extract embeddings
        node_embeds_final = encoded[:, :num_nodes, :]  # [batch, num_nodes, d_model]
        label_embeds_final = encoded[:, num_nodes:, :]  # [batch, 4, d_model]

        # 7. Project each label to node predictions
        outputs = {}
        for i, task_name in enumerate(self.TASK_NAMES):
            label_embed = label_embeds_final[:, i, :]  # [batch, d_model]
            classifier = getattr(self, f'classifier_task{i+1}')
            preds = self._label_to_node_predictions(
                label_embed, node_embeds_final, classifier
            )
            outputs[task_name] = preds  # [batch, num_nodes, 1]

        return outputs


class CachedSequentialMultiTask(BaseMultiTaskTransformer):
    """
    Optimized sequential multi-task transformer with cached node embeddings.

    Optimization over BasicSequentialMultiTask:
    - After processing nodes + LABEL1, cache node embeddings
    - For LABEL2/3/4: only process new label with cached nodes
    - ~40% faster than BasicSequentialMultiTask

    Forward pass:
    1. Process [nodes + LABEL1] through transformer → cache node embeddings
    2. For LABEL2: Only process LABEL2 with cached nodes + LABEL1
    3. For LABEL3: Only process LABEL3 with cached nodes + LABEL1 + LABEL2
    4. For LABEL4: Only process LABEL4 with cached nodes + LABEL1 + LABEL2 + LABEL3

    Note: Requires manual layer-by-layer processing (can't use nn.TransformerEncoder)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Create transformer encoder layers manually (for layer-by-layer processing)
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=self.d_model,
                nhead=self.nhead,
                dim_feedforward=self.dim_feedforward,
                dropout=self.dropout,
                activation=self.activation,
                batch_first=True,
                norm_first=True
            )
            for _ in range(self.num_layers)
        ])

        self.final_norm = nn.LayerNorm(self.d_model)

        # Initialize weights
        self._init_weights()

    def forward(
        self,
        features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[Dict[str, torch.Tensor]] = None,
        use_teacher_forcing: bool = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with cached node embeddings.

        Cache is updated after each label is processed, so node embeddings
        accumulate information from all previous labels.

        Args:
            features: [batch_size, num_nodes, feature_dim] - node features
            attention_mask: [batch_size, num_nodes] - padding mask (optional)

        Returns:
            Dict with 4 task predictions, each [batch_size, num_nodes, 1]
        """
        batch_size, num_nodes = features.shape[:2]
        device = features.device

        # Apply learned positional encoding if configured
        features = self._apply_learned_positional_encoding(features)

        # 1. Project nodes
        node_embeds = self.input_projection(features)  # [batch, num_nodes, d_model]

        outputs = {}
        cached_sequence = node_embeds  # Initialize cache with projected nodes

        # Process each label sequentially, updating cache after each
        for task_idx, task_name in enumerate(self.TASK_NAMES):
            # Get label embedding for this task
            label_embed = self.label_embeds[task_idx:task_idx+1].unsqueeze(0).expand(batch_size, 1, -1)

            # FIXED: Append new label to entire cached sequence
            seq = torch.cat([cached_sequence, label_embed], dim=1)  # [batch, num_nodes+task_idx+1, d_model]
            seq_len = seq.shape[1]
            mask = self._create_hybrid_attention_mask(seq_len, num_nodes, device)

            # Pass through transformer layers
            for layer in self.transformer_layers:
                seq = layer(seq, src_mask=mask)

            seq = self.final_norm(seq)

            # FIXED: Update cache to include EVERYTHING (nodes + all processed labels)
            cached_sequence = seq

            # Extract label at its correct position
            label_final = seq[:, num_nodes + task_idx, :]

            # Also keep updated nodes for prediction
            cached_nodes = seq[:, :num_nodes, :]

            # Project to node predictions
            classifier = getattr(self, f'classifier_task{task_idx+1}')
            preds = self._label_to_node_predictions(label_final, cached_nodes, classifier)
            outputs[task_name] = preds

        return outputs


class ComputationVectorMultiTask(BaseMultiTaskTransformer):
    """
    Sequential multi-task transformer with computation vectors (working memory).

    Adds N computation vectors before each label token to give the transformer
    more "working memory" for complex reasoning.

    Sequence (N=3): [nodes(256), COMP1-3, LABEL1, COMP4-6, LABEL2, COMP7-9, LABEL3, COMP10-12, LABEL4]
    Total sequence length: 256 + 4*(3+1) = 272

    Attention pattern:
    - Nodes: Full bidirectional
    - Computation vectors: Follow same causal rules as their associated labels
    - COMP1-3 (before LABEL1): Attend to all nodes
    - COMP4-6 (before LABEL2): Attend to nodes + LABEL1 + COMP1-3
    - etc.
    """

    def __init__(self, num_comp_vectors: int = 3, **kwargs):
        self.num_comp_vectors = num_comp_vectors
        super().__init__(**kwargs)

        # Computation vector embeddings (num_comp_vectors per task)
        total_comp_vectors = 4 * num_comp_vectors
        self.comp_embeds = nn.Parameter(torch.randn(total_comp_vectors, self.d_model) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            activation=self.activation,
            batch_first=True,
            norm_first=True
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.num_layers,
            norm=nn.LayerNorm(self.d_model)
        )

        # Initialize weights
        self._init_weights()

    def forward(
        self,
        features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[Dict[str, torch.Tensor]] = None,
        use_teacher_forcing: bool = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with computation vectors.

        Args:
            features: [batch_size, num_nodes, feature_dim] - node features
            attention_mask: [batch_size, num_nodes] - padding mask (optional)

        Returns:
            Dict with 4 task predictions, each [batch_size, num_nodes, 1]
        """
        batch_size, num_nodes = features.shape[:2]
        device = features.device

        # Apply learned positional encoding if configured
        features = self._apply_learned_positional_encoding(features)

        # 1. Project node features
        node_embeds = self.input_projection(features)  # [batch, num_nodes, d_model]

        # 2. Build sequence: nodes + (comp_vectors + label) * 4
        sequence_parts = [node_embeds]

        for task_idx in range(4):
            # Add computation vectors for this task
            comp_start = task_idx * self.num_comp_vectors
            comp_end = comp_start + self.num_comp_vectors
            comp_vecs = self.comp_embeds[comp_start:comp_end].unsqueeze(0).expand(batch_size, -1, -1)
            sequence_parts.append(comp_vecs)

            # Add label for this task
            label_vec = self.label_embeds[task_idx:task_idx+1].unsqueeze(0).expand(batch_size, 1, -1)
            sequence_parts.append(label_vec)

        # 3. Concatenate sequence
        sequence = torch.cat(sequence_parts, dim=1)  # [batch, num_nodes + 4*(num_comp+1), d_model]

        # 4. Create hybrid attention mask
        seq_len = sequence.shape[1]
        attn_mask = self._create_hybrid_attention_mask(seq_len, num_nodes, device)

        # 5. Pass through transformer
        encoded = self.transformer_encoder(sequence, mask=attn_mask)

        # 6. Extract node embeddings and label embeddings
        node_embeds_final = encoded[:, :num_nodes, :]  # [batch, num_nodes, d_model]

        # Extract label embeddings (they're at positions: num_nodes + num_comp, num_nodes + num_comp*2 + 1, ...)
        label_positions = []
        for task_idx in range(4):
            label_pos = num_nodes + (task_idx + 1) * self.num_comp_vectors + task_idx
            label_positions.append(label_pos)

        # 7. Project each label to node predictions
        outputs = {}
        for i, task_name in enumerate(self.TASK_NAMES):
            label_embed = encoded[:, label_positions[i], :]  # [batch, d_model]
            classifier = getattr(self, f'classifier_task{i+1}')
            preds = self._label_to_node_predictions(
                label_embed, node_embeds_final, classifier
            )
            outputs[task_name] = preds

        return outputs


class CachedComputationVectorMultiTask(BaseMultiTaskTransformer):
    """
    Optimized sequential multi-task transformer with both caching and computation vectors.

    Combines benefits of:
    - CachedSequentialMultiTask: Fast (cached node embeddings)
    - ComputationVectorMultiTask: High capacity (working memory)

    Sequence per task (N=3 comp vectors):
    - Task 1: [nodes(256), COMP1-3, LABEL1]
    - Task 2: [cached_nodes, LABEL1_embed, COMP4-6, LABEL2]
    - Task 3: [cached_nodes, LABEL1_embed, LABEL2_embed, COMP7-9, LABEL3]
    - Task 4: [cached_nodes, LABEL1_embed, LABEL2_embed, LABEL3_embed, COMP10-12, LABEL4]
    """

    def __init__(self, num_comp_vectors: int = 3, **kwargs):
        self.num_comp_vectors = num_comp_vectors
        super().__init__(**kwargs)

        # Computation vector embeddings
        total_comp_vectors = 4 * num_comp_vectors
        self.comp_embeds = nn.Parameter(torch.randn(total_comp_vectors, self.d_model) * 0.02)

        # Create transformer encoder layers manually
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=self.d_model,
                nhead=self.nhead,
                dim_feedforward=self.dim_feedforward,
                dropout=self.dropout,
                activation=self.activation,
                batch_first=True,
                norm_first=True
            )
            for _ in range(self.num_layers)
        ])

        self.final_norm = nn.LayerNorm(self.d_model)

        # Initialize weights
        self._init_weights()

    def forward(
        self,
        features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[Dict[str, torch.Tensor]] = None,
        use_teacher_forcing: bool = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with cached node embeddings and computation vectors.

        Cache is updated after each label is processed, so node embeddings
        accumulate information from all previous labels.

        Args:
            features: [batch_size, num_nodes, feature_dim] - node features
            attention_mask: [batch_size, num_nodes] - padding mask (optional)

        Returns:
            Dict with 4 task predictions, each [batch_size, num_nodes, 1]
        """
        batch_size, num_nodes = features.shape[:2]
        device = features.device

        # Apply learned positional encoding if configured
        features = self._apply_learned_positional_encoding(features)

        # 1. Project nodes
        node_embeds = self.input_projection(features)  # [batch, num_nodes, d_model]

        outputs = {}
        cached_sequence = node_embeds  # Initialize cache with projected nodes

        # Process each label sequentially, updating cache after each
        for task_idx, task_name in enumerate(self.TASK_NAMES):
            # Get computation vectors for this task
            comp_start = task_idx * self.num_comp_vectors
            comp_end = comp_start + self.num_comp_vectors
            comp_vecs = self.comp_embeds[comp_start:comp_end].unsqueeze(0).expand(batch_size, -1, -1)

            # Get label embedding for this task
            label_embed = self.label_embeds[task_idx:task_idx+1].unsqueeze(0).expand(batch_size, 1, -1)

            # FIXED: Append comp vectors and label to entire cached sequence
            seq = torch.cat([cached_sequence, comp_vecs, label_embed], dim=1)
            seq_len = seq.shape[1]
            mask = self._create_hybrid_attention_mask(seq_len, num_nodes, device)

            # Pass through transformer
            for layer in self.transformer_layers:
                seq = layer(seq, src_mask=mask)

            seq = self.final_norm(seq)

            # FIXED: Update cache to include EVERYTHING (nodes + all comp vectors + all labels)
            cached_sequence = seq

            # Extract label embedding (always at the end)
            label_final = seq[:, -1, :]

            # Keep updated nodes for prediction
            cached_nodes = seq[:, :num_nodes, :]

            # Project to node predictions
            classifier = getattr(self, f'classifier_task{task_idx+1}')
            preds = self._label_to_node_predictions(label_final, cached_nodes, classifier)
            outputs[task_name] = preds

        return outputs


class TeacherForcingComputationVectorMultiTask(BaseMultiTaskTransformer):
    """
    Sequential multi-task transformer with teacher forcing and computation vectors.

    During training:
    - Uses ground truth labels from task i as input to task i+1
    - Uses INITIAL node embeddings (before transformer processing) for the ground truth nodes
    - Inserts GT node embedding RIGHT AFTER each label token

    During inference:
    - Uses predicted labels autoregressively
    - Predicts task i → find positive node → get its initial embedding → feed to task i+1

    Sequence structure (interleaved comp, label, GT tokens):
        Task 1: [nodes(256), COMP1-3(3), LABEL1(1)]
                = 260 tokens
        Task 2: [nodes(256), COMP1-3(3), LABEL1(1), GT1(1), COMP4-6(3), LABEL2(1)]
                = 265 tokens
        Task 3: [nodes(256), COMP1-3(3), LABEL1(1), GT1(1), COMP4-6(3), LABEL2(1),
                 GT2(1), COMP7-9(3), LABEL3(1)]
                = 270 tokens
        Task 4: [nodes(256), COMP1-3(3), LABEL1(1), GT1(1), COMP4-6(3), LABEL2(1),
                 GT2(1), COMP7-9(3), LABEL3(1), GT3(1), COMP10-12(3), LABEL4(1)]
                = 275 tokens

    Memory consideration:
    - 4 sequential forward passes with growing sequences: 260 + 265 + 270 + 275 = 1070 tokens
    - Requires storing activations from all passes for backpropagation
    - ~4x memory of single-pass architectures
    """

    def __init__(self, num_comp_vectors: int = 3, **kwargs):
        self.num_comp_vectors = num_comp_vectors
        super().__init__(**kwargs)

        # Computation vector embeddings (num_comp_vectors per task)
        total_comp_vectors = 4 * num_comp_vectors
        self.comp_embeds = nn.Parameter(torch.randn(total_comp_vectors, self.d_model) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            activation=self.activation,
            batch_first=True,
            norm_first=True
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.num_layers,
            norm=nn.LayerNorm(self.d_model)
        )

        # Initialize weights
        self._init_weights()

    def forward(
        self,
        features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[Dict[str, torch.Tensor]] = None,
        use_teacher_forcing: bool = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with teacher forcing support.

        Args:
            features: [batch_size, num_nodes, feature_dim] - node features
            attention_mask: [batch_size, num_nodes] - padding mask (optional)
            labels: Dict with ground truth labels for teacher forcing (optional)
                    Each value is [batch_size, num_nodes] with 0/1 labels
            use_teacher_forcing: Override training mode behavior
                                - None (default): use self.training
                                - True: force teacher forcing (use labels)
                                - False: force autoregressive (use predictions)

        Returns:
            Dict with 4 task predictions, each [batch_size, num_nodes, 1]
        """
        # Determine mode
        if use_teacher_forcing is None:
            use_teacher_forcing = self.training

        if use_teacher_forcing and labels is None:
            raise ValueError("labels must be provided when use_teacher_forcing=True")

        batch_size, num_nodes = features.shape[:2]
        device = features.device

        # Apply learned positional encoding if configured
        features = self._apply_learned_positional_encoding(features)

        # Project node features
        node_embeds = self.input_projection(features)  # [batch, num_nodes, d_model]

        outputs = {}
        gt_tokens_list = []  # Store GT tokens from previous tasks

        # Process each task sequentially
        for task_idx, task_name in enumerate(self.TASK_NAMES):
            # Build sequence: [nodes, comp1, label1, GT1, comp2, label2, GT2, ..., comp_current, label_current]
            sequence_parts = [node_embeds]  # Start with nodes (256)

            # Add all previous tasks' comp vectors, labels, and GT tokens
            for prev_task_idx in range(task_idx):
                # Add comp vectors for previous task
                comp_start = prev_task_idx * self.num_comp_vectors
                comp_end = comp_start + self.num_comp_vectors
                comp_vecs = self.comp_embeds[comp_start:comp_end].unsqueeze(0).expand(batch_size, -1, -1)
                sequence_parts.append(comp_vecs)

                # Add label token for previous task
                label_vec = self.label_embeds[prev_task_idx:prev_task_idx+1].unsqueeze(0).expand(batch_size, 1, -1)
                sequence_parts.append(label_vec)

                # Add GT token for previous task
                sequence_parts.append(gt_tokens_list[prev_task_idx])

            # Add comp vectors for current task
            comp_start = task_idx * self.num_comp_vectors
            comp_end = comp_start + self.num_comp_vectors
            comp_vecs = self.comp_embeds[comp_start:comp_end].unsqueeze(0).expand(batch_size, -1, -1)
            sequence_parts.append(comp_vecs)

            # Add label token for current task
            label_vec = self.label_embeds[task_idx:task_idx+1].unsqueeze(0).expand(batch_size, 1, -1)
            sequence_parts.append(label_vec)

            # Concatenate sequence
            sequence = torch.cat(sequence_parts, dim=1)
            seq_len = sequence.shape[1]

            # Create hybrid attention mask
            attn_mask = self._create_hybrid_attention_mask(seq_len, num_nodes, device)

            # Pass through transformer
            encoded = self.transformer_encoder(sequence, mask=attn_mask)

            # Extract label position: nodes + (comp + label + GT) * prev_tasks + comp
            label_pos = num_nodes + (self.num_comp_vectors + 1 + 1) * task_idx + self.num_comp_vectors
            label_embed = encoded[:, label_pos, :]  # [batch, d_model]

            # Extract node embeddings
            node_embeds_final = encoded[:, :num_nodes, :]  # [batch, num_nodes, d_model]

            # Project to node predictions
            classifier = getattr(self, f'classifier_task{task_idx+1}')
            preds = self._label_to_node_predictions(
                label_embed, node_embeds_final, classifier
            )
            outputs[task_name] = preds  # [batch, num_nodes, 1]

            # Extract GT token for this task (to be used in next tasks)
            if use_teacher_forcing:
                # Use ground truth labels
                labels_binary = labels[task_name].float()  # [batch, 256]
            else:
                # Use predicted labels (autoregressive)
                pred_probs = F.softmax(preds.squeeze(-1), dim=-1)  # [batch, 256]
                # Convert to one-hot via argmax
                predicted_indices = pred_probs.argmax(dim=-1)  # [batch]
                labels_binary = torch.zeros_like(pred_probs)
                labels_binary.scatter_(1, predicted_indices.unsqueeze(-1), 1.0)  # [batch, 256]

            # Extract index of the positive node
            positive_indices = labels_binary.argmax(dim=1)  # [batch]

            # Extract the INITIAL embedding of the ground truth node (before transformer processing)
            # This represents the raw features/identity of the GT node that task i+1 should condition on
            # (e.g., if GT is node 221, we take the initial embedding: node_embeds[:, 221, :])
            batch_indices = torch.arange(batch_size, device=device)
            gt_token = node_embeds[batch_indices, positive_indices, :].unsqueeze(1)  # [batch, 1, d_model]

            gt_tokens_list.append(gt_token)

        return outputs


class MultiTaskConfig:
    """Configuration for multi-task transformer models."""

    def __init__(
        self,
        feature_dim: int = 44,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        activation: str = "gelu",
        architecture: str = "basic",  # 'basic', 'cached', 'comp', 'cached_comp'
        num_comp_vectors: int = 3,
        positional_encoding: Optional[str] = None,
        board_size: int = 16,
        pos_encoding_dim: int = 0,
        pos_combine_method: str = "concat",
    ):
        """
        Args:
            feature_dim: Input feature dimension
            d_model: Transformer embedding dimension
            nhead: Number of attention heads
            num_layers: Number of transformer encoder layers
            dim_feedforward: Feedforward network dimension
            dropout: Dropout probability
            activation: Activation function ('relu' or 'gelu')
            architecture: Architecture type ('basic', 'cached', 'comp', 'cached_comp')
            num_comp_vectors: Number of computation vectors per task (for comp architectures)
            positional_encoding: Type of positional encoding
            board_size: Board size
            pos_encoding_dim: Positional encoding dimension
            pos_combine_method: How to combine positional encodings
        """
        self.feature_dim = feature_dim
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.activation = activation
        self.architecture = architecture
        self.num_comp_vectors = num_comp_vectors
        self.positional_encoding = positional_encoding
        self.board_size = board_size
        self.pos_encoding_dim = pos_encoding_dim
        self.pos_combine_method = pos_combine_method

    def to_dict(self):
        """Convert config to dictionary for serialization."""
        return {
            "feature_dim": self.feature_dim,
            "d_model": self.d_model,
            "nhead": self.nhead,
            "num_layers": self.num_layers,
            "dim_feedforward": self.dim_feedforward,
            "dropout": self.dropout,
            "activation": self.activation,
            "architecture": self.architecture,
            "num_comp_vectors": self.num_comp_vectors,
            "positional_encoding": self.positional_encoding,
            "board_size": self.board_size,
            "pos_encoding_dim": self.pos_encoding_dim,
            "pos_combine_method": self.pos_combine_method,
        }


def create_multitask_model(config: MultiTaskConfig) -> BaseMultiTaskTransformer:
    """
    Factory function to create multi-task model from config.

    Args:
        config: MultiTaskConfig instance

    Returns:
        Instantiated multi-task model

    Raises:
        ValueError: If architecture type is invalid
    """
    # Remove architecture from kwargs (it's not a model parameter)
    model_kwargs = config.to_dict()
    architecture = model_kwargs.pop('architecture')
    num_comp_vectors = model_kwargs.pop('num_comp_vectors')

    if architecture == 'basic':
        return BasicSequentialMultiTask(**model_kwargs)
    elif architecture == 'cached':
        return CachedSequentialMultiTask(**model_kwargs)
    elif architecture == 'comp':
        return ComputationVectorMultiTask(num_comp_vectors=num_comp_vectors, **model_kwargs)
    elif architecture == 'cached_comp':
        return CachedComputationVectorMultiTask(num_comp_vectors=num_comp_vectors, **model_kwargs)
    elif architecture == 'teacher_forcing':
        return TeacherForcingComputationVectorMultiTask(num_comp_vectors=num_comp_vectors, **model_kwargs)
    else:
        raise ValueError(
            f"Invalid architecture: {architecture}. "
            f"Must be one of: 'basic', 'cached', 'comp', 'cached_comp', 'teacher_forcing'"
        )
