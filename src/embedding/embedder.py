"""
CSM (Causal Sequence Modeling) Embedder.
Handles masking strategy and loss computation for pretraining.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CSMEmbedder(nn.Module):
    """
    CSM Embedder for foundation model pretraining.

    Responsibilities:
    1. Project chunks to embedding dimension
    2. Add positional encodings (learned or sinusoidal)
    3. Add learnable mask token
    4. Implement CSM masking strategy
    5. Compute reconstruction loss

    Args:
        in_dim: Input dimension (parcellation_dim)
        embed_dim: Embedding dimension
        dropout: Dropout rate
        max_seq_len: Maximum sequence length
        pos_encoding_type: Type of positional encoding ('learned', 'sinusoidal', 'none')
    """

    def __init__(self, in_dim, embed_dim, dropout=0.1, max_seq_len=512, pos_encoding_type='learned',
                 masking_strategy='span', mask_ratio=0.15, span_length=3,
                 use_contrastive_loss=False, contrastive_temperature=0.07, contrastive_weight=0.5):
        super().__init__()

        self.in_dim = in_dim
        self.embed_dim = embed_dim
        self.pos_encoding_type = pos_encoding_type
        self.masking_strategy = masking_strategy
        self.mask_ratio = mask_ratio
        self.span_length = span_length
        self.use_contrastive_loss = use_contrastive_loss
        self.contrastive_temperature = contrastive_temperature
        self.contrastive_weight = contrastive_weight

        # Projection layer
        self.projection = nn.Linear(in_dim, embed_dim)

        # Learnable mask token
        self.mask_token = nn.Parameter(torch.randn(1, 1, in_dim))

        # Positional encoding
        if pos_encoding_type == 'learned':
            # Learnable positional embeddings
            self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, embed_dim))
        elif pos_encoding_type == 'sinusoidal':
            # Sinusoidal positional encoding (fixed)
            self.register_buffer('pos_embedding', self._get_sinusoidal_encoding(max_seq_len, embed_dim))
        # else: no positional encoding

        # Contrastive learning projection head
        if use_contrastive_loss:
            self.contrast_proj = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, embed_dim // 2)
            )

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def _get_sinusoidal_encoding(self, max_len, d_model):
        """Generate sinusoidal positional encoding."""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # (1, max_len, d_model)

    def forward(self, x):
        """
        Project inputs to embedding space with positional encoding.

        Args:
            x: (batch, num_chunks, in_dim)

        Returns:
            (batch, num_chunks, embed_dim)
        """
        # Project to embedding dimension
        x = self.projection(x)

        # Add positional encoding
        if self.pos_encoding_type != 'none':
            seq_len = x.size(1)
            x = x + self.pos_embedding[:, :seq_len, :]

        x = self.dropout(x)
        return x

    def prepare_batch(self, inputs):
        """
        Prepare batch for CSM training with advanced masking strategies.

        Args:
            inputs: (batch, num_chunks, in_dim)

        Returns:
            dict with:
                - inputs: Inputs with masked chunks replaced by mask_token
                - attention_mask: Causal mask up to masked position
                - mask_positions: Which positions were masked (list of tensors)
                - original_inputs: Original inputs (for loss computation)
        """
        batch_size, num_chunks, in_dim = inputs.shape
        device = inputs.device

        # Save original inputs for loss
        original_inputs = inputs.clone()

        # Apply masking strategy
        if self.masking_strategy == 'random':
            mask_positions = self._random_masking(batch_size, num_chunks, device)
        elif self.masking_strategy == 'span':
            mask_positions = self._span_masking(batch_size, num_chunks, device)
        elif self.masking_strategy == 'block':
            mask_positions = self._block_masking(batch_size, num_chunks, device)
        else:
            # Default: single random position
            mask_positions = [torch.randint(0, num_chunks, (1,), device=device) for _ in range(batch_size)]

        # Replace masked positions with mask token
        for i in range(batch_size):
            for pos in mask_positions[i]:
                inputs[i, pos] = self.mask_token

        # Create causal attention mask (attend up to max masked position)
        attention_mask = torch.zeros(batch_size, num_chunks, device=device)
        for i in range(batch_size):
            if len(mask_positions[i]) > 0:
                max_pos = mask_positions[i].max().item()
                attention_mask[i, :max_pos + 1] = 1

        return {
            'inputs': inputs,
            'attention_mask': attention_mask,
            'mask_positions': mask_positions,
            'original_inputs': original_inputs
        }

    def _random_masking(self, batch_size, num_chunks, device):
        """Random masking: mask random individual positions."""
        mask_positions = []
        num_mask = max(1, int(num_chunks * self.mask_ratio))

        for _ in range(batch_size):
            positions = torch.randperm(num_chunks, device=device)[:num_mask]
            mask_positions.append(positions)

        return mask_positions

    def _span_masking(self, batch_size, num_chunks, device):
        """Span masking: mask consecutive spans of tokens."""
        mask_positions = []
        target_mask_count = max(1, int(num_chunks * self.mask_ratio))

        for _ in range(batch_size):
            masked = []
            while len(masked) < target_mask_count:
                # Random span length (Poisson distribution)
                span_len = min(torch.poisson(torch.tensor([self.span_length])).int().item(), num_chunks)
                span_len = max(1, span_len)

                # Random start position
                start = torch.randint(0, max(1, num_chunks - span_len + 1), (1,), device=device).item()
                span = list(range(start, min(start + span_len, num_chunks)))

                # Add span to masked positions
                masked.extend(span)

                # Avoid masking too many
                if len(masked) >= target_mask_count:
                    masked = masked[:target_mask_count]
                    break

            mask_positions.append(torch.tensor(masked, device=device, dtype=torch.long))

        return mask_positions

    def _block_masking(self, batch_size, num_chunks, device):
        """Block masking: mask in a block pattern."""
        mask_positions = []
        target_mask_count = max(1, int(num_chunks * self.mask_ratio))

        for _ in range(batch_size):
            # Create contiguous block
            start = torch.randint(0, max(1, num_chunks - target_mask_count + 1), (1,), device=device).item()
            positions = torch.arange(start, min(start + target_mask_count, num_chunks), device=device)
            mask_positions.append(positions)

        return mask_positions

    def compute_loss(self, predictions, batch_dict):
        """
        Compute CSM reconstruction loss.

        Only compute loss at the masked positions.

        Args:
            predictions: Model outputs (batch, num_chunks, in_dim or embed_dim)
            batch_dict: Dictionary from prepare_batch

        Returns:
            loss: Scalar tensor
        """
        batch_size = predictions.shape[0]
        mask_positions = batch_dict['mask_positions']
        original_inputs = batch_dict['original_inputs']

        # Extract predictions and targets at all masked positions
        all_predictions = []
        all_targets = []

        for i in range(batch_size):
            for pos in mask_positions[i]:
                all_predictions.append(predictions[i, pos])
                all_targets.append(original_inputs[i, pos])

        # Stack into tensors
        if len(all_predictions) > 0:
            all_predictions = torch.stack(all_predictions)  # (total_masked, dim)
            all_targets = torch.stack(all_targets)          # (total_masked, in_dim)

            # MSE loss
            loss = F.mse_loss(all_predictions, all_targets)
        else:
            # No masked positions (shouldn't happen)
            loss = torch.tensor(0.0, device=predictions.device)

        return loss

    def compute_contrastive_loss(self, embeddings, batch_dict):
        """
        Compute contrastive loss using masked and unmasked representations.

        Creates positive pairs from the same sequence and negative pairs from different sequences.

        Args:
            embeddings: Embedded representations (batch, num_chunks, embed_dim)
            batch_dict: Dictionary from prepare_batch

        Returns:
            loss: Scalar contrastive loss
        """
        batch_size = embeddings.shape[0]
        mask_positions = batch_dict['mask_positions']

        # Extract representations at masked positions (anchors)
        anchors = []
        for i in range(batch_size):
            if len(mask_positions[i]) > 0:
                # Average over all masked positions for this sequence
                masked_emb = embeddings[i, mask_positions[i]].mean(dim=0)
                anchors.append(masked_emb)

        if len(anchors) == 0:
            return torch.tensor(0.0, device=embeddings.device)

        anchors = torch.stack(anchors)  # (batch, embed_dim)

        # Project to contrastive space
        anchors = self.contrast_proj(anchors)  # (batch, embed_dim // 2)

        # Normalize
        anchors = F.normalize(anchors, dim=-1)

        # Compute similarity matrix
        similarity = torch.mm(anchors, anchors.t()) / self.contrastive_temperature  # (batch, batch)

        # Labels: positives are on the diagonal (same sequence)
        labels = torch.arange(batch_size, device=embeddings.device)

        # Contrastive loss (InfoNCE)
        loss = F.cross_entropy(similarity, labels)

        return loss

    def compute_combined_loss(self, predictions, embeddings, batch_dict):
        """
        Compute combined reconstruction and contrastive loss.

        Args:
            predictions: Model outputs (batch, num_chunks, in_dim)
            embeddings: Embedded representations (batch, num_chunks, embed_dim)
            batch_dict: Dictionary from prepare_batch

        Returns:
            loss: Combined loss
            loss_dict: Dictionary with individual losses
        """
        # Reconstruction loss
        recon_loss = self.compute_loss(predictions, batch_dict)

        if self.use_contrastive_loss:
            # Contrastive loss
            contrast_loss = self.compute_contrastive_loss(embeddings, batch_dict)

            # Weighted combination
            total_loss = (1 - self.contrastive_weight) * recon_loss + \
                         self.contrastive_weight * contrast_loss

            loss_dict = {
                'total': total_loss,
                'reconstruction': recon_loss,
                'contrastive': contrast_loss
            }
        else:
            total_loss = recon_loss
            loss_dict = {
                'total': total_loss,
                'reconstruction': recon_loss
            }

        return total_loss, loss_dict
