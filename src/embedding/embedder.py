"""
CSM (Causal Sequence Modeling) Embedder.
Handles masking strategy and loss computation for pretraining.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CSMEmbedder(nn.Module):
    """
    CSM Embedder for foundation model pretraining.

    Responsibilities:
    1. Project chunks to embedding dimension
    2. Add learnable mask token
    3. Implement CSM masking strategy
    4. Compute reconstruction loss

    Args:
        in_dim: Input dimension (parcellation_dim)
        embed_dim: Embedding dimension
        dropout: Dropout rate
    """

    def __init__(self, in_dim, embed_dim, dropout=0.1):
        super().__init__()

        self.in_dim = in_dim
        self.embed_dim = embed_dim

        # Projection layer
        self.projection = nn.Linear(in_dim, embed_dim)

        # Learnable mask token
        self.mask_token = nn.Parameter(torch.randn(1, 1, in_dim))

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Project inputs to embedding space.

        Args:
            x: (batch, num_chunks, in_dim)

        Returns:
            (batch, num_chunks, embed_dim)
        """
        x = self.projection(x)
        x = self.dropout(x)
        return x

    def prepare_batch(self, inputs):
        """
        Prepare batch for CSM training.

        Randomly masks one chunk per sequence and creates causal attention mask.

        Args:
            inputs: (batch, num_chunks, in_dim)

        Returns:
            dict with:
                - inputs: Inputs with one chunk replaced by mask_token
                - attention_mask: Causal mask up to masked position
                - mask_positions: Which positions were masked
                - original_inputs: Original inputs (for loss computation)
        """
        batch_size, num_chunks, in_dim = inputs.shape
        device = inputs.device

        # Save original inputs for loss
        original_inputs = inputs.clone()

        # Randomly select one position per sequence to mask
        mask_positions = torch.randint(0, num_chunks, (batch_size,), device=device)

        # Replace masked positions with mask token
        for i in range(batch_size):
            inputs[i, mask_positions[i]] = self.mask_token

        # Create causal attention mask
        # Only attend up to (and including) the masked position
        attention_mask = torch.zeros(batch_size, num_chunks, device=device)
        for i in range(batch_size):
            attention_mask[i, :mask_positions[i] + 1] = 1

        return {
            'inputs': inputs,
            'attention_mask': attention_mask,
            'mask_positions': mask_positions,
            'original_inputs': original_inputs
        }

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

        # Extract predictions at masked positions
        masked_predictions = torch.stack([
            predictions[i, mask_positions[i]]
            for i in range(batch_size)
        ])  # (batch, dim)

        # Extract original values at masked positions
        masked_targets = torch.stack([
            original_inputs[i, mask_positions[i]]
            for i in range(batch_size)
        ])  # (batch, in_dim)

        # MSE loss
        loss = F.mse_loss(masked_predictions, masked_targets)

        return loss
