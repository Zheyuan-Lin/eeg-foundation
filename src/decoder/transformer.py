"""
Simple Transformer decoder.
Causal transformer for modeling chunk sequences.
"""

import torch
import torch.nn as nn
import math


class TransformerDecoder(nn.Module):
    """
    Simple causal transformer decoder.

    Args:
        embed_dim: Embedding dimension
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        ff_dim: Feed-forward dimension
        dropout: Dropout rate
        max_seq_len: Maximum sequence length
    """

    def __init__(
        self,
        embed_dim=256,
        num_layers=4,
        num_heads=4,
        ff_dim=1024,
        dropout=0.1,
        max_seq_len=512,
        output_dim=None  # For reconstruction - typically parcellation_dim
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.output_dim = output_dim if output_dim is not None else embed_dim

        # Positional encoding
        self.pos_encoding = PositionalEncoding(embed_dim, max_seq_len, dropout)

        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Output projection (back to input space for reconstruction)
        # This acts like the "unembedder" in the full model
        self.output_proj = nn.Linear(embed_dim, self.output_dim)

    def forward(self, x, attention_mask=None):
        """
        Forward pass.

        Args:
            x: (batch, seq_len, embed_dim)
            attention_mask: (batch, seq_len) - 1 for positions to attend, 0 to ignore

        Returns:
            (batch, seq_len, embed_dim)
        """
        # Add positional encoding
        x = self.pos_encoding(x)

        # Create causal mask for transformer
        seq_len = x.size(1)
        causal_mask = self._generate_square_subsequent_mask(seq_len).to(x.device)

        # Convert attention_mask to format expected by transformer
        # attention_mask: (batch, seq_len) -> src_key_padding_mask
        if attention_mask is not None:
            # Invert: transformer expects True for positions to IGNORE
            src_key_padding_mask = (attention_mask == 0)
        else:
            src_key_padding_mask = None

        # Transformer
        x = self.transformer(
            x,
            mask=causal_mask,
            src_key_padding_mask=src_key_padding_mask
        )

        # Output projection
        x = self.output_proj(x)

        return x

    def _generate_square_subsequent_mask(self, sz):
        """Generate causal mask (upper triangular)."""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding.
    """

    def __init__(self, embed_dim, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, embed_dim)

        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, embed_dim)
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)
