"""
Simple Transformer decoder.
Causal transformer for modeling chunk sequences.
"""

import torch
import torch.nn as nn
import math


class RelativePositionBias(nn.Module):
    """
    Relative positional bias for transformer attention.

    Args:
        num_heads: Number of attention heads
        max_distance: Maximum relative distance
    """

    def __init__(self, num_heads, max_distance=32):
        super().__init__()

        self.num_heads = num_heads
        self.max_distance = max_distance

        # Learnable relative position bias
        # +1 for distances beyond max_distance
        self.relative_bias = nn.Parameter(
            torch.randn(num_heads, 2 * max_distance + 1)
        )

    def forward(self, seq_len):
        """
        Compute relative position bias.

        Args:
            seq_len: Sequence length

        Returns:
            bias: (num_heads, seq_len, seq_len) relative position bias
        """
        # Create position indices
        positions = torch.arange(seq_len, device=self.relative_bias.device)

        # Compute relative positions: (seq_len, seq_len)
        relative_positions = positions.unsqueeze(0) - positions.unsqueeze(1)

        # Clip to max distance
        clipped_positions = torch.clamp(
            relative_positions,
            -self.max_distance,
            self.max_distance
        )

        # Shift to positive indices
        bias_indices = clipped_positions + self.max_distance

        # Get bias values: (seq_len, seq_len, num_heads)
        bias = self.relative_bias[:, bias_indices]  # (num_heads, seq_len, seq_len)

        return bias


class TransformerDecoder(nn.Module):
    """
    Simple causal transformer decoder with optional relative positional encoding.

    Args:
        embed_dim: Embedding dimension
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        ff_dim: Feed-forward dimension
        dropout: Dropout rate
        max_seq_len: Maximum sequence length
        output_dim: Output dimension (for reconstruction)
        use_relative_pos: Whether to use relative positional encoding
        max_relative_distance: Maximum relative distance for bias
    """

    def __init__(
        self,
        embed_dim=256,
        num_layers=4,
        num_heads=4,
        ff_dim=1024,
        dropout=0.1,
        max_seq_len=512,
        output_dim=None,  # For reconstruction - typically parcellation_dim
        use_relative_pos=False,
        max_relative_distance=32
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.output_dim = output_dim if output_dim is not None else embed_dim
        self.use_relative_pos = use_relative_pos

        # Positional encoding (only if not using relative)
        if not use_relative_pos:
            self.pos_encoding = PositionalEncoding(embed_dim, max_seq_len, dropout)

        # Relative positional bias
        if use_relative_pos:
            self.relative_pos_bias = RelativePositionBias(num_heads, max_relative_distance)

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
        # Add positional encoding (if not using relative)
        if not self.use_relative_pos:
            x = self.pos_encoding(x)

        # Create causal mask for transformer
        seq_len = x.size(1)
        causal_mask = self._generate_square_subsequent_mask(seq_len).to(x.device)

        # Add relative positional bias if enabled
        if self.use_relative_pos:
            rel_bias = self.relative_pos_bias(seq_len)  # (num_heads, seq_len, seq_len)
            # Add bias to causal mask (broadcast over batch and heads)
            # Note: This is a simplified version; for full integration, would need
            # custom attention layers. Here we add it to the mask.
            causal_mask = causal_mask + rel_bias.mean(0)  # Average over heads for simplicity

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
