"""
Simple EEG encoder.
Processes raw EEG chunks into feature vectors.
"""

import torch
import torch.nn as nn


class SimpleEncoder(nn.Module):
    """
    Basic encoder for EEG chunks.
    Uses temporal convolution + pooling to extract features.

    Args:
        n_channels: Number of EEG channels (e.g., 20)
        chunk_len: Length of each chunk in samples
        n_filters: Number of temporal filters
        filter_len: Length of temporal filter
        pool_len: Pooling window size
        pool_stride: Pooling stride
    """

    def __init__(
        self,
        n_channels=20,
        chunk_len=500,
        n_filters=40,
        filter_len=25,
        pool_len=75,
        pool_stride=15
    ):
        super().__init__()

        self.n_channels = n_channels
        self.chunk_len = chunk_len

        # Temporal convolution
        self.temporal_conv = nn.Conv1d(
            in_channels=n_channels,
            out_channels=n_filters,
            kernel_size=filter_len,
            bias=True
        )

        # Batch normalization
        self.bn = nn.BatchNorm1d(n_filters)

        # Activation
        self.activation = nn.ELU()

        # Temporal pooling
        self.pool = nn.AvgPool1d(
            kernel_size=pool_len,
            stride=pool_stride
        )

        # Calculate output dimension
        conv_out = chunk_len - filter_len + 1
        pool_out = (conv_out - pool_len) // pool_stride + 1
        self.output_dim = pool_out * n_filters

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, channels, time)

        Returns:
            Feature vector of shape (batch, output_dim)
        """
        # x: (batch, channels, time)
        x = self.temporal_conv(x)      # (batch, n_filters, time')
        x = self.bn(x)
        x = self.activation(x)
        x = self.pool(x)                # (batch, n_filters, time'')

        # Flatten
        x = x.reshape(x.size(0), -1)   # (batch, output_dim)

        return x


class ChunkEncoder(nn.Module):
    """
    Wrapper that processes multiple chunks.

    Args:
        encoder: Base encoder (SimpleEncoder)
    """

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.output_dim = encoder.output_dim

    def forward(self, x):
        """
        Process multiple chunks.

        Args:
            x: (batch, num_chunks, channels, time)

        Returns:
            (batch, num_chunks, output_dim)
        """
        batch_size, num_chunks, channels, time = x.shape

        # Reshape: (batch * num_chunks, channels, time)
        x = x.view(batch_size * num_chunks, channels, time)

        # Encode each chunk
        x = self.encoder(x)  # (batch * num_chunks, output_dim)

        # Reshape back: (batch, num_chunks, output_dim)
        x = x.view(batch_size, num_chunks, -1)

        return x
