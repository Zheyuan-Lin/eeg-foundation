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
        use_multiscale: Whether to use multi-scale convolutions
        multiscale_kernels: List of kernel sizes for multi-scale branches
    """

    def __init__(
        self,
        n_channels=20,
        chunk_len=500,
        n_filters=40,
        filter_len=25,
        pool_len=75,
        pool_stride=15,
        use_multiscale=False,
        multiscale_kernels=None
    ):
        super().__init__()

        self.n_channels = n_channels
        self.chunk_len = chunk_len
        self.use_multiscale = use_multiscale

        if use_multiscale and multiscale_kernels:
            # Multi-scale temporal convolutions
            self.conv_branches = nn.ModuleList([
                nn.Conv1d(
                    in_channels=n_channels,
                    out_channels=n_filters,
                    kernel_size=k,
                    bias=True
                ) for k in multiscale_kernels
            ])

            # Batch normalization for each branch
            self.bn_branches = nn.ModuleList([
                nn.BatchNorm1d(n_filters) for _ in multiscale_kernels
            ])

            # Total filters after concatenation
            total_filters = n_filters * len(multiscale_kernels)

            # Use the smallest kernel to calculate output size
            min_kernel = min(multiscale_kernels)
            conv_out = chunk_len - min_kernel + 1
        else:
            # Single-scale temporal convolution
            self.temporal_conv = nn.Conv1d(
                in_channels=n_channels,
                out_channels=n_filters,
                kernel_size=filter_len,
                bias=True
            )

            # Batch normalization
            self.bn = nn.BatchNorm1d(n_filters)

            total_filters = n_filters
            conv_out = chunk_len - filter_len + 1

        # Activation
        self.activation = nn.ELU()

        # Temporal pooling
        self.pool = nn.AvgPool1d(
            kernel_size=pool_len,
            stride=pool_stride
        )

        # Calculate output dimension
        pool_out = (conv_out - pool_len) // pool_stride + 1
        self.output_dim = pool_out * total_filters

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, channels, time)

        Returns:
            Feature vector of shape (batch, output_dim)
        """
        # x: (batch, channels, time)

        if self.use_multiscale:
            # Multi-scale convolutions: apply each branch and concatenate
            branch_outputs = []
            min_time = None

            for conv, bn in zip(self.conv_branches, self.bn_branches):
                out = conv(x)  # (batch, n_filters, time')
                out = bn(out)
                out = self.activation(out)

                # Track minimum time dimension
                if min_time is None:
                    min_time = out.size(2)
                else:
                    min_time = min(min_time, out.size(2))

                branch_outputs.append(out)

            # Align all branches to same time dimension (crop to minimum)
            aligned_outputs = [out[:, :, :min_time] for out in branch_outputs]

            # Concatenate along channel dimension
            x = torch.cat(aligned_outputs, dim=1)  # (batch, total_filters, time')
        else:
            # Single-scale convolution
            x = self.temporal_conv(x)      # (batch, n_filters, time')
            x = self.bn(x)
            x = self.activation(x)

        # Pooling
        x = self.pool(x)                # (batch, filters, time'')

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
