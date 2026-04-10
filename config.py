"""
Core configuration for foundation model.
Simple dictionary-based config - no complex CLI parsing.
"""

def get_config():
    """Get default configuration."""
    return {
        # Data
        'data_path': './data',
        'num_channels': 20,
        'sampling_rate': 250,
        'normalization': 'minmax',  # Options: 'minmax', 'zscore', 'none'

        # Data augmentation (applied to training data only)
        'augmentation': False,            # Enable/disable augmentation
        'aug_prob': 0.5,                  # Probability of applying augmentation per sample
        'aug_noise_std': 0.1,             # Gaussian noise standard deviation
        'aug_dropout_prob': 0.1,          # Probability of dropping each channel
        'aug_scale_range': (0.9, 1.1),    # Amplitude scaling range

        # Chunking
        'chunk_len': 500,          # Samples per chunk
        'num_chunks': 34,          # Number of chunks per sequence
        'chunk_overlap': 100,      # Overlap between chunks

        # Encoder
        'use_encoder': True,
        'n_filters': 40,           # Number of temporal filters
        'filter_len': 25,          # Temporal filter length
        'pool_len': 75,            # Pooling window
        'pool_stride': 15,         # Pooling stride

        # Multi-scale temporal convolutions
        'use_multiscale': True,    # Enable multi-scale convolutions
        'multiscale_kernels': [15, 25, 35],  # Kernel sizes for multi-scale branches

        # Embedder
        'embed_dim': 256,          # Embedding dimension
        'parcellation_dim': None,  # Computed from encoder output

        # Decoder (Transformer)
        'num_layers': 4,           # Number of transformer layers
        'num_heads': 4,            # Attention heads
        'ff_dim': 1024,            # Feed-forward dimension
        'dropout': 0.1,

        # Training
        'batch_size': 8,
        'learning_rate': 1e-4,
        'num_epochs': 10,
        'device': 'cuda',

        # Learning rate scheduler
        'use_scheduler': True,         # Enable/disable LR scheduler
        'warmup_ratio': 0.1,           # Warmup as fraction of total steps
        'min_lr': 1e-6,                # Minimum LR for cosine annealing

        # Gradient clipping
        'clip_grad_norm': 1.0,         # Max gradient norm (None to disable)
        'log_grad_norm': False,        # Log gradient norms during training

        # Logging
        'log_dir': './logs',
        'save_every': 1000,        # Save checkpoint every N steps
        'log_every': 100,          # Log every N steps

        # Checkpointing
        'max_checkpoints': 3,      # Keep only best N checkpoints
    }


def update_config(config, **kwargs):
    """Update config with custom values."""
    config.update(kwargs)

    # Compute parcellation_dim if encoder is used
    if config['use_encoder']:
        if config.get('use_multiscale', False) and config.get('multiscale_kernels'):
            # Multi-scale: use minimum kernel size
            min_kernel = min(config['multiscale_kernels'])
            num_scales = len(config['multiscale_kernels'])
            config['parcellation_dim'] = (
                (config['chunk_len'] - min_kernel + 1 - config['pool_len'])
                // config['pool_stride'] + 1
            ) * config['n_filters'] * num_scales
        else:
            # Single-scale: original formula
            config['parcellation_dim'] = (
                (config['chunk_len'] - config['filter_len'] + 1 - config['pool_len'])
                // config['pool_stride'] + 1
            ) * config['n_filters']
    else:
        config['parcellation_dim'] = config['chunk_len'] * config['num_channels']

    return config
