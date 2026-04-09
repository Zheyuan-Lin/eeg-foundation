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

        # Logging
        'log_dir': './logs',
        'save_every': 1000,        # Save checkpoint every N steps
        'log_every': 100,          # Log every N steps
    }


def update_config(config, **kwargs):
    """Update config with custom values."""
    config.update(kwargs)

    # Compute parcellation_dim if encoder is used
    if config['use_encoder']:
        # Formula from the full model
        config['parcellation_dim'] = (
            (config['chunk_len'] - config['filter_len'] + 1 - config['pool_len'])
            // config['pool_stride'] + 1
        ) * config['n_filters']
    else:
        config['parcellation_dim'] = config['chunk_len'] * config['num_channels']

    return config
