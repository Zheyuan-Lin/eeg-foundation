# EEG Foundation Model

A state-of-the-art foundation model for EEG signal analysis using self-supervised learning with Causal Sequence Modeling (CSM).

## Features

### Advanced Architecture

#### Encoder Enhancements
- **Multi-scale Temporal Convolutions**: Parallel convolution branches with different kernel sizes (15, 25, 35) to capture temporal patterns at multiple scales
- **Attention Pooling**: Learned attention weights for intelligent temporal feature aggregation
- **Channel-wise Attention**: Multi-head self-attention across EEG channels to model spatial dependencies

#### Embedder Capabilities
- **Flexible Positional Encoding**: Choose between learned, sinusoidal, or no positional encoding
- **Advanced Masking Strategies**:
  - Random masking: Independent random positions
  - Span masking: Consecutive token spans with Poisson-distributed lengths
  - Block masking: Contiguous blocks of tokens
- **Contrastive Learning**: InfoNCE contrastive loss combined with reconstruction loss for better representations

#### Decoder Features
- **Relative Positional Encoding**: Learnable relative position bias for improved sequence modeling
- **Sparse Attention Patterns**: Efficient local + global attention mechanism for long sequences
- **Causal Transformer**: Auto-regressive modeling for temporal sequence prediction

### Training Infrastructure
- Learning rate scheduling with cosine annealing and warmup
- Gradient clipping for stable training
- Enhanced checkpointing with best model tracking
- Multi-format data support (.edf, .fif)
- Data augmentation (noise, dropout, scaling, jitter)
- Normalization strategies (min-max, z-score)

## Installation

```bash
# Clone the repository
git clone https://github.com/Zheyuan-Lin/eeg-foundation.git
cd eeg-foundation

# Install dependencies
pip install torch numpy mne pyedflib
```

## Quick Start

### Training

```python
from config import get_config, update_config
from model import build_model
from data.dataset import EEGDataset
from trainer.trainer import Trainer

# Load and configure
config = get_config()
config = update_config(config)

# Build model
model = build_model(config)

# Create dataset
dataset = EEGDataset(
    data_path=config['data_path'],
    num_channels=config['num_channels'],
    chunk_len=config['chunk_len'],
    num_chunks=config['num_chunks'],
    normalization=config['normalization'],
    augmentation=config['augmentation']
)

# Train
trainer = Trainer(model, dataset, config)
trainer.train()
```

### Inference

```python
# Load trained model
model = build_model(config)
checkpoint = torch.load('checkpoints/best_model.pt')
model.load_state_dict(checkpoint['model'])

# Get embeddings
with torch.no_grad():
    embeddings = model(eeg_data, return_embeddings=True)
```

## Configuration

The model is highly configurable through `config.py`. Key parameters:

### Data Configuration
```python
'num_channels': 20,              # Number of EEG channels
'sampling_rate': 250,            # Sampling rate in Hz
'chunk_len': 500,                # Samples per chunk
'num_chunks': 34,                # Chunks per sequence
'normalization': 'minmax',       # 'minmax', 'zscore', or 'none'
```

### Encoder Options
```python
'use_multiscale': True,          # Enable multi-scale convolutions
'multiscale_kernels': [15, 25, 35],  # Kernel sizes
'use_attention_pooling': True,   # Use attention pooling
'use_channel_attention': True,   # Enable channel attention
'channel_attn_heads': 4,         # Attention heads
```

### Embedder Options
```python
'embed_dim': 256,                # Embedding dimension
'pos_encoding_type': 'learned',  # 'learned', 'sinusoidal', 'none'
'masking_strategy': 'span',      # 'random', 'span', 'block'
'mask_ratio': 0.15,              # Fraction to mask
'span_length': 3,                # Avg span length
```

### Contrastive Learning
```python
'use_contrastive_loss': True,    # Enable contrastive learning
'contrastive_temperature': 0.07, # Temperature scaling
'contrastive_weight': 0.5,       # Loss weight (vs reconstruction)
```

### Decoder Options
```python
'num_layers': 4,                 # Transformer layers
'num_heads': 4,                  # Attention heads
'ff_dim': 1024,                  # Feed-forward dim
'use_relative_pos': True,        # Relative positional encoding
'use_sparse_attention': False,   # Sparse attention patterns
'local_window_size': 16,         # Local attention window
```

### Training
```python
'batch_size': 8,
'learning_rate': 1e-4,
'num_epochs': 10,
'use_scheduler': True,           # LR scheduling
'warmup_ratio': 0.1,             # Warmup fraction
'clip_grad_norm': 1.0,           # Gradient clipping
```

## Model Architecture

```
Raw EEG (batch, num_chunks, channels, time)
    ↓
┌──────────────────────────────────────┐
│ Encoder (per chunk)                  │
│  ├─ Channel Attention (optional)     │
│  ├─ Multi-scale Convolutions         │
│  │   ├─ Conv 15 ──┐                  │
│  │   ├─ Conv 25 ──┼─→ Concatenate    │
│  │   └─ Conv 35 ──┘                  │
│  ├─ Batch Norm + ELU                 │
│  ├─ Average Pooling                  │
│  └─ Attention Pooling (optional)     │
└──────────────────────────────────────┘
    ↓ (batch, num_chunks, parcellation_dim)
┌──────────────────────────────────────┐
│ Embedder                             │
│  ├─ Linear Projection                │
│  ├─ Positional Encoding              │
│  ├─ Masking (random/span/block)      │
│  └─ Dropout                           │
└──────────────────────────────────────┘
    ↓ (batch, num_chunks, embed_dim)
┌──────────────────────────────────────┐
│ Decoder (Transformer)                │
│  ├─ Relative Pos Encoding (optional) │
│  ├─ Sparse Attention (optional)      │
│  ├─ Multi-head Self-Attention        │
│  ├─ Feed-Forward Network             │
│  └─ Layer Norm + Residual            │
│  × num_layers                        │
└──────────────────────────────────────┘
    ↓ (batch, num_chunks, parcellation_dim)
┌──────────────────────────────────────┐
│ Loss Computation                     │
│  ├─ Reconstruction Loss (MSE)        │
│  └─ Contrastive Loss (InfoNCE)       │
└──────────────────────────────────────┘
```

## Project Structure

```
eeg-foundation/
├── config.py              # Configuration management
├── model.py               # Main model definition
├── train.py               # Training script
├── data/
│   └── dataset.py         # Dataset implementation
├── src/
│   ├── encoder/
│   │   ├── simpleEncoder.py    # Multi-scale encoder
│   │   └── Conformer.py        # Alternative encoder
│   ├── embedding/
│   │   └── embedder.py         # CSM embedder with masking
│   └── decoder/
│       └── transformer.py      # Transformer decoder
└── trainer/
    └── trainer.py         # Training loop
```

## Development Status

**Phase 1: Core Infrastructure** ✅ (7/7 commits)
- Data normalization and augmentation
- Multi-format support
- LR scheduling and gradient clipping
- Enhanced checkpointing

**Phase 2: Advanced Architecture** ✅ (8/8 commits)
- Multi-scale convolutions
- Attention mechanisms
- Advanced masking strategies
- Contrastive learning
- Relative positional encoding
- Sparse attention patterns

**Phase 3: Training Optimizations** (Planned)
- Mixed precision training
- Gradient accumulation
- Dynamic batch sizing
- EMA of model weights

See [ROADMAP.md](ROADMAP.md) for the complete development plan (66 total commits across 10 phases).

## Performance Tips

1. **Memory optimization**: Enable sparse attention for long sequences
2. **Training speed**: Use multi-scale convolutions with attention pooling
3. **Better representations**: Enable contrastive loss with span masking
4. **Variable sequences**: Use relative positional encoding

## Citation

If you use this code in your research, please cite:

```bibtex
@software{eeg_foundation_model,
  title={EEG Foundation Model},
  author={Your Name},
  year={2024},
  url={https://github.com/Zheyuan-Lin/eeg-foundation}
}
```

## License

MIT License

## Contributing

Contributions are welcome! Please see the roadmap for planned features or propose new improvements.

## Acknowledgments

Built with PyTorch and inspired by modern self-supervised learning techniques in vision and language models adapted for neuroscience applications.
