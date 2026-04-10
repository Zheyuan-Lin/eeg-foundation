"""
Main Foundation Model.
Orchestrates encoder, embedder, and decoder.
"""

import torch
import torch.nn as nn


class FoundationModel(nn.Module):
    """
    EEG Foundation Model.

    Architecture:
        Raw EEG → Encoder → Embedder → Decoder → Output

    Args:
        encoder: Encoder module (or None to skip)
        embedder: Embedder module
        decoder: Decoder module
    """

    def __init__(self, encoder, embedder, decoder):
        super().__init__()

        self.encoder = encoder
        self.embedder = embedder
        self.decoder = decoder

    def forward(self, x, return_embeddings=False):
        """
        Forward pass.

        Args:
            x: Raw EEG chunks (batch, num_chunks, channels, time)
               OR encoded features (batch, num_chunks, features) if no encoder
            return_embeddings: If True, return intermediate embeddings

        Returns:
            outputs: (batch, num_chunks, embed_dim)
            embeddings (optional): Intermediate representations
        """
        # 1. Encode (if encoder exists)
        if self.encoder is not None:
            x = self.encoder(x)  # (batch, num_chunks, parcellation_dim)

        # 2. Prepare batch for CSM
        batch_dict = self.embedder.prepare_batch(x)
        inputs = batch_dict['inputs']
        attention_mask = batch_dict['attention_mask']

        # 3. Embed
        embeddings = self.embedder(inputs)  # (batch, num_chunks, embed_dim)

        # 4. Decode
        outputs = self.decoder(embeddings, attention_mask)  # (batch, num_chunks, embed_dim)

        if return_embeddings:
            return outputs, embeddings, batch_dict
        else:
            return outputs

    def compute_loss(self, x):
        """
        Compute CSM training loss.

        Args:
            x: Raw EEG chunks

        Returns:
            loss: Scalar tensor
        """
        # Forward pass with batch dict
        outputs, _, batch_dict = self.forward(x, return_embeddings=True)

        # Compute loss at masked positions
        loss = self.embedder.compute_loss(outputs, batch_dict)

        return loss

    def save_checkpoint(self, path, epoch=None, optimizer=None, scheduler=None, **kwargs):
        """
        Save model checkpoint with full training state.

        Args:
            path: Path to save checkpoint
            epoch: Current epoch number
            optimizer: Optimizer state to save
            scheduler: LR scheduler state to save
            **kwargs: Additional items to save (e.g., metrics)
        """
        checkpoint = {
            'encoder': self.encoder.state_dict() if self.encoder else None,
            'embedder': self.embedder.state_dict(),
            'decoder': self.decoder.state_dict(),
        }

        # Add training state
        if epoch is not None:
            checkpoint['epoch'] = epoch
        if optimizer is not None:
            checkpoint['optimizer'] = optimizer.state_dict()
        if scheduler is not None:
            checkpoint['scheduler'] = scheduler.state_dict()

        # Add any additional data
        checkpoint.update(kwargs)

        torch.save(checkpoint, path)
        print(f'Saved checkpoint to {path}')

    def load_checkpoint(self, path, optimizer=None, scheduler=None):
        """
        Load model checkpoint and optionally restore training state.

        Args:
            path: Path to checkpoint
            optimizer: Optimizer to load state into
            scheduler: Scheduler to load state into

        Returns:
            checkpoint dict with additional info (epoch, etc.)
        """
        checkpoint = torch.load(path)

        # Load model weights
        if self.encoder and checkpoint.get('encoder'):
            self.encoder.load_state_dict(checkpoint['encoder'])
        self.embedder.load_state_dict(checkpoint['embedder'])
        self.decoder.load_state_dict(checkpoint['decoder'])

        # Load optimizer state
        if optimizer and 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])

        # Load scheduler state
        if scheduler and 'scheduler' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler'])

        print(f'Loaded checkpoint from {path}')

        return checkpoint


def build_model(config):
    """
    Build foundation model from config.

    Args:
        config: Configuration dictionary

    Returns:
        FoundationModel instance
    """
    # Build encoder
    if config['use_encoder']:
        from src.encoder.simpleEncoder import SimpleEncoder, ChunkEncoder

        base_encoder = SimpleEncoder(
            n_channels=config['num_channels'],
            chunk_len=config['chunk_len'],
            n_filters=config['n_filters'],
            filter_len=config['filter_len'],
            pool_len=config['pool_len'],
            pool_stride=config['pool_stride'],
            use_multiscale=config.get('use_multiscale', False),
            multiscale_kernels=config.get('multiscale_kernels', None),
            use_attention_pooling=config.get('use_attention_pooling', False),
            use_channel_attention=config.get('use_channel_attention', False),
            channel_attn_heads=config.get('channel_attn_heads', 4)
        )
        encoder = ChunkEncoder(base_encoder)
    else:
        encoder = None

    # Build embedder
    from src.embedding.embedder import CSMEmbedder

    embedder = CSMEmbedder(
        in_dim=config['parcellation_dim'],
        embed_dim=config['embed_dim'],
        dropout=config['dropout'],
        max_seq_len=config['num_chunks'],
        pos_encoding_type=config.get('pos_encoding_type', 'learned')
    )

    # Build decoder
    from src.decoder.transformer import TransformerDecoder

    decoder = TransformerDecoder(
        embed_dim=config['embed_dim'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        ff_dim=config['ff_dim'],
        dropout=config['dropout'],
        max_seq_len=config['num_chunks'],
        output_dim=config['parcellation_dim']  # Project back to input space
    )

    # Build full model
    model = FoundationModel(encoder, embedder, decoder)

    return model
