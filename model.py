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

    def save_checkpoint(self, path):
        """Save model checkpoint."""
        torch.save({
            'encoder': self.encoder.state_dict() if self.encoder else None,
            'embedder': self.embedder.state_dict(),
            'decoder': self.decoder.state_dict(),
        }, path)
        print(f'Saved checkpoint to {path}')

    def load_checkpoint(self, path):
        """Load model checkpoint."""
        checkpoint = torch.load(path)

        if self.encoder and checkpoint['encoder']:
            self.encoder.load_state_dict(checkpoint['encoder'])
        self.embedder.load_state_dict(checkpoint['embedder'])
        self.decoder.load_state_dict(checkpoint['decoder'])

        print(f'Loaded checkpoint from {path}')


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
        from encoder.simple_encoder import SimpleEncoder, ChunkEncoder

        base_encoder = SimpleEncoder(
            n_channels=config['num_channels'],
            chunk_len=config['chunk_len'],
            n_filters=config['n_filters'],
            filter_len=config['filter_len'],
            pool_len=config['pool_len'],
            pool_stride=config['pool_stride']
        )
        encoder = ChunkEncoder(base_encoder)
    else:
        encoder = None

    # Build embedder
    from embedder.csm_embedder import CSMEmbedder

    embedder = CSMEmbedder(
        in_dim=config['parcellation_dim'],
        embed_dim=config['embed_dim'],
        dropout=config['dropout']
    )

    # Build decoder
    from decoder.transformer import TransformerDecoder

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
