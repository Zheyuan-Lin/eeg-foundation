"""
Main training script.
Simple and minimal - just train the model.
"""

import torch
import argparse
from config import get_config, update_config
from model import build_model
from data.dataset import create_dataloaders
from trainer.trainer import Trainer


def main():
    """Main training function."""

    # Parse arguments
    parser = argparse.ArgumentParser(description='Train EEG Foundation Model')
    parser.add_argument('--data-path', type=str, default='./data',
                        help='Path to EEG data directory')
    parser.add_argument('--log-dir', type=str, default='./logs',
                        help='Directory for logs and checkpoints')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--num-epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--no-encoder', action='store_true',
                        help='Skip encoder (use raw chunks)')

    args = parser.parse_args()

    # Get config
    config = get_config()

    # Update with CLI args
    config = update_config(
        config,
        data_path=args.data_path,
        log_dir=args.log_dir,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        device=args.device,
        use_encoder=not args.no_encoder
    )

    print('=' * 60)
    print('EEG Foundation Model - Core Skeleton')
    print('=' * 60)
    print('\nConfiguration:')
    for key, value in config.items():
        print(f'  {key}: {value}')
    print()

    # Set random seed
    torch.manual_seed(42)

    # Create dataloaders
    print('Loading data...')
    train_loader, val_loader = create_dataloaders(config)

    # Build model
    print('\nBuilding model...')
    model = build_model(config)

    # Create trainer
    print('\nInitializing trainer...')
    trainer = Trainer(model, train_loader, val_loader, config)

    # Train
    trainer.train()


if __name__ == '__main__':
    main()
