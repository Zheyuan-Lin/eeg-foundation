"""
Simple training loop.
No fancy features - just basic training.
"""

import torch
import os
from tqdm import tqdm


class Trainer:
    """
    Simple trainer for foundation model.

    Args:
        model: FoundationModel instance
        train_loader: Training dataloader
        val_loader: Validation dataloader
        config: Configuration dictionary
    """

    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        # Setup device
        self.device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        # Setup optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config['learning_rate']
        )

        # Setup logging
        os.makedirs(config['log_dir'], exist_ok=True)

        print(f'Device: {self.device}')
        print(f'Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M')

    def train(self):
        """Run training loop."""
        print(f'\nStarting training for {self.config["num_epochs"]} epochs...\n')

        global_step = 0

        for epoch in range(self.config['num_epochs']):
            # Training
            train_loss = self._train_epoch(epoch, global_step)

            # Validation
            val_loss = self._validate()

            print(f'Epoch {epoch+1}/{self.config["num_epochs"]} - '
                  f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

            # Save checkpoint
            checkpoint_path = os.path.join(
                self.config['log_dir'],
                f'checkpoint_epoch_{epoch+1}.pt'
            )
            self.model.save_checkpoint(checkpoint_path)

        # Save final model
        final_path = os.path.join(self.config['log_dir'], 'model_final.pt')
        self.model.save_checkpoint(final_path)
        print(f'\nTraining complete! Final model saved to {final_path}')

    def _train_epoch(self, epoch, global_step):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}')

        for batch_idx, batch in enumerate(pbar):
            chunks = batch['chunks'].to(self.device)

            # Forward pass
            loss = self.model.compute_loss(chunks)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Logging
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})

            global_step += 1

            # Periodic logging
            if global_step % self.config['log_every'] == 0:
                avg_loss = total_loss / (batch_idx + 1)
                print(f'\nStep {global_step} - Loss: {avg_loss:.4f}')

        return total_loss / len(self.train_loader)

    def _validate(self):
        """Run validation."""
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch in self.val_loader:
                chunks = batch['chunks'].to(self.device)
                loss = self.model.compute_loss(chunks)
                total_loss += loss.item()

        return total_loss / len(self.val_loader) if len(self.val_loader) > 0 else 0
