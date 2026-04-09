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

        # Setup learning rate scheduler
        self.scheduler = None
        if config.get('use_scheduler', False):
            total_steps = len(train_loader) * config['num_epochs']
            warmup_steps = int(total_steps * config.get('warmup_ratio', 0.1))

            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps - warmup_steps,
                eta_min=config.get('min_lr', 1e-6)
            )
            self.warmup_steps = warmup_steps
            self.total_steps = total_steps
            self.current_step = 0

        # Setup logging
        os.makedirs(config['log_dir'], exist_ok=True)

        print(f'Device: {self.device}')
        print(f'Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M')

    def train(self):
        """Run training loop."""
        print(f'\nStarting training for {self.config["num_epochs"]} epochs...\n')

        global_step = 0
        best_val_loss = float('inf')
        best_epoch = 0

        for epoch in range(self.config['num_epochs']):
            # Training
            train_loss = self._train_epoch(epoch, global_step)

            # Validation
            val_loss = self._validate()

            print(f'Epoch {epoch+1}/{self.config["num_epochs"]} - '
                  f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

            # Save checkpoint with full state
            checkpoint_path = os.path.join(
                self.config['log_dir'],
                f'checkpoint_epoch_{epoch+1}.pt'
            )
            self.model.save_checkpoint(
                checkpoint_path,
                epoch=epoch+1,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                train_loss=train_loss,
                val_loss=val_loss
            )

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                best_path = os.path.join(self.config['log_dir'], 'best_model.pt')
                self.model.save_checkpoint(
                    best_path,
                    epoch=epoch+1,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    train_loss=train_loss,
                    val_loss=val_loss
                )
                print(f'  → Saved new best model (val_loss: {val_loss:.4f})')

            # Keep only best N checkpoints
            self._cleanup_checkpoints()

        # Save final model
        final_path = os.path.join(self.config['log_dir'], 'model_final.pt')
        self.model.save_checkpoint(
            final_path,
            epoch=self.config['num_epochs'],
            optimizer=self.optimizer,
            scheduler=self.scheduler
        )
        print(f'\nTraining complete!')
        print(f'Best model at epoch {best_epoch} with val_loss: {best_val_loss:.4f}')
        print(f'Final model saved to {final_path}')

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

            # Gradient clipping
            if self.config.get('clip_grad_norm', None):
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['clip_grad_norm']
                )
            else:
                # Compute gradient norm for logging even without clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    float('inf')
                )

            self.optimizer.step()

            # Update learning rate
            if self.scheduler is not None:
                self.current_step += 1

                # Warmup phase: linear increase
                if self.current_step <= self.warmup_steps:
                    warmup_lr = self.config['learning_rate'] * (self.current_step / self.warmup_steps)
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = warmup_lr
                else:
                    # Cosine annealing phase
                    self.scheduler.step()

            # Logging
            total_loss += loss.item()
            current_lr = self.optimizer.param_groups[0]['lr']
            log_dict = {'loss': loss.item(), 'lr': f'{current_lr:.2e}'}

            # Log gradient norm
            if self.config.get('log_grad_norm', False):
                log_dict['grad_norm'] = f'{grad_norm:.2f}'

            pbar.set_postfix(log_dict)

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

    def _cleanup_checkpoints(self):
        """Keep only the best N checkpoints to save disk space."""
        max_checkpoints = self.config.get('max_checkpoints', 3)

        # Get all checkpoint files (excluding best_model and model_final)
        checkpoints = [
            f for f in os.listdir(self.config['log_dir'])
            if f.startswith('checkpoint_epoch_') and f.endswith('.pt')
        ]

        if len(checkpoints) > max_checkpoints:
            # Sort by modification time (oldest first)
            checkpoints.sort(key=lambda x: os.path.getmtime(
                os.path.join(self.config['log_dir'], x)
            ))

            # Remove oldest checkpoints
            for checkpoint in checkpoints[:-max_checkpoints]:
                os.remove(os.path.join(self.config['log_dir'], checkpoint))
