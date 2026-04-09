"""
Simple EEG dataset.
Loads preprocessed .pt files and creates chunks.
"""

import torch
from torch.utils.data import Dataset
import os


class EEGDataset(Dataset):
    """
    Simple EEG dataset.

    Loads .pt files containing EEG data and creates overlapping chunks.

    Args:
        data_path: Path to directory containing .pt files
        chunk_len: Length of each chunk
        num_chunks: Number of chunks per sample
        chunk_overlap: Overlap between consecutive chunks
        num_channels: Number of EEG channels
    """

    def __init__(
        self,
        data_path,
        chunk_len=500,
        num_chunks=34,
        chunk_overlap=100,
        num_channels=20,
        normalization='minmax',  # Options: 'minmax', 'zscore', 'none'
        augmentation=False,      # Enable/disable augmentation
        aug_prob=0.5,            # Probability of applying augmentation
        aug_noise_std=0.1,       # Gaussian noise std
        aug_dropout_prob=0.1,    # Channel dropout probability
        aug_scale_range=(0.9, 1.1)  # Amplitude scaling range
    ):
        self.data_path = data_path
        self.chunk_len = chunk_len
        self.num_chunks = num_chunks
        self.chunk_overlap = chunk_overlap
        self.num_channels = num_channels
        self.normalization = normalization

        # Augmentation settings
        self.augmentation = augmentation
        self.aug_prob = aug_prob
        self.aug_noise_std = aug_noise_std
        self.aug_dropout_prob = aug_dropout_prob
        self.aug_scale_range = aug_scale_range

        # Get list of .pt files
        self.files = [
            f for f in os.listdir(data_path)
            if f.endswith('.pt')
        ]

        if len(self.files) == 0:
            raise ValueError(f'No .pt files found in {data_path}')

        print(f'Found {len(self.files)} EEG files')

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        """
        Load EEG file and create chunks.

        Returns:
            chunks: (num_chunks, num_channels, chunk_len)
            attention_mask: (num_chunks,) - 1 for real chunks, 0 for padding
        """
        # Load .pt file
        file_path = os.path.join(self.data_path, self.files[idx])
        eeg_data = torch.load(file_path)  # (num_channels, time_samples)

        # Ensure correct shape
        if eeg_data.dim() == 1:
            # If 1D, reshape assuming it's flattened (channels, time)
            total_samples = eeg_data.shape[0]
            time_samples = total_samples // self.num_channels
            eeg_data = eeg_data.reshape(self.num_channels, time_samples)

        # Apply normalization
        if self.normalization != 'none':
            eeg_data = self._normalize(eeg_data)

        # Apply augmentation (if enabled and with probability)
        if self.augmentation and torch.rand(1).item() < self.aug_prob:
            eeg_data = self._augment(eeg_data)

        # Create overlapping chunks
        chunks, attention_mask = self._create_chunks(eeg_data)

        return {
            'chunks': chunks,
            'attention_mask': attention_mask
        }

    def _create_chunks(self, eeg_data):
        """
        Create overlapping chunks from continuous EEG.

        Args:
            eeg_data: (num_channels, time_samples)

        Returns:
            chunks: (num_chunks, num_channels, chunk_len)
            attention_mask: (num_chunks,)
        """
        num_channels, time_samples = eeg_data.shape

        # Calculate stride
        stride = self.chunk_len - self.chunk_overlap

        # Create chunks
        chunks = []
        for i in range(self.num_chunks):
            start = i * stride
            end = start + self.chunk_len

            if end <= time_samples:
                chunk = eeg_data[:, start:end]
            else:
                # Pad if we run out of data
                chunk = torch.zeros(num_channels, self.chunk_len)
                available = max(0, time_samples - start)
                if available > 0:
                    chunk[:, :available] = eeg_data[:, start:time_samples]

            chunks.append(chunk)

        chunks = torch.stack(chunks)  # (num_chunks, num_channels, chunk_len)

        # Create attention mask
        attention_mask = torch.ones(self.num_chunks)

        # Mark padded chunks
        last_valid = min(self.num_chunks, (time_samples - self.chunk_len) // stride + 1)
        if last_valid < self.num_chunks:
            attention_mask[last_valid:] = 0

        return chunks, attention_mask

    def _normalize(self, eeg_data):
        """
        Normalize EEG data per channel.

        Args:
            eeg_data: (num_channels, time_samples)

        Returns:
            Normalized EEG data with same shape
        """
        if self.normalization == 'minmax':
            # Scale to [-1, 1] per channel
            min_val = eeg_data.min(dim=-1, keepdim=True)[0]
            max_val = eeg_data.max(dim=-1, keepdim=True)[0]
            # Avoid division by zero
            range_val = max_val - min_val
            range_val = torch.where(range_val == 0, torch.ones_like(range_val), range_val)
            normalized = 2 * (eeg_data - min_val) / range_val - 1
            return normalized

        elif self.normalization == 'zscore':
            # Z-score normalization per channel
            mean = eeg_data.mean(dim=-1, keepdim=True)
            std = eeg_data.std(dim=-1, keepdim=True)
            # Avoid division by zero
            std = torch.where(std == 0, torch.ones_like(std), std)
            normalized = (eeg_data - mean) / std
            return normalized

        else:
            return eeg_data

    def _augment(self, eeg_data):
        """
        Apply data augmentation to EEG data.

        Args:
            eeg_data: (num_channels, time_samples)

        Returns:
            Augmented EEG data with same shape
        """
        augmented = eeg_data.clone()

        # 1. Gaussian noise injection
        if torch.rand(1).item() < 0.5:
            noise = torch.randn_like(augmented) * self.aug_noise_std
            augmented = augmented + noise

        # 2. Channel dropout (randomly zero out some channels)
        if torch.rand(1).item() < 0.5:
            num_channels = augmented.shape[0]
            dropout_mask = torch.rand(num_channels) > self.aug_dropout_prob
            augmented = augmented * dropout_mask.unsqueeze(-1)

        # 3. Amplitude scaling
        if torch.rand(1).item() < 0.5:
            scale = torch.FloatTensor(1).uniform_(*self.aug_scale_range).item()
            augmented = augmented * scale

        # 4. Time jittering (small temporal shifts)
        if torch.rand(1).item() < 0.5:
            max_shift = min(50, eeg_data.shape[1] // 20)  # Max 5% shift
            shift = torch.randint(-max_shift, max_shift + 1, (1,)).item()
            if shift != 0:
                augmented = torch.roll(augmented, shifts=shift, dims=1)

        return augmented


def create_dataloaders(config, train_split=0.8):
    """
    Create train and validation dataloaders.

    Args:
        config: Configuration dictionary
        train_split: Fraction of data for training

    Returns:
        train_loader, val_loader
    """
    from torch.utils.data import DataLoader, random_split

    # Create training dataset (with augmentation)
    train_dataset_full = EEGDataset(
        data_path=config['data_path'],
        chunk_len=config['chunk_len'],
        num_chunks=config['num_chunks'],
        chunk_overlap=config['chunk_overlap'],
        num_channels=config['num_channels'],
        normalization=config.get('normalization', 'minmax'),
        augmentation=config.get('augmentation', False),
        aug_prob=config.get('aug_prob', 0.5),
        aug_noise_std=config.get('aug_noise_std', 0.1),
        aug_dropout_prob=config.get('aug_dropout_prob', 0.1),
        aug_scale_range=config.get('aug_scale_range', (0.9, 1.1))
    )

    # Create validation dataset (NO augmentation)
    val_dataset_full = EEGDataset(
        data_path=config['data_path'],
        chunk_len=config['chunk_len'],
        num_chunks=config['num_chunks'],
        chunk_overlap=config['chunk_overlap'],
        num_channels=config['num_channels'],
        normalization=config.get('normalization', 'minmax'),
        augmentation=False  # Never augment validation data
    )

    # Get total dataset size
    total_size = len(train_dataset_full)

    # Create indices for split
    train_size = int(total_size * train_split)
    val_size = total_size - train_size

    indices = torch.randperm(total_size).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    # Create subsets
    from torch.utils.data import Subset
    train_dataset = Subset(train_dataset_full, train_indices)
    val_dataset = Subset(val_dataset_full, val_indices)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0  # Set to 0 to avoid dataloader stalls
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0
    )

    print(f'Train samples: {train_size}, Val samples: {val_size}')

    return train_loader, val_loader
