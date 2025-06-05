import torch
import torch.nn as nn
import torch.nn.functional as F


class BaselineCNN(nn.Module):
    """A baseline 1D CNN model for EEG signal classification."""
    def __init__(self, in_channels: int = 29, num_classes: int = 4):
        """Initializes the BaselineCNN model.

        Args:
            in_channels: Number of input channels (EEG electrodes).
            num_classes: Number of output classes for classification.
        """
        # Import constants locally to avoid circular imports
        from src.clarity.training.config import SAMPLING_RATE, WINDOW_SIZE
        
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(16)
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.pool2 = nn.MaxPool1d(2)

        # Calculate flattened size after two MaxPool1d(2) layers
        self.flattened_size = 32 * (SAMPLING_RATE * WINDOW_SIZE // 4)
        self.fc1 = nn.Linear(self.flattened_size, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Defines the forward pass of the BaselineCNN model.

        Args:
            x: Input tensor of shape (batch_size, in_channels, sequence_length).

        Returns:
            Output tensor of shape (batch_size, num_classes).
        """
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, self.flattened_size)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
