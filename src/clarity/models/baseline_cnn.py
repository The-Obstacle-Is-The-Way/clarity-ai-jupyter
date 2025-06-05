from typing import Optional

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
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(16)
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.pool2 = nn.MaxPool1d(2)

        # We'll calculate flattened size dynamically in forward pass
        # Initialize with a placeholder that will be replaced in the first forward pass
        self.fc1: Optional[nn.Linear] = None
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

        # Calculate flattened size dynamically based on actual shape after convolutions
        batch_size = x.size(0)
        flattened_size = x.size(1) * x.size(2)  # channels * time points after pooling

        # If fc1 is None or has wrong input size, create a new one
        if self.fc1 is None or self.fc1.in_features != flattened_size:
            # Create new fc1 layer with correct size
            self.fc1 = nn.Linear(flattened_size, 128).to(x.device)

        x = x.view(batch_size, flattened_size)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
