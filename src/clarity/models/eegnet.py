import torch
import torch.nn as nn

from ...clarity.training.config import CHANNELS_29, NUM_CLASSES, SAMPLING_RATE


class EEGNet(nn.Module):
    """
    EEGNet: A Compact Convolutional Network for EEG-based Brain-Computer Interfaces.

    This implementation is adapted from the original paper and its PyTorch versions.
    It is configured to work with the data shape from the MODMA dataset pipeline.
    """
    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        in_channels: int = len(CHANNELS_29),
        sampling_rate: int = SAMPLING_RATE,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        kernel_1_size: int = 64,
        kernel_2_size: int = 16,
        dropout_rate: float = 0.5
    ):
        super().__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.sampling_rate = sampling_rate

        # Block 1: Temporal and Depthwise Convolutions
        self.conv1 = nn.Conv2d(
            1, F1, (1, kernel_1_size), padding=(0, kernel_1_size // 2), bias=False
        )
        self.bn1 = nn.BatchNorm2d(F1)
        self.depthwise_conv = nn.Conv2d(
            F1, F1 * D, (in_channels, 1), groups=F1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.avg_pool1 = nn.AvgPool2d((1, 4))
        self.dropout1 = nn.Dropout(p=dropout_rate)

        # Block 2: Separable Convolution
        self.separable_conv = nn.Conv2d(
            F1 * D, F2, (1, kernel_2_size), padding=(0, kernel_2_size // 2), bias=False
        )
        self.bn3 = nn.BatchNorm2d(F2)
        self.avg_pool2 = nn.AvgPool2d((1, 8))
        self.dropout2 = nn.Dropout(p=dropout_rate)

        # Fully Connected Layer
        # The flattened size calculation depends on the output of the last
        # pooling layer.
        # Let's calculate it based on the architecture.
        # Input samples: WINDOW_SIZE * SAMPLING_RATE = 2 * 250 = 500
        # After pool1 (stride 4): 500 / 4 = 125
        final_conv_length = sampling_rate * 2 // 4 // 8
        self.fc1 = nn.Linear(F2 * final_conv_length, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reshape input from (batch, channels, time) to (batch, 1, channels, time)
        if len(x.shape) == 3:
            x = x.unsqueeze(1)

        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.depthwise_conv(x)
        x = self.bn2(x)
        x = nn.functional.elu(x)
        x = self.avg_pool1(x)
        x = self.dropout1(x)

        # Block 2
        x = self.separable_conv(x)
        x = self.bn3(x)
        x = nn.functional.elu(x)
        x = self.avg_pool2(x)
        x = self.dropout2(x)

        # Flatten and Classify
        x = x.view(x.size(0), -1)
        x = self.fc1(x)

        return x
