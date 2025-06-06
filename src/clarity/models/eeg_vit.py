import timm
import torch.nn as nn

from ...clarity.training.config import NUM_CLASSES


class SpectrogramViT(nn.Module):
    """
    Vision Transformer (ViT) model for classifying EEG spectrograms.

    This class wraps a pre-trained ViT model from the `timm` library and
    replaces its classifier head with a new one suitable for the number of
    EEG classes.
    """

    def __init__(self, num_classes: int = NUM_CLASSES, pretrained: bool = True):
        """
        Initializes the SpectrogramViT model.

        Args:
            num_classes: The number of output classes.
            pretrained: Whether to load a model pre-trained on ImageNet.
        """
        super().__init__()
        # Load a pre-trained Vision Transformer model
        # 'vit_base_patch16_224' is a standard choice.
        self.vit = timm.create_model(
            "vit_base_patch16_224", pretrained=pretrained, num_classes=num_classes
        )

    def forward(self, x):
        """
        Defines the forward pass for the SpectrogramViT.

        Args:
            x: A batch of spectrograms with shape (batch_size, 3, 224, 224).

        Returns:
            The output logits from the model.
        """
        return self.vit(x) 