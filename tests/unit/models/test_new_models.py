import unittest
import torch
from src.clarity.models import EEGNet, SpectrogramViT
from src.clarity.training.config import NUM_CLASSES, CHANNELS_29

class TestNewModels(unittest.TestCase):

    def test_eegnet_forward_pass(self):
        """Test the forward pass of the EEGNet model."""
        batch_size = 4
        num_channels = len(CHANNELS_29)
        sequence_length = 500  # 2 seconds * 250 Hz
        
        model = EEGNet(num_classes=NUM_CLASSES, in_channels=num_channels)
        model.eval() # Set to evaluation mode

        # Create a dummy input tensor
        dummy_input = torch.randn(batch_size, num_channels, sequence_length)
        
        with torch.no_grad():
            output = model(dummy_input)
            
        # Check output shape
        self.assertEqual(output.shape, (batch_size, NUM_CLASSES))

    def test_spectrogram_vit_forward_pass(self):
        """Test the forward pass of the SpectrogramViT model."""
        batch_size = 2
        # ViT model from timm expects a 3-channel image of size 224x224
        img_height = 224
        img_width = 224
        
        # We test with pretrained=False to avoid downloading weights during testing
        model = SpectrogramViT(num_classes=NUM_CLASSES, pretrained=False)
        model.eval()

        # Create a dummy input tensor
        dummy_input = torch.randn(batch_size, 3, img_height, img_width)
        
        with torch.no_grad():
            output = model(dummy_input)
            
        # Check output shape
        self.assertEqual(output.shape, (batch_size, NUM_CLASSES))

if __name__ == '__main__':
    unittest.main() 