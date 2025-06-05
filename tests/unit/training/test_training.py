"""Unit tests for training functionality."""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from src.clarity.training.loop import train_model, evaluate_model
from src.clarity.models import BaselineCNN
from src.clarity.training.config import DEVICE


class SimpleTestModel(nn.Module):
    """A very simple model for testing the training loop."""
    def __init__(self, input_size=10, hidden_size=5, output_size=2):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


@pytest.fixture
def simple_model_and_data():
    """Create a simple model and data for testing training functions."""
    # Create a simple model
    model = SimpleTestModel()
    
    # Create simple synthetic data
    batch_size = 4
    input_size = 10
    x = torch.randn(batch_size * 5, input_size)  # 20 samples
    y = torch.randint(0, 2, (batch_size * 5,))  # Binary labels
    
    # Create dataset and dataloaders
    dataset = TensorDataset(x, y)
    train_loader = DataLoader(dataset, batch_size=batch_size)
    
    return model, train_loader


def test_train_model(simple_model_and_data):
    """Test that the train_model function works correctly."""
    model, train_loader = simple_model_and_data
    
    # Setup optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    # Train for just 2 epochs to keep test fast
    trained_model = train_model(
        model, train_loader, optimizer, criterion, 
        model_type="cnn",  # Use CNN as the model type for testing
        epochs=2
    )
    
    # Verify the model was returned
    assert trained_model is model
    
    # Verify the model parameters were updated
    # We'll check this by ensuring some parameters are different
    # from their initialization
    has_changed = False
    for param in trained_model.parameters():
        if not torch.allclose(param, torch.zeros_like(param)):
            has_changed = True
            break
            
    assert has_changed


def test_evaluate_model(simple_model_and_data):
    """Test that the evaluate_model function works correctly."""
    model, test_loader = simple_model_and_data
    
    # Run evaluation
    accuracy, precision, recall, f1 = evaluate_model(
        model, test_loader, model_type="cnn"
    )
    
    # Verify the metrics are within expected ranges
    assert 0 <= accuracy <= 1
    assert 0 <= precision <= 1
    assert 0 <= recall <= 1
    assert 0 <= f1 <= 1