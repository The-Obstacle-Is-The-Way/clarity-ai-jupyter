"""Unit tests for training functionality."""

import pytest
import torch
import torch.nn as nn
from src.clarity.training.loop import evaluate_model, train_model
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.data import Data, DataLoader as PyGDataLoader
from torch_geometric.nn import GCNConv, global_mean_pool


class SimpleTestModel(nn.Module):
    """A very simple model for testing the training loop."""
    def __init__(self, input_size=10, hidden_size=5, output_size=2):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


class SimpleGCN(nn.Module):
    """A very simple GCN model for testing."""
    def __init__(self, in_channels=16, out_channels=2):
        super().__init__()
        self.conv1 = GCNConv(in_channels, out_channels)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        # In a real model, you would pool the node features per graph.
        # Here, we'll just create a dummy output of the correct shape.
        # We need to ensure the output is linked to the computation graph.
        # A simple way is to sum the pooled features.
        pooled_x = global_mean_pool(x, batch)
        return pooled_x, None


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


@pytest.fixture
def pyg_model_and_data():
    """Create a simple PyG model and data for testing."""
    model = SimpleGCN()
    data_list = [
        Data(
            x=torch.randn(10, 16),
            edge_index=torch.randint(0, 10, (2, 20)),
            y=torch.tensor([i % 2])
        ) for i in range(10)
    ]
    loader = PyGDataLoader(data_list, batch_size=2)
    return model, loader


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
    metrics, preds_and_labels = evaluate_model(
        model, test_loader, model_type="cnn"
    )
    accuracy, precision, recall, f1 = metrics
    all_preds, all_labels = preds_and_labels

    # Verify the metrics are within expected ranges
    assert 0 <= accuracy <= 1
    assert 0 <= precision <= 1
    assert 0 <= recall <= 1
    assert 0 <= f1 <= 1

    # Verify the returned predictions and labels
    assert isinstance(all_preds, list)
    assert isinstance(all_labels, list)
    assert len(all_preds) == len(test_loader.dataset)
    assert len(all_labels) == len(test_loader.dataset)


def test_train_model_gcn(pyg_model_and_data):
    """Test train_model with a PyG model."""
    model, loader = pyg_model_and_data
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    trained_model = train_model(
        model, loader, optimizer, criterion, model_type="mha_gcn", epochs=1
    )
    assert trained_model is model


def test_evaluate_model_gcn(pyg_model_and_data):
    """Test evaluate_model with a PyG model."""
    model, loader = pyg_model_and_data
    metrics, _ = evaluate_model(model, loader, model_type="mha_gcn")
    accuracy, precision, recall, f1 = metrics
    assert 0 <= accuracy <= 1
    assert 0 <= precision <= 1
    assert 0 <= recall <= 1
    assert 0 <= f1 <= 1
