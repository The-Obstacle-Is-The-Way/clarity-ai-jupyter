"""Unit tests for the neural network model architectures."""

import pytest
import torch
import numpy as np
from src.clarity.models import BaselineCNN, MHA_GCN
from src.clarity.training.config import CHANNELS_29, NUM_CLASSES


def test_baseline_cnn_forward():
    """Test that the BaselineCNN model properly processes input data."""
    # Define test parameters
    batch_size = 4
    in_channels = len(CHANNELS_29)
    time_points = 250  # Typical EEG segment length
    
    # Create a random input tensor
    x = torch.randn(batch_size, in_channels, time_points)
    
    # Initialize the model
    model = BaselineCNN(in_channels=in_channels, num_classes=NUM_CLASSES)
    
    # Set to evaluation mode
    model.eval()
    
    # Forward pass
    with torch.no_grad():
        output = model(x)
    
    # Check output shape
    assert output.shape == (batch_size, NUM_CLASSES)
    
    # Check output is valid (contains proper logits)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


def test_mha_gcn_forward():
    """Test that the MHA_GCN model properly processes input data."""
    # Define test parameters
    node_feature_dim = 64  # Example dimension
    num_nodes = len(CHANNELS_29)
    
    # Create random input tensors for a single graph
    # Node features should be of shape (num_nodes, node_feature_dim)
    node_features = torch.randn(num_nodes, node_feature_dim)
    adj_matrix = torch.randn(num_nodes, num_nodes)
    
    # Make adjacency matrix symmetric as it would be in real usage
    adj_matrix = (adj_matrix + adj_matrix.T) / 2
    
    # Initialize the model
    model = MHA_GCN(node_feature_dim=node_feature_dim, num_classes=NUM_CLASSES)
    
    # Set to evaluation mode
    model.eval()
    
    # Forward pass
    with torch.no_grad():
        output = model(node_features, adj_matrix)
    
    # Check output shape (a single prediction for the whole graph)
    assert output.shape == (NUM_CLASSES,)
    
    # Check output is valid (contains proper logits)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


def test_model_parameter_gradients():
    """Test that model parameters properly receive gradients during training."""
    # Create a small batch of data
    batch_size = 2
    in_channels = len(CHANNELS_29)
    time_points = 250
    
    # Input and target for BaselineCNN
    x = torch.randn(batch_size, in_channels, time_points)
    target = torch.randint(0, NUM_CLASSES, (batch_size,))
    
    # Initialize model and optimizer
    model = BaselineCNN(in_channels=in_channels, num_classes=NUM_CLASSES)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Training step
    model.train()
    optimizer.zero_grad()
    output = model(x)
    loss = criterion(output, target)
    loss.backward()
    
    # Check that all parameters have gradients
    for name, param in model.named_parameters():
        assert param.grad is not None, f"Parameter {name} has no gradient"
        # Ensure gradients are not zero (extremely unlikely for all to be exactly zero)
        assert not torch.allclose(param.grad, torch.zeros_like(param.grad))