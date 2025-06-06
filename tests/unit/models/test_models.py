"""Unit tests for the neural network model architectures."""

import torch
from src.clarity.models import MHA_GCN, BaselineCNN
from src.clarity.training.config import CHANNELS_29, NUM_CLASSES
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader


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
    """Test that the MHA_GCN model properly processes a batch of graph data."""
    # Define test parameters
    node_feature_dim = 64
    num_nodes = len(CHANNELS_29)
    num_classes = NUM_CLASSES
    batch_size = 4
    mha_heads = 4

    # Create a list of PyG Data objects to form a batch
    data_list = []
    for _ in range(batch_size):
        node_features = torch.randn(num_nodes, node_feature_dim)
        # Create a random adjacency matrix and convert to edge_index
        adj = torch.randint(0, 2, (num_nodes, num_nodes)).float()
        edge_index = adj.nonzero().t().contiguous()
        data_list.append(Data(x=node_features, edge_index=edge_index))

    # Use PyG's DataLoader to create a batch
    loader = PyGDataLoader(data_list, batch_size=batch_size)
    batch = next(iter(loader))

    # Initialize the model
    model = MHA_GCN(
        node_feature_dim=node_feature_dim,
        num_classes=num_classes,
        mha_heads=mha_heads
    )
    model.eval()

    # Forward pass with the batched data
    with torch.no_grad():
        logits, attention_weights = model(batch.x, batch.edge_index, batch.batch)

    # --- Check Logits ---
    # The output shape should now be (batch_size, num_classes)
    assert logits.shape == (batch_size, num_classes)
    assert not torch.isnan(logits).any()
    assert not torch.isinf(logits).any()

    # --- Check Attention Weights ---
    assert attention_weights is not None
    # The MHA layer was applied to the entire batch of nodes as one sequence
    # Input to MHA is (1, total_nodes, features), so output attention is (1, total_nodes, total_nodes)
    total_nodes = num_nodes * batch_size
    assert attention_weights.shape == (1, total_nodes, total_nodes)


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
