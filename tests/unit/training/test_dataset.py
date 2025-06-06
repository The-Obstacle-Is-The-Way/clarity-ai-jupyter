"""Unit tests for the CustomEEGDataset class."""


import numpy as np
import torch
from src.clarity.training.loop import CustomEEGDataset
from torch_geometric.data import Data


def test_custom_eeg_dataset_initialization(sample_epochs, subject_labels):
    """Test that the CustomEEGDataset initializes correctly."""
    # Mock subject_ids
    subject_ids = list(subject_labels.keys())[:2]  # Take first two subjects

    # Initialize dataset for CNN model type
    dataset_cnn = CustomEEGDataset(subject_ids, subject_labels, model_type="cnn")

    # Verify dataset properties
    assert dataset_cnn.model_type == "cnn"
    assert dataset_cnn.subject_ids == subject_ids
    assert dataset_cnn.labels_dict == subject_labels
    assert len(dataset_cnn.data) >= 0  # Should have data or be empty list if no data found
    assert len(dataset_cnn.labels) >= 0  # Should have labels or be empty list


def test_custom_eeg_dataset_len(sample_epochs, subject_labels):
    """Test that the dataset __len__ method works correctly."""
    # Mock subject_ids where data is ensured to be available from fixtures
    subject_ids = list(subject_labels.keys())[:2]

    # Create a CustomEEGDataset with mocked data
    dataset = CustomEEGDataset(subject_ids, subject_labels, model_type="cnn")

    # Manually set some test data to ensure __len__ works
    dataset.data = [np.random.randn(29, 250) for _ in range(5)]
    dataset.labels = [1, 1, 0, 0, 1]

    # Check length
    assert len(dataset) == 5


def test_custom_eeg_dataset_getitem_cnn(sample_epochs, subject_labels):
    """Test that the dataset __getitem__ method works correctly for CNN model type."""
    # Mock subject_ids
    subject_ids = list(subject_labels.keys())[:2]

    # Create dataset
    dataset = CustomEEGDataset(subject_ids, subject_labels, model_type="cnn")

    # Manually set some test data to ensure __getitem__ works
    dataset.data = [np.random.randn(1, 29, 250) for _ in range(3)]
    dataset.labels = [1, 0, 1]

    # Get an item
    # Since we overloaded 'get', we simulate the tuple return for this test
    data_point, label_val = dataset.data[1], dataset.labels[1]
    data = torch.FloatTensor(data_point)
    label = torch.tensor(label_val, dtype=torch.long)

    # Check types and shapes
    assert isinstance(data, torch.Tensor)
    assert isinstance(label, torch.Tensor)
    assert data.shape == (1, 29, 250)
    assert label.item() == 0  # Should match the label we set


def test_custom_eeg_dataset_getitem_mha_gcn(subject_labels):
    """Test that the dataset __getitem__ method works correctly for the MHA-GCN model type."""
    # Mock subject_ids
    subject_ids = list(subject_labels.keys())[:2]

    # Create dataset
    dataset = CustomEEGDataset(subject_ids, subject_labels, model_type="mha_gcn")

    # Manually create and set a PyG Data object
    node_features = torch.randn(29, 11520) # (num_nodes, feature_dim)
    edge_index = torch.randint(0, 29, (2, 50)) # Dummy edge index
    label = torch.tensor(1, dtype=torch.long)
    graph_data = Data(x=node_features, edge_index=edge_index, y=label)

    dataset.data = [graph_data]
    dataset.labels = [1] # Kept for len(), though y is in graph_data

    # Get the item, which should be the Data object itself
    retrieved_item = dataset[0]

    # Check that the retrieved item is a PyG Data object and has the correct attributes
    assert isinstance(retrieved_item, Data)
    assert 'x' in retrieved_item
    assert 'edge_index' in retrieved_item
    assert 'y' in retrieved_item
    assert retrieved_item.x is not None and torch.equal(retrieved_item.x, node_features)
    assert retrieved_item.edge_index is not None and torch.equal(retrieved_item.edge_index, edge_index)
    assert isinstance(retrieved_item.y, torch.Tensor)
    assert retrieved_item.y.item() == 1
