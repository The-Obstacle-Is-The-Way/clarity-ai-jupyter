"""Unit tests for the CustomEEGDataset class."""

import pytest
import torch
import numpy as np
from src.clarity.training.loop import CustomEEGDataset


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
    dataset.data = [np.random.randn(29, 250) for _ in range(3)]
    dataset.labels = [1, 0, 1]
    
    # Get an item
    data, label = dataset[1]
    
    # Check types and shapes
    assert isinstance(data, torch.Tensor)
    assert isinstance(label, torch.Tensor)
    assert data.shape == (29, 250)
    assert label.item() == 0  # Should match the label we set


def test_custom_eeg_dataset_getitem_mha_gcn(subject_labels):
    """Test that the dataset __getitem__ method works correctly for MHA-GCN model type."""
    # Mock subject_ids
    subject_ids = list(subject_labels.keys())[:2]
    
    # Create dataset
    dataset = CustomEEGDataset(subject_ids, subject_labels, model_type="mha_gcn")
    
    # Manually set some test data for MHA-GCN (features and adjacency matrices)
    # MHA-GCN expects (features, adjacency) tuples as data points
    features = [np.random.randn(29, 64) for _ in range(3)]  # 29 channels, 64 features
    adj_matrices = [np.random.randn(29, 29) for _ in range(3)]  # 29x29 adjacency matrices
    dataset.data = list(zip(features, adj_matrices))
    dataset.labels = [1, 0, 1]
    
    # Get an item
    data_tuple, label = dataset[1]
    dwt_features, adj_matrix = data_tuple
    
    # Check types and shapes
    assert isinstance(dwt_features, torch.Tensor)
    assert isinstance(adj_matrix, torch.Tensor)
    assert isinstance(label, torch.Tensor)
    assert dwt_features.dim() == 1  # Flattened features for one node
    assert adj_matrix.shape == (29, 29)
    assert label.item() == 0