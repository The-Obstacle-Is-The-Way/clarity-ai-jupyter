"""Unit tests for the feature extraction functionality."""

import numpy as np
from src.clarity.features import compute_adjacency_matrix, extract_dwt_features


def test_extract_dwt_features():
    """Test that extract_dwt_features correctly extracts wavelet features from a signal."""
    # Create a sample signal - a sine wave of 10Hz with 250Hz sampling rate
    sampling_rate = 250
    duration = 1.0  # 1 second
    t = np.linspace(0, duration, int(sampling_rate * duration))
    signal = np.sin(2 * np.pi * 10 * t)  # 10 Hz sine wave

    # Extract features
    features = extract_dwt_features(signal)

    # Verify output is a numpy array
    assert isinstance(features, np.ndarray)

    # Verify feature dimensionality (this will depend on the implementation)
    # For common wavelet decompositions, we expect multiple features
    assert features.size > 0


def test_compute_adjacency_matrix():
    """Test that compute_adjacency_matrix correctly computes the adjacency matrix."""
    # Create a sample EEG data array (29 channels, 250 time points)
    eeg_data = np.random.randn(29, 250)

    # Compute adjacency matrix
    adj_matrix = compute_adjacency_matrix(eeg_data)

    # Verify it's a numpy array
    assert isinstance(adj_matrix, np.ndarray)

    # Verify it's a square matrix with dimensions matching the number of channels
    assert adj_matrix.shape == (29, 29)

    # Verify it's symmetric (a property of adjacency matrices)
    assert np.allclose(adj_matrix, adj_matrix.T)

    # Verify diagonal is zero (no self-loops)
    assert np.allclose(np.diag(adj_matrix), 0)
