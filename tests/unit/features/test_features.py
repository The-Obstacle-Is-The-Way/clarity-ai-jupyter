"""Unit tests for the feature extraction functionality."""

import numpy as np
from src.clarity.features import (
    calculate_de_features,
    compute_adjacency_matrix,
    extract_dwt_features,
    extract_stft_spectrogram_eeg,
)
from src.clarity.training.config import FREQ_BANDS


def test_calculate_de_features():
    """Test the differential entropy feature calculation."""
    # 2 channels, 2 seconds of data at 250Hz
    epoch_data = np.random.randn(2, 500)
    de_features = calculate_de_features(epoch_data)
    assert de_features.shape == (2, len(FREQ_BANDS))
    assert not np.isnan(de_features).any()


def test_extract_stft_spectrogram_eeg():
    """Test the STFT spectrogram extraction."""
    epoch_data = np.random.randn(29, 500)  # 29 channels, 2s at 250Hz
    target_size = (128, 128)
    spectrogram = extract_stft_spectrogram_eeg(epoch_data, target_size=target_size)

    assert spectrogram.shape == (3, *target_size)
    assert spectrogram.min() >= 0.0
    assert spectrogram.max() <= 1.0
    assert not np.isnan(spectrogram).any()


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
