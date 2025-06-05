"""Unit tests for the modma data module functionality."""

import numpy as np
import pytest
import mne
from src.clarity.data.modma import preprocess_raw_data, segment_data


def test_preprocess_raw_data(sample_eeg_data):
    """Test that preprocess_raw_data correctly filters and preprocesses the data."""
    # Process the raw data
    raw_processed = preprocess_raw_data(sample_eeg_data)
    
    # Verify return type
    assert isinstance(raw_processed, mne.io.BaseRaw)
    
    # Verify data shape preserved (preprocessing shouldn't change number of channels/timepoints)
    assert raw_processed.get_data().shape == sample_eeg_data.get_data().shape
    
    # Verify some preprocessing was applied (data values should be different)
    assert not np.allclose(raw_processed.get_data(), sample_eeg_data.get_data())


def test_segment_data(sample_eeg_data):
    """Test that segment_data correctly segments the continuous data into epochs."""
    # First preprocess the data
    raw_processed = preprocess_raw_data(sample_eeg_data)
    
    # Segment the data
    epochs = segment_data(raw_processed)
    
    # Verify return type
    assert isinstance(epochs, list)
    
    # Verify we have at least one epoch
    assert len(epochs) > 0
    
    # Verify each epoch has the expected dimensions (1 trial, n_channels, n_times)
    # The exact dimensions will depend on the implementation of segment_data
    for epoch in epochs:
        assert len(epoch.shape) == 3  # 3D array: (trials, channels, samples)
        assert epoch.shape[0] == 1  # One trial per epoch when using this function
        assert epoch.shape[1] == len(sample_eeg_data.ch_names)  # Same number of channels