"""Global pytest fixtures and configuration.

This file contains fixtures that can be used across all test modules.
"""

import os
import sys

import mne
import numpy as np
import pytest
import torch

# Add the src directory to the Python path to ensure imports work correctly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


@pytest.fixture(scope="session")
def seed():
    """Return a seed for reproducible tests."""
    return 42


@pytest.fixture(scope="session", autouse=True)
def set_random_seeds(seed):
    """Set random seeds for reproducibility across all tests."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


@pytest.fixture(scope="session")
def sample_eeg_data():
    """Generate a small sample of synthetic EEG data for testing."""
    # Create a simple 10-channel, 1000-sample synthetic EEG signal
    sfreq = 250  # Hz
    data = np.random.randn(10, 1000)  # 10 channels, 1000 time points

    # MNE standardized channel names for 10 channels
    ch_names = ['Fz', 'Cz', 'Pz', 'C3', 'C4', 'F3', 'F4', 'P3', 'P4', 'Oz']

    # Create info structure
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')

    # Create Raw object
    raw = mne.io.RawArray(data, info)
    return raw


@pytest.fixture(scope="session")
def sample_epochs(sample_eeg_data):
    """Create sample epochs from the sample EEG data."""
    # Create some events
    events = np.array([
        [100, 0, 1],  # Event at time 100, class 1
        [300, 0, 2],  # Event at time 300, class 2
        [500, 0, 1],  # Event at time 500, class 1
        [700, 0, 2],  # Event at time 700, class 2
    ])

    # Create epochs
    epochs = mne.Epochs(
        sample_eeg_data,
        events,
        tmin=0,
        tmax=1,  # 1 second epoch
        baseline=None,
        preload=True
    )
    return epochs


@pytest.fixture(scope="session")
def subject_labels():
    """Create a sample dictionary mapping subject IDs to labels."""
    return {
        "1": 1,  # Subject 1: Depressed
        "2": 1,  # Subject 2: Depressed
        "3": 0,  # Subject 3: Control
        "4": 0,  # Subject 4: Control
    }
