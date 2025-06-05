"""Centralized configuration settings for the EEG analysis project."""
import torch

# --- Reproducibility ---
SEED = 42

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Data Configuration ---
DATA_DIR = "./data/MODMA/"  # Path to the MODMA dataset
NUM_SUBJECTS = 53  # Total number of subjects in the MODMA dataset

# --- EEG Configuration ---
CHANNELS_29 = [
    "F7",
    "F3",
    "Fz",
    "F4",
    "F8",
    "T7",
    "C3",
    "Cz",
    "C4",
    "T8",
    "P7",
    "P3",
    "Pz",
    "P4",
    "P8",
    "O1",
    "O2",
    "Fpz",
    "F9",
    "FT9",
    "FT10",
    "TP9",
    "TP10",
    "PO9",
    "PO10",
    "Iz",
    "A1",
    "A2",
    "POz",
]  # List of 29 EEG channel names used in the study
SAMPLING_RATE = 250  # Hz
WINDOW_SIZE = 2  # seconds, duration of each EEG epoch
OVERLAP = 0.5  # Percentage of overlap between consecutive windows

# --- Feature Configuration ---
FREQ_BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 40),
}  # Frequency bands (Hz) for Differential Entropy (DE) feature extraction

# --- Model & Training Configuration ---
DEPRESSION_LEVELS = {
    "Normal": (0, 4),
    "Mild": (5, 9),
    "Moderate": (10, 14),
    "Moderate to Major": (15, 19),
    "Major": (20, 27),
}  # Mapping of BDI-II scores to depression severity levels
NUM_CLASSES = 2  # Number of classes for the classification task (e.g., 2 for depressed/control)
BATCH_SIZE = 4
EPOCHS = 100
LR = 0.001  # Learning rate for the optimizer
