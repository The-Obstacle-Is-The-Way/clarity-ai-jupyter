"""Clarity: EEG Analysis Toolkit

This package provides tools for EEG data processing, feature extraction,
model training, and visualization for depression detection research.
"""

from . import data, features, models, training, viz

__all__ = [
    "data",
    "features",
    "models",
    "training",
    "viz",
]

__version__ = "0.1.0"  # Initial version
