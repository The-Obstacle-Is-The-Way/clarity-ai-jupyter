"""Clarity: EEG Analysis Toolkit

This package provides tools for EEG data processing, feature extraction, 
model training, and visualization for depression detection research.
"""

from . import data
from . import features
from . import models
from . import training
from . import viz

__all__ = [
    "data",
    "features",
    "models",
    "training",
    "viz",
]

__version__ = "0.1.0"  # Initial version
