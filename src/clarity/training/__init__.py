from . import config
from .loop import CustomEEGDataset, evaluate_model, train_model

__all__ = [
    "CustomEEGDataset",
    "train_model",
    "evaluate_model",
    "config",
]
