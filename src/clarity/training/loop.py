from typing import Dict, List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Data
from tqdm.notebook import tqdm

from ..data.caching import load_from_cache, save_to_cache

# Import data functions inside methods to avoid circular imports
from ..features import (
    compute_adjacency_matrix,
    extract_dwt_features,
    extract_stft_spectrogram_eeg,
)
from .config import CHANNELS_29, DEVICE, EPOCHS  # Corrected relative import


class CustomEEGDataset(Dataset):
    """Custom PyTorch Dataset for EEG data.

    Handles feature extraction and data loading for different model types.
    """
    # Dataset handles both CNN and MHA-GCN feature extraction

    def __init__(self,
        subject_ids: List[str],
        labels_dict: Dict[str, int],
        model_type: str = "cnn"
    ):
        """Initializes the CustomEEGDataset.

        Args:
            subject_ids: List of subject identifiers to include in the dataset.
            labels_dict: Dictionary mapping subject IDs to their labels.
            model_type: Type of model for which data is being prepared
                ('cnn' or 'mha_gcn').
        """
        from ..data.modma import load_subject_data, preprocess_raw_data, segment_data

        self.subject_ids = subject_ids
        self.labels_dict = labels_dict
        self.model_type = model_type
        self.data: list = []
        self.labels: list = []

        print(f"Loading data for {model_type}...")
        for subj_id in tqdm(self.subject_ids):
            processed_epochs, label = self._load_and_process_subject(
                subj_id, segment_data, load_subject_data, preprocess_raw_data
            )
            if processed_epochs is None:
                continue

            self._extract_features(processed_epochs, label)

    def _load_and_process_subject(
        self, subj_id, segment_data, load_subject_data, preprocess_raw_data
    ):
        """Loads and preprocesses data for a single subject, using cache."""
        cached_data = load_from_cache(subj_id, self.model_type)
        if cached_data is not None:
            return cached_data

        raw = load_subject_data(subj_id)
        if raw is None:
            return None, None

        raw_p = preprocess_raw_data(raw)
        processed_epochs = segment_data(raw_p)
        label = self.labels_dict[subj_id]
        save_to_cache((processed_epochs, label), subj_id, self.model_type)
        return processed_epochs, label

    def _extract_features(self, processed_epochs, label):
        """Extracts features based on the model type."""
        if self.model_type == "vit":
            self._process_vit_data(processed_epochs, label)
        elif self.model_type == "mha_gcn":
            self._process_gcn_data(processed_epochs, label)
        else:  # Default is 'cnn'
            self._process_cnn_data(processed_epochs, label)

    def _process_cnn_data(self, processed_epochs, label):
        """Processes data for the baseline CNN model."""
        for epoch_item in processed_epochs:
            self.data.append(epoch_item)
            self.labels.append(label)

    def _process_vit_data(self, processed_epochs, label):
        """Processes data for the Vision Transformer model."""
        for epoch_item in processed_epochs:
            spectrogram = extract_stft_spectrogram_eeg(epoch_item[0])
            self.data.append(spectrogram)
            self.labels.append(label)

    def _process_gcn_data(self, processed_epochs, label):
        """Processes data for the MHA-GCN model."""
        num_windows = 180
        for i in range(len(processed_epochs) - num_windows + 1):
            dwt_feature_stack = []
            adj_matrices = []
            for ch_idx in range(len(CHANNELS_29)):
                channel_dwt_features = [
                    extract_dwt_features(
                        processed_epochs[j].get_data(copy=False)[0, ch_idx, :]
                    )
                    for j in range(i, i + num_windows)
                ]
                dwt_feature_stack.append(np.array(channel_dwt_features).T)

            for j in range(i, i + num_windows):
                adj_matrices.append(
                    compute_adjacency_matrix(processed_epochs[j].get_data(copy=False)[0])
                )
            avg_adj = np.mean(adj_matrices, axis=0)

            # Convert adjacency matrix to edge index for PyG
            adj_tensor = torch.tensor(avg_adj, dtype=torch.float32)
            edge_index = adj_tensor.nonzero().t().contiguous()

            node_features = torch.tensor(np.array(dwt_feature_stack), dtype=torch.float32)
            # Flatten the node features
            node_features_flat = node_features.reshape(node_features.shape[0], -1)

            graph_data = Data(
                x=node_features_flat,
                edge_index=edge_index,
                y=torch.tensor(label, dtype=torch.long)
            )
            self.data.append(graph_data)
            # self.labels list is no longer needed for GCN as it's in the Data object
            self.labels.append(label) # Keep for now to avoid breaking __len__

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Union[Tuple[torch.Tensor, torch.Tensor], Data]:
        """Retrieves a sample from the dataset at the given index."""
        if self.model_type == "mha_gcn":
            return self.data[idx]

        data_point = self.data[idx]
        label = self.labels[idx]
        return torch.FloatTensor(data_point), torch.tensor(label, dtype=torch.long)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    model_type: str,
    epochs: int = EPOCHS,
) -> nn.Module:
    """Trains a given PyTorch model."""
    model.to(DEVICE)
    model.train()
    for _epoch in range(epochs):
        for data in train_loader:
            optimizer.zero_grad()

            if model_type == "mha_gcn":
                data = data.to(DEVICE)
                outputs, _ = model(data.x, data.edge_index, data.batch)
                loss = criterion(outputs, data.y)
            else: # Covers 'cnn', 'vit', and any other standard model
                inputs, labels = data
                if isinstance(inputs, tuple):
                    inputs = tuple(t.to(DEVICE) for t in inputs)
                else:
                    inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
    return model


def evaluate_model(
    model: nn.Module, test_loader: DataLoader, model_type: str
) -> Tuple[
    Tuple[float, float, float, float],
    Tuple[List[int], List[int]]
]:
    """Evaluates a trained PyTorch model on a test dataset."""
    model.to(DEVICE)
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for data in test_loader:
            if model_type == "mha_gcn":
                data = data.to(DEVICE)
                outputs, _ = model(data.x, data.edge_index, data.batch)
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy().tolist())
                all_labels.extend(data.y.cpu().numpy().tolist())
            else: # Covers 'cnn', 'vit', and any other standard model
                inputs, labels = data
                if isinstance(inputs, tuple):
                    inputs = tuple(t.to(DEVICE) for t in inputs)
                else:
                    inputs = inputs.to(DEVICE)

                cpu_labels = labels.cpu().numpy()
                outputs = model(inputs)
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy().tolist())
                all_labels.extend(cpu_labels)

    accuracy = float(accuracy_score(all_labels, all_preds))
    precision = float(precision_score(
        all_labels, all_preds, average="macro", zero_division="warn"
    ))
    recall = float(recall_score(
        all_labels, all_preds, average="macro", zero_division="warn"
    ))
    f1 = float(f1_score(all_labels, all_preds, average="macro", zero_division="warn"))
    print(
        f"Epoch metrics - "
        f"Accuracy: {accuracy:.4f}, "
        f"Precision: {precision:.4f}, "
        f"Recall: {recall:.4f}, F1: {f1:.4f}"
    )
    return (accuracy, precision, recall, f1), (all_preds, all_labels)
