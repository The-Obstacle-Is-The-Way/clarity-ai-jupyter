from typing import Dict, List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset
from tqdm.notebook import tqdm

# Import data functions inside methods to avoid circular imports
from ..features import compute_adjacency_matrix, extract_dwt_features
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
        # Import here to avoid circular imports
        from ..data.modma import load_subject_data, preprocess_raw_data, segment_data

        self.subject_ids = subject_ids
        self.labels_dict = labels_dict
        self.model_type = model_type

        self.data = []
        self.labels = []

        print(f"Loading data for {model_type}...")
        for subj_id in tqdm(self.subject_ids):
            raw = load_subject_data(subj_id)
            if raw is None:
                continue

            raw_p = preprocess_raw_data(raw)
            epochs = segment_data(raw_p)
            label = self.labels_dict[subj_id]

            if self.model_type == "mha_gcn":
                # TODO: Consider making num_windows a configurable parameter
                num_windows = 180  # Number of 2s windows to stack for MHA-GCN features
                for i in range(len(epochs) - num_windows + 1):
                    dwt_feature_stack = []
                    adj_matrices = []
                    for ch_idx in range(len(CHANNELS_29)):
                        channel_dwt_features = [
                            extract_dwt_features(
                                epochs[j].get_data(copy=False)[0, ch_idx, :]
                            )
                            for j in range(i, i + num_windows)
                        ]
                        dwt_feature_stack.append(np.array(channel_dwt_features).T)

                    for j in range(i, i + num_windows):
                        adj_matrices.append(
                            compute_adjacency_matrix(epochs[j].get_data(copy=False)[0])
                        )
                    avg_adj = np.mean(adj_matrices, axis=0)

                    self.data.append((np.array(dwt_feature_stack), avg_adj))
                    self.labels.append(label)
            else:
                for epoch_item in epochs:
                    epoch_data = epoch_item
                    self.data.append(epoch_data)
                    self.labels.append(label)

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Union[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ]:
        """Retrieves a sample from the dataset at the given index.

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            A tuple containing the data and label. The structure depends on model_type:
            - For 'cnn': (features_tensor, label_tensor)
            - For 'mha_gcn': (dwt_features_tensor, adj_matrix_tensor, label_tensor)
        """
        data_point = self.data[idx]
        label = self.labels[idx]

        if self.model_type == "cnn":
            return torch.FloatTensor(data_point), torch.tensor(label, dtype=torch.long)

        elif self.model_type == "mha_gcn":
            dwt_features, adj_matrix = data_point
            dwt_flat = dwt_features.reshape(dwt_features.shape[0], -1)
            # Return a flat tuple of three elements as expected by tests
            return (
                torch.FloatTensor(dwt_flat),
                torch.FloatTensor(adj_matrix),
                torch.tensor(label, dtype=torch.long)
            )

        # Fallback, should ideally not be reached if model_type is always cnn or mha_gcn
        return torch.FloatTensor(data_point), torch.tensor(label, dtype=torch.long)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    model_type: str,
    epochs: int = EPOCHS,
) -> nn.Module:
    """Trains a given PyTorch model.

    Args:
        model: The PyTorch model to train.
        train_loader: DataLoader for the training data.
        optimizer: The optimizer to use for training.
        criterion: The loss function.
        model_type: Type of the model ('cnn' or 'mha_gcn') for specific data handling.
        epochs: Number of epochs to train for.

    Returns:
        The trained model.
    """
    model.to(DEVICE)
    model.train()
    for _epoch in range(epochs):
        for data in train_loader:
            if model_type == "mha_gcn":
                dwt, adj, labels = data
                # Keep inputs as a tuple of (dwt, adj) as required by the model
                inputs = (dwt.to(DEVICE), adj.to(DEVICE))
                labels = labels.to(DEVICE)
            else:
                inputs, labels = data
                # Handle inputs which could be a tensor or a tuple of tensors
                if isinstance(inputs, tuple):
                    inputs = tuple(t.to(DEVICE) for t in inputs)
                else:
                    inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

            optimizer.zero_grad()

            if model_type == "mha_gcn":
                # For MHA-GCN, process each graph in the batch individually
                # as the MHA_GCN.forward currently handles one graph.
                batch_loss = torch.tensor(0.0, device=DEVICE, requires_grad=True)
                for i in range(inputs[0].shape[0]):
                    output = model(inputs[0][i], inputs[1][i])
                    loss = criterion(output.unsqueeze(0), labels[i].unsqueeze(0))
                    batch_loss += loss
                batch_loss /= inputs[0].shape[0]
                batch_loss.backward()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()

            optimizer.step()
    return model


def evaluate_model(
    model: nn.Module, test_loader: DataLoader, model_type: str
) -> Tuple[float, float, float, float]:
    """Evaluates a trained PyTorch model on a test dataset.

    Args:
        model: The trained PyTorch model to evaluate.
        test_loader: DataLoader for the test data.
        model_type: Type of the model ('cnn' or 'mha_gcn') for specific data handling.

    Returns:
        A tuple containing (accuracy, precision, recall, f1_score).
    """
    model.to(DEVICE)
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for data in test_loader:
            if model_type == "mha_gcn":
                dwt, adj, labels = data
                inputs = (dwt.to(DEVICE), adj.to(DEVICE))
            else:
                inputs, labels = data
                # Handle inputs which could be a tensor or a tuple of tensors
                if isinstance(inputs, tuple):
                    inputs = tuple(t.to(DEVICE) for t in inputs)
                else:
                    inputs = inputs.to(DEVICE)

            labels = labels.cpu().numpy()

            if model_type == "mha_gcn":
                # For MHA-GCN, process each graph in the batch individually
                for i in range(inputs[0].shape[0]):
                    output = model(inputs[0][i], inputs[1][i])
                    pred = torch.argmax(output, dim=0)
                    all_preds.append(pred.cpu().numpy())
                    all_labels.append(labels[i])
            else:
                outputs = model(inputs)
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels)

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
    return accuracy, precision, recall, f1
