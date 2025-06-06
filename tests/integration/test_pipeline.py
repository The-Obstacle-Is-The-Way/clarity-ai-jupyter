"""Integration tests for the complete EEG processing and model training pipeline."""

import torch
from src.clarity.data.modma import preprocess_raw_data, segment_data
from src.clarity.models import BaselineCNN
from src.clarity.training.loop import CustomEEGDataset, evaluate_model, train_model
from torch.utils.data import DataLoader


def test_full_cnn_pipeline(sample_eeg_data, subject_labels):
    """Test the full pipeline from raw EEG data to trained model and evaluation."""
    # Only use this test for small-scale testing - not full dataset processing
    subject_id = list(subject_labels.keys())[0]  # Use first subject

    # Step 1: Preprocess raw data
    raw_processed = preprocess_raw_data(sample_eeg_data)

    # Step 2: Segment data
    epochs = segment_data(raw_processed)

    # Modify CustomEEGDataset instance for testing to avoid loading real data
    class TestDataset(CustomEEGDataset):
        def __init__(self, data, labels):
            self.data = data
            self.labels = labels
            self.model_type = "cnn"

        def __getitem__(self, idx):
            # For CNN model, reshape to remove the extra dimension and get to shape (channels, time_points)
            if self.model_type == "cnn":
                # Convert to float32 to match model parameter dtype and remove the first dimension
                return torch.tensor(self.data[idx].squeeze(0), dtype=torch.float32), self.labels[idx]
            return self.data[idx], self.labels[idx]

    # Create a small dataset with the epochs data
    data_points = []
    labels = []
    for epoch in epochs[:5]:  # Use just first 5 epochs to keep test fast
        data_points.append(epoch)
        labels.append(subject_labels[subject_id])

    # Create dataset
    dataset = TestDataset(data_points, labels)

    # Step 3: Create DataLoader
    batch_size = 2
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Step 4: Initialize model with correct number of channels from data
    num_channels = data_points[0].shape[1]  # Get channel count from first epoch
    model = BaselineCNN(in_channels=num_channels)

    # Step 5: Setup optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    # Step 6: Train model
    trained_model = train_model(
        model, dataloader, optimizer, criterion,
        model_type="cnn",
        epochs=2  # Only train for 2 epochs to keep test fast
    )

    # Step 7: Evaluate model
    metrics_tuple, preds_and_labels_tuple = evaluate_model(trained_model, dataloader, model_type="cnn")

    # Verify the metrics are returned properly
    assert len(metrics_tuple) == 4
    accuracy, precision, recall, f1 = metrics_tuple

    # Metrics should be within valid range
    assert 0 <= accuracy <= 1
    assert 0 <= precision <= 1
    assert 0 <= recall <= 1
    assert 0 <= f1 <= 1

    # Verify the predictions and labels
    assert len(preds_and_labels_tuple) == 2
    assert isinstance(preds_and_labels_tuple[0], list)
    assert isinstance(preds_and_labels_tuple[1], list)
