# ---
# jupyter:
#   jupytext_format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Psychiatry Digital Twin - EEG Analysis Pipeline
#
# This notebook implements an end-to-end analysis pipeline for the [MODMA dataset](https://modma.lzu.edu.cn/data/index/), focusing on classifying multiple levels of depression severity from resting-state EEG signals.
#
# **Objective:** To compare the performance of several deep learning architectures using a robust Leave-One-Out Cross-Validation (LOOCV) strategy.
#
# **Models Implemented:**
# - `BaselineCNN`: A simple 1D CNN for raw EEG time-series.
# - `EEGNet`: A compact, well-established CNN architecture designed for EEG data.
# - `MHA_GCN`: A Graph Convolutional Network with Multi-Head Attention that models brain connectivity.
# - `SpectrogramViT`: A Vision Transformer that classifies 2D spectrogram representations of EEG signals.
#
# All core logic is imported from the `clarity` library in the `src` directory.

# %%
# CELL 1: Imports
from typing import Dict, List, TypedDict

import matplotlib.pyplot as plt
import mne
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from ipywidgets import fixed, interact
from scipy.stats import ttest_rel
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import LeaveOneOut
from src.clarity.data.modma import load_subject_data, preprocess_raw_data
from src.clarity.features import calculate_de_features
from src.clarity.models import MHA_GCN, BaselineCNN, EEGNet, SpectrogramViT

# Local imports from our library
# Ensure 'src' is in PYTHONPATH or the notebook is run from project root
from src.clarity.training.config import (
    BATCH_SIZE,
    BDI_SCORES,
    CHANNELS_29,
    DEPRESSION_LEVELS,
    DEVICE,
    EPOCHS,
    FREQ_BANDS,
    LR,
    NUM_CLASSES,
    NUM_SUBJECTS,
    SEED,
)
from src.clarity.training.loop import CustomEEGDataset, evaluate_model, train_model
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as PyGDataLoader
from tqdm.notebook import tqdm

# %% [markdown]
# ### Cell 2: Setup & Configuration
# All parameters are imported from `src/clarity/training/config.py`.

# %%
# --- Development Flags ---
DEBUG_MODE = True # If True, runs on a small subset of subjects for quick testing.
# -------------------------

np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

print(f"Using device: {DEVICE}")

subject_ids_all = [str(i) for i in range(1, NUM_SUBJECTS + 1)]

def get_severity_class(score):
    if 0 <= score <= 4:
        return 0  # Normal
    elif 5 <= score <= 9:
        return 1  # Mild
    elif 10 <= score <= 14:
        return 2 # Moderate
    elif 15 <= score <= 19:
        return 3 # Moderate to Major
    else:
        return 4  # Major

# Create labels based on multi-class severity
labels_dict = {
    subj_id: get_severity_class(score)
    for subj_id, score in BDI_SCORES.items()
}

if DEBUG_MODE:
    # Filter subject_ids_all AND labels_dict for debug mode
    debug_subjects = subject_ids_all[:3]
    labels_dict = {k: v for k, v in labels_dict.items() if k in debug_subjects}
    subject_ids_all = debug_subjects
    print(f"!!! RUNNING IN DEBUG MODE on {len(subject_ids_all)} subjects !!!")

# %% [markdown]
# ### Cell 3: Main Experiment Loop (LOOCV)
#
# This cell executes the full experiment. We use **Leave-One-Out Cross-Validation (LOOCV)**, a rigorous evaluation method where the model is trained on all subjects except one, which is held out for testing. This process is repeated for every subject, ensuring that each one is used as a test case exactly once.
#
# LOOCV is computationally expensive but provides an unbiased and reliable estimate of model performance, which is especially important for smaller datasets like MODMA.
#
# The `MODELS_TO_RUN` list below can be configured to train and compare multiple models.

# %%
class ModelResult(TypedDict):
    results: Dict[str, List[float]]
    preds: List[int]
    labels: List[int]

# --- Model Selection ---
# Define all available models. Comment out any you wish to skip.
MODELS_TO_RUN = [
    'cnn',
    'eegnet',
    # 'mha_gcn',
    # 'vit',
]
# -----------------------

all_model_results: Dict[str, ModelResult] = {}

for model_name in MODELS_TO_RUN:
    # --- Model Instantiation ---
    # Moved outside the loop for efficiency. The model is re-initialized for each fold.
    if model_name == "cnn":
        model_class = BaselineCNN
        model_args = {'num_classes': NUM_CLASSES}
    elif model_name == "mha_gcn":
        model_class = MHA_GCN
        model_args = {'node_feature_dim': 15 * 180, 'num_classes': NUM_CLASSES}
    elif model_name == "eegnet":
        model_class = EEGNet
        model_args = {'num_classes': NUM_CLASSES}
    elif model_name == "vit":
        model_class = SpectrogramViT
        model_args = {'num_classes': NUM_CLASSES}
    else:
        raise ValueError(f"Unsupported model: {model_name}.")

    loo = LeaveOneOut()
    results: Dict[str, List[float]] = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
    all_fold_preds: List[int] = []
    all_fold_labels: List[int] = []

    print(f"--- Starting LOOCV for model: {model_name} ---")
    for fold, (train_indices, test_indices) in tqdm(
        enumerate(loo.split(subject_ids_all)), total=len(subject_ids_all)
    ):
        train_subject_ids = [subject_ids_all[i] for i in train_indices]
        test_subject_ids = [subject_ids_all[i] for i in test_indices]

        print(
            f"\nFold {fold + 1}/{len(subject_ids_all)}: "
            f"Testing on subject {test_subject_ids[0]}"
        )

        train_dataset = CustomEEGDataset(train_subject_ids, labels_dict, model_type=model_name)
        test_dataset = CustomEEGDataset(test_subject_ids, labels_dict, model_type=model_name)

        if len(train_dataset) == 0 or len(test_dataset) == 0:
            print(f"Skipping fold {fold+1} due to missing data.")
            continue

        if model_name == "mha_gcn":
            train_loader = PyGDataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True) # type: ignore
            test_loader = PyGDataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False) # type: ignore
        else:
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        # Re-initialize the model for each fold to ensure independent training
        model = model_class(**model_args) # type: ignore

        optimizer = optim.Adam(model.parameters(), lr=LR)
        criterion = nn.CrossEntropyLoss()

        print("Training...")
        model = train_model(model, train_loader, optimizer, criterion, model_type=model_name, epochs=EPOCHS)

        print("Evaluating...")
        metrics, preds_and_labels = evaluate_model(model, test_loader, model_type=model_name)
        acc, prec, rec, f1 = metrics
        preds, labels = preds_and_labels

        all_fold_preds.extend(preds)
        all_fold_labels.extend(labels)

        print(f"Fold {fold+1} Results: Accuracy={acc:.4f}, F1={f1:.4f}")
        results['accuracy'].append(acc)
        results['precision'].append(prec)
        results['recall'].append(rec)
        results['f1'].append(f1)

    result_data: ModelResult = {
        'results': results,
        'preds': all_fold_preds,
        'labels': all_fold_labels
    }
    all_model_results[model_name] = result_data

# %% [markdown]
# ### Cell 4: Results & Comparison

# %%
for model_name, model_data in all_model_results.items():
    results = model_data['results']
    all_fold_labels = model_data['labels']
    all_fold_preds = model_data['preds']

    if results['accuracy']:
        avg_accuracy = np.mean(results['accuracy'])
        avg_precision = np.mean(results['precision'])
        avg_recall = np.mean(results['recall'])
        avg_f1 = np.mean(results['f1'])

        print(f"\n--- Overall LOOCV Results for {model_name} ---")
        print(f"Average Accuracy: {avg_accuracy:.4f} ± {np.std(results['accuracy']):.4f}")
        print(f"Average Precision: {avg_precision:.4f} ± {np.std(results['precision']):.4f}")
        print(f"Average Recall: {avg_recall:.4f} ± {np.std(results['recall']):.4f}")
        print(f"Average F1-Score: {avg_f1:.4f} ± {np.std(results['f1']):.4f}")

        # --- Confusion Matrix ---
        cm = confusion_matrix(all_fold_labels, all_fold_preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=list(DEPRESSION_LEVELS.keys()),
            yticklabels=list(DEPRESSION_LEVELS.keys()),
        )
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title(f"Aggregated Confusion Matrix for {model_name}")
        plt.show()

    else:
        print(f"No results to display for {model_name}.")

# --- Statistical Significance Testing ---
if len(MODELS_TO_RUN) == 2:
    model1_name, model2_name = MODELS_TO_RUN
    model1_acc = all_model_results[model1_name]['results']['accuracy']
    model2_acc = all_model_results[model2_name]['results']['accuracy']

    if model1_acc and model2_acc:
        t_stat, p_value = ttest_rel(model1_acc, model2_acc)

        print("\n--- Model Comparison ---")
        print(f"Paired t-test between {model1_name} and {model2_name} accuracies:")
        print(f"T-statistic: {t_stat:.4f}")
        print(f"P-value: {p_value:.4f}")

        if p_value < 0.05:
            print("The difference in performance is statistically significant.")
        else:
            print("The difference in performance is not statistically significant.")

# %% [markdown]
# ### Cell 5: Interactive Visualization
#
# This cell provides an interactive tool to visualize the **Differential Entropy (DE)** of the preprocessed EEG signals as a topomap.
#
# **Differential Entropy** is a feature used in EEG analysis that measures the complexity of a signal in different frequency bands (Delta, Theta, Alpha, Beta). It is a powerful indicator of brain activity and is often used in studies of depression. Visualizing DE across the scalp can help identify spatial patterns of brain activity associated with different mental states.
#
# *Note: This visualization uses data from a single, pre-selected subject to demonstrate the feature.*

# %%
def plot_topomap_for_band(band_name: str, epochs, info, subj_id):
    """Calculates and plots a DE topomap for a specific frequency band."""
    if not epochs or info is None:
        print("Sample data not available. Cannot plot.")
        return

    all_de = [calculate_de_features(epoch.get_data(copy=False)[0]) for epoch in epochs]
    avg_de_all_bands = np.mean(all_de, axis=0)

    band_idx = list(FREQ_BANDS.keys()).index(band_name)
    de_to_plot = avg_de_all_bands[:, band_idx]

    fig, ax = plt.subplots(figsize=(6, 6))
    im, _ = mne.viz.plot_topomap(de_to_plot, info, axes=ax, show=False, cmap='viridis')
    fig.colorbar(im, ax=ax)
    ax.set_title(
        f"Differential Entropy Topomap for {band_name.capitalize()} Band (Subject {subj_id})"
    )
    plt.show()

# --- Load data for visualization ---
try:
    SAMPLE_SUBJECT_ID = '1'
    raw_sample = load_subject_data(SAMPLE_SUBJECT_ID)
    if raw_sample:
        raw_p = preprocess_raw_data(raw_sample.copy(), perform_ica=False)
        epochs_sample = mne.make_fixed_length_epochs(raw_p, duration=2.0, overlap=1.0)
        info_for_topo = epochs_sample.info

        # Make the plotting function interactive
        interact(
            plot_topomap_for_band,
            band_name=list(FREQ_BANDS.keys()),
            epochs=fixed(epochs_sample),
            info=fixed(info_for_topo),
            subj_id=fixed(SAMPLE_SUBJECT_ID)
        )
    else:
        print("Skipping interactive plot: Sample data could not be loaded.")
except Exception as e:
    print(f"An error occurred during visualization setup: {e}")
    print("Skipping interactive plot.")


# %% [markdown]
# ### Cell 6: GCN Attention Visualization

# %%
# This cell runs only if 'mha_gcn' was in MODELS_TO_RUN and successfully completed.
if 'mha_gcn' in all_model_results:
    print("\n--- Visualizing MHA-GCN Attention ---")

    # To visualize attention, we need a trained MHA-GCN model and a sample batch of data.
    # We will re-load a model and a single subject's data for this purpose.

    # 1. Instantiate a fresh model
    gcn_model = MHA_GCN(node_feature_dim=15 * 180, num_classes=NUM_CLASSES)

    # 2. Load data for a sample subject
    VIS_SUBJECT_ID = '1'
    vis_dataset = CustomEEGDataset([VIS_SUBJECT_ID], labels_dict, model_type='mha_gcn')

    if vis_dataset:
        vis_loader = PyGDataLoader(vis_dataset, batch_size=1, shuffle=False) # type: ignore
        sample_batch = next(iter(vis_loader))
        sample_batch = sample_batch.to(DEVICE)

        # 3. We can't easily retrieve the trained model from the LOOCV loop.
        # For a stable visualization, it would be best to load saved model weights.
        # Since we are not saving weights in this example, we will pass the data
        # through an *untrained* model to demonstrate the visualization code.
        # The attention patterns will be random but will confirm the code runs.

        gcn_model.to(DEVICE)
        gcn_model.eval()
        with torch.no_grad():
            _, attention_weights = gcn_model(sample_batch.x, sample_batch.edge_index, sample_batch.batch)

        if attention_weights is not None:
            attention_matrix = attention_weights.squeeze().cpu().numpy()
            plt.figure(figsize=(12, 10))
            sns.heatmap(
                attention_matrix,
                cmap='viridis',
                xticklabels=CHANNELS_29,
                yticklabels=CHANNELS_29
            )
            plt.title(f'Random Initialized Attention Weights for Subject {VIS_SUBJECT_ID}')
            plt.xlabel('Channels')
            plt.ylabel('Channels')
            plt.show()
        else:
            print("Could not retrieve attention weights from the model.")
    else:
        print(f"Could not load data for subject {VIS_SUBJECT_ID} to visualize attention.")
else:
    print("\nSkipping GCN Attention Visualization: MHA_GCN model was not run.")
