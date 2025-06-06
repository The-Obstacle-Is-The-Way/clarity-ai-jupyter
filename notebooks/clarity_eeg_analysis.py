# ---
# jupyter:
#   jupytext_format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Psychiatry Digital Twin - EEG Analysis Demo
#
# This notebook demonstrates the end-to-end analysis pipeline for the MODMA dataset.
# It imports all core logic from the `clarity` library in the `src` directory.

# %%
# CELL 1: Imports
import matplotlib.pyplot as plt
import mne
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from ipywidgets import interact
from sklearn.model_selection import LeaveOneOut
from src.clarity.data.modma import load_subject_data, preprocess_raw_data, segment_data
from src.clarity.features import calculate_de_features
from src.clarity.models import MHA_GCN, BaselineCNN, EEGNet
from typing import Dict, List, Union

# Local imports from our library
# Ensure 'src' is in PYTHONPATH or the notebook is run from project root
from src.clarity.training.config import (
    BATCH_SIZE,
    BDI_SCORES,
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
    if 0 <= score <= 4: return 0  # Normal
    if 5 <= score <= 9: return 1  # Mild
    if 10 <= score <= 14: return 2 # Moderate
    if 15 <= score <= 19: return 3 # Moderate to Major
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
# This cell executes the full LOOCV experiment.

# %%
loo = LeaveOneOut()
results: Dict[str, List[float]] = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
model: Union[nn.Module, None] = None

MODEL_TO_RUN = 'eegnet'  # Options: 'cnn', 'mha_gcn', 'eegnet'.

print(f"Starting LOOCV for model: {MODEL_TO_RUN}...")
for fold, (train_indices, test_indices) in tqdm(
    enumerate(loo.split(subject_ids_all)), total=len(subject_ids_all)
):
    train_subject_ids = [subject_ids_all[i] for i in train_indices]
    test_subject_ids = [subject_ids_all[i] for i in test_indices]

    print(
        f"\nFold {fold + 1}/{len(subject_ids_all)}: "
        f"Testing on subject {test_subject_ids[0]}"
    )

    train_dataset = CustomEEGDataset(train_subject_ids, labels_dict, model_type=MODEL_TO_RUN)
    test_dataset = CustomEEGDataset(test_subject_ids, labels_dict, model_type=MODEL_TO_RUN)

    if len(train_dataset) == 0 or len(test_dataset) == 0:
        print(f"Skipping fold {fold+1} due to missing data.")
        continue

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    if MODEL_TO_RUN == "cnn":
        model = BaselineCNN(num_classes=NUM_CLASSES)
    elif MODEL_TO_RUN == "mha_gcn":
        model = MHA_GCN(node_feature_dim=15 * 180, num_classes=NUM_CLASSES)
    elif MODEL_TO_RUN == "eegnet":
        model = EEGNet(num_classes=NUM_CLASSES)
    else:
        raise ValueError(f"Unsupported MODEL_TO_RUN: {MODEL_TO_RUN}. Choose from 'cnn', 'mha_gcn', or 'eegnet'.")

    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    print("Training...")
    model = train_model(model, train_loader, optimizer, criterion, model_type=MODEL_TO_RUN, epochs=EPOCHS)

    print("Evaluating...")
    acc, prec, rec, f1 = evaluate_model(model, test_loader, model_type=MODEL_TO_RUN)

    print(f"Fold {fold+1} Results: Accuracy={acc:.4f}, F1={f1:.4f}")
    results['accuracy'].append(acc)
    results['precision'].append(prec)
    results['recall'].append(rec)
    results['f1'].append(f1)

# %% [markdown]
# ### Cell 4: Results

# %%
if results['accuracy']:
    avg_accuracy = np.mean(results['accuracy'])
    avg_precision = np.mean(results['precision'])
    avg_recall = np.mean(results['recall'])
    avg_f1 = np.mean(results['f1'])

    print("\n--- Overall LOOCV Results ---")
    print(f"Model: {MODEL_TO_RUN}")
    print(f"Average Accuracy: {avg_accuracy:.4f} ± {np.std(results['accuracy']):.4f}")
    print(f"Average Precision: {avg_precision:.4f} ± {np.std(results['precision']):.4f}")
    print(f"Average Recall: {avg_recall:.4f} ± {np.std(results['recall']):.4f}")
    print(f"Average F1-Score: {avg_f1:.4f} ± {np.std(results['f1']):.4f}")
else:
    print("No results to display.")

# %% [markdown]
# ### Cell 5: Interactive Visualization

# %%
try:
    sample_subj_id = 1
    raw_sample = load_subject_data(sample_subj_id)
    if raw_sample:
        raw_sample_p = preprocess_raw_data(raw_sample.copy())
        epochs_sample = segment_data(raw_sample_p)
        info_for_topo = epochs_sample[0].info if epochs_sample else None
        # Keep sample_subj_id in global scope for use in plot function
        global sample_subj_id_for_plots
        sample_subj_id_for_plots = sample_subj_id
    else:
        info_for_topo = None
        epochs_sample = []
        sample_subj_id_for_plots = 0
except Exception:
    info_for_topo = None
    epochs_sample = []
    sample_subj_id_for_plots = 0

def plot_topomap_for_band(band_name: str):
    if info_for_topo is None:
        print("Sample data not loaded. Cannot plot.")
        return

    all_de = [calculate_de_features(epoch) for epoch in epochs_sample]
    avg_de_all_bands = np.mean(all_de, axis=0)

    band_idx = list(FREQ_BANDS.keys()).index(band_name)
    de_to_plot = avg_de_all_bands[:, band_idx]

    fig, ax = plt.subplots(figsize=(6, 6))
    im, _ = mne.viz.plot_topomap(de_to_plot, info_for_topo, axes=ax, show=False, cmap='viridis')
    fig.colorbar(im, ax=ax)
    ax.set_title(
        f"Differential Entropy Topomap for {band_name.capitalize()} Band "
        f"(Subject {sample_subj_id_for_plots})"
    )
    plt.show()

if info_for_topo:
    interact(plot_topomap_for_band, band_name=list(FREQ_BANDS.keys()))
else:
    print("Skipping interactive plot as sample data could not be loaded.")
