#   jupytext_format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Psychiatry Digital Twin - EEG Analysis Pipeline
#
# This notebook implements an end-to-end analysis pipeline for the [MODMA dataset](http://modma.lzu.edu.cn/data_sources/sharing/), focusing on classifying multiple levels of depression severity from resting-state EEG signals.
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
import matplotlib.pyplot as plt

# %% [markdown]
# ### Cell 3: Main Experiment Loop (LOOCV)
#
# This cell executes the full experiment. We use **Leave-One-Out Cross-Validation (LOOCV)**, a rigorous evaluation method where the model is trained on all subjects except one, which is held out for testing. This process is repeated for every subject, ensuring that each one is used as a test case exactly once.
#
# LOOCV is computationally expensive but provides an unbiased and reliable estimate of model performance, which is especially important for smaller datasets like MODMA.
#
# The `MODELS_TO_RUN` list below can be configured to train and compare multiple models.

# %%
MODELS_TO_RUN = ['cnn', 'eegnet'] # Models to train and compare
all_model_results = {}

# %% [markdown]
# ### Cell 5: Interactive Visualization
#
# This cell provides an interactive tool to visualize the **Differential Entropy (DE)** of the preprocessed EEG signals as a topomap.
#
# **Differential Entropy** is a feature used in EEG analysis that measures the complexity of a signal in different frequency bands (Delta, Theta, Alpha, Beta). It is a powerful indicator of brain activity and is often used in studies of depression. Visualizing DE across the scalp can help identify spatial patterns of brain activity associated with different mental states.

# %%
try:
    sample_subj_id = 1
    # ... existing code ...

    # ... existing code ...
    else:
        print("The difference in performance is not statistically significant.")

    # ... existing code ...

    # ... existing code ...
except Exception as e:
    print(f"Error in interactive visualization: {e}")

# ... existing code ... 