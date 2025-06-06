# Module 3: Model Enhancements

This module details the plan to expand and improve the model architectures used in the repository. The goals are to incorporate more powerful, optimized libraries, add stronger baselines, and prepare the groundwork for more advanced classification tasks.

## Checklist

- [ ] **Refactor GCN with PyTorch Geometric (`PyG`) for Batching**
- [ ] **Add `EEGNet` as a Strong Baseline Model**
- [ ] **Enable Multi-Class Classification**
- [ ] **Implement a Spectrogram-based Vision Transformer (ViT) Model**

---

## Implementation Details

### 1. Refactor GCN with PyTorch Geometric (`PyG`)

**Task:** The current `SimpleGCNConv` and `MHA_GCN` process graphs one by one. By switching to PyTorch Geometric, we can leverage optimized GCN layers that support batching, which will simplify and speed up the training loop.

**Files to Edit:**
*   `requirements.txt`: Add `torch-geometric`.
*   `src/clarity/models/mha_gcn.py`: Replace `SimpleGCNConv` and refactor `MHA_GCN`.
*   `src/clarity/training/loop.py`: Update `CustomEEGDataset` and `train_model`/`evaluate_model` to handle `PyG` data objects.

**Instructions:**

1.  **Add Dependency:** Add `torch-geometric` to `requirements.txt`. Note that its installation can be tricky, so it's good to point to the official instructions.
2.  **Refactor `MHA_GCN`:**
    *   Replace `SimpleGCNConv` with `torch_geometric.nn.GCNConv`.
    *   The `forward` pass will now accept a `data` object from `PyG`'s `DataLoader`, which contains batched node features (`data.x`), edge indices (`data.edge_index`), and a batch vector (`data.batch`).
3.  **Update `CustomEEGDataset`:**
    *   The `__getitem__` method for `mha_gcn` should now return a `torch_geometric.data.Data` object for each graph.
4.  **Update Training Loop:**
    *   The `DataLoader` for the GCN should be from `torch_geometric.loader`.
    *   The `train_model` and `evaluate_model` loops can be simplified as they no longer need to manually iterate over graphs in a batch.

*(This is a major refactoring. Below is a conceptual sketch.)*

**Conceptual Code Sketch (`src/clarity/models/mha_gcn.py`):**

```python
# import torch.nn as nn
# import torch.nn.functional as F
# from torch_geometric.nn import GCNConv, global_mean_pool

# class MHA_GCN_PyG(nn.Module):
#     def __init__(...):
#         super().__init__()
#         self.gcn1 = GCNConv(node_feature_dim, gcn1_out)
#         self.gcn2 = GCNConv(gcn1_out, gcn2_out)
#         # ... mha, fc_out ...
#
#     def forward(self, x, edge_index, batch):
#         x = self.gcn1(x, edge_index).relu()
#         x = self.gcn2(x, edge_index).relu()
#         # ... apply mha ...
#         x = global_mean_pool(x, batch) # Pool graph embeddings
#         # ... pass to fc_out ...
#         return x
```

---

### 2. Add `EEGNet` as a Strong Baseline Model

**Task:** The audit suggests adding `EEGNet`, a well-established compact CNN for EEG, as a stronger baseline than the current `BaselineCNN`.

**Files to Edit:**
*   Create new file: `src/clarity/models/eegnet.py`
*   Modify: `src/clarity/models/__init__.py`
*   Modify: `notebooks/clarity_eeg_analysis.py`

**Instructions:**

1.  **Create `eegnet.py`:** Implement the `EEGNet` architecture in a new file. The architecture is widely available in literature and other open-source projects.
2.  **Expose in `__init__.py`:** Add `EEGNet` to the `__all__` list in `src/clarity/models/__init__.py`.
3.  **Add to Notebook:** Add `'eegnet'` as an option to the `MODEL_TO_RUN` variable in the notebook and instantiate it in the model selection logic.

---

### 3. Enable Multi-Class Classification

**Task:** The project is set up for binary classification, but the data and audit suggest moving towards multi-class severity prediction. This task involves modifying the configuration and model output layers.

**Files to Edit:**
*   `src/clarity/training/config.py`
*   `src/clarity/models/baseline_cnn.py`
*   `src/clarity/models/mha_gcn.py`
*   `notebooks/clarity_eeg_analysis.py`

**Instructions:**

1.  **Update `config.py`:**
    *   Change `NUM_CLASSES` from `2` to `5` (to match the `DEPRESSION_LEVELS`).
    *   Add a new dictionary `BDI_SCORES` to map subject IDs to their BDI scores (this will need to be sourced from the dataset's documentation).
2.  **Update Models:** Change the `num_classes` default value in all model `__init__` methods to `5` and ensure the final `nn.Linear` layer outputs this many classes.
3.  **Update Labels in Notebook:** Modify the `labels_dict` in `notebooks/clarity_eeg_analysis.py` to map subject IDs to the 5 severity classes instead of binary labels. You will need a function to map the BDI score to the correct class from `DEPRESSION_LEVELS`.

**Proposed Code Edit (`notebooks/clarity_eeg_analysis.py`):**

```python
# ... in Cell 2: Setup & Configuration ...

# FAKE BDI SCORES FOR DEMONSTRATION - REPLACE WITH REAL DATA
BDI_SCORES = {str(i): np.random.randint(0, 28) for i in range(1, NUM_SUBJECTS + 1)}

def get_severity_class(score):
    if 0 <= score <= 4: return 0  # Normal
    if 5 <= score <= 9: return 1  # Mild
    if 10 <= score <= 14: return 2 # Moderate
    if 15 <= score <= 19: return 3 # Moderate to Major
    return 4  # Major

# Update NUM_CLASSES in config.py to 5
labels_dict = {
    subj_id: get_severity_class(score)
    for subj_id, score in BDI_SCORES.items()
}
```

---

### 4. Implement a Spectrogram-based Vision Transformer (ViT) Model

**Task:** The audit highlighted the potential of using Vision Transformers on EEG spectrograms. The `timm` library is already a dependency, and `features.py` can generate spectrograms. This task is to connect them.

**Files to Edit:**
*   Create new file: `src/clarity/models/eeg_vit.py`
*   Modify: `src/clarity/features.py` (if needed)
*   Modify: `src/clarity/training/loop.py`
*   Modify: `notebooks/clarity_eeg_analysis.py`

**Instructions:**

1.  **Create `eeg_vit.py`:** Create a new model class that wraps a `timm` ViT model. The `forward` pass will take the 3-channel spectrogram image produced by `extract_stft_spectrogram_eeg`.

    **Conceptual Code Sketch (`src/clarity/models/eeg_vit.py`):**

    ```python
    # import torch.nn as nn
    # import timm
    #
    # class SpectrogramViT(nn.Module):
    #     def __init__(self, num_classes=5, pretrained=True):
    #         super().__init__()
    #         # Load a pretrained ViT model from timm
    #         self.vit = timm.create_model('vit_base_patch16_224', pretrained=pretrained)
    #         # Replace the classifier head with one for our number of classes
    #         self.vit.head = nn.Linear(self.vit.head.in_features, num_classes)
    #
    #     def forward(self, x):
    #         # x is a batch of spectrograms (batch, 3, 224, 224)
    #         return self.vit(x)
    ```
2.  **Update `CustomEEGDataset`:** Add a new `model_type` option, `'vit'`. When selected, the dataset should generate and return spectrograms from `extract_stft_spectrogram_eeg` instead of raw data or DWT features.
3.  **Add to Notebook:** Add `'vit'` as an option to the `MODEL_TO_RUN` variable and instantiate the new model. 