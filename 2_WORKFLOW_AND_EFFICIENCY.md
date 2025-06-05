# Module 2: Workflow and Efficiency

This module addresses the most critical performance issue in the repository: the redundant data loading and preprocessing in the Leave-One-Out Cross-Validation (LOOCV) loop. Implementing a caching strategy will dramatically speed up experimentation.

## Checklist

- [ ] **Implement a Caching Mechanism for Processed Data**
- [ ] **Integrate Caching into the `CustomEEGDataset`**
- [ ] **Add Configurable Debug/Development Mode**

---

## Implementation Details

### 1. Implement a Caching Mechanism for Processed Data

**Task:** Create functions to save and load processed data for each subject. We will save the preprocessed and segmented data to disk (e.g., as `.pkl` or `.npy` files) to avoid recomputing them on every run.

**Files to Edit:**
*   Create a new file: `src/clarity/data/caching.py`
*   Modify: `src/clarity/training/config.py`

**Instructions:**

1.  **Update Config:** Add a `CACHE_DIR` path to `config.py`.

    ```python
    # In src/clarity/training/config.py
    # ... existing code ...
    # --- Data Configuration ---
    DATA_DIR = "./data/MODMA/"  # Path to the MODMA dataset
    CACHE_DIR = "./data/processed_cache/" # Directory to store cached data
    NUM_SUBJECTS = 53  # Total number of subjects in the MODMA dataset
    # ... existing code ...
    ```

2.  **Create `caching.py`:** Create a new file for saving and loading logic. This keeps the caching implementation separate and clean.

    **Proposed Code (`src/clarity/data/caching.py`):**

    ```python
    import os
    import pickle
    from ..training.config import CACHE_DIR

    def save_to_cache(data, subject_id, model_type):
        """Saves processed data for a subject to a cache file."""
        if not os.path.exists(CACHE_DIR):
            os.makedirs(CACHE_DIR)
        
        cache_file = os.path.join(CACHE_DIR, f"subject_{subject_id}_{model_type}.pkl")
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)
        print(f"Saved processed data for subject {subject_id} to cache.")

    def load_from_cache(subject_id, model_type):
        """Loads processed data for a subject from a cache file if it exists."""
        cache_file = os.path.join(CACHE_DIR, f"subject_{subject_id}_{model_type}.pkl")
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                print(f"Loading processed data for subject {subject_id} from cache.")
                return pickle.load(f)
        return None
    ```

### 2. Integrate Caching into the `CustomEEGDataset`

**Task:** Modify the `CustomEEGDataset` to use the new caching functions. Before processing data for a subject, it should first try to load it from the cache. If not available, it should process the data and then save it to the cache for future use.

**File to Edit:** `src/clarity/training/loop.py`

**Instructions:**

1.  Import the new caching functions.
2.  In the `__init__` method of `CustomEEGDataset`, wrap the data processing logic with the cache check.

**Proposed Code Edit (`src/clarity/training/loop.py`):**

```python
// ... existing code ...
from ..features import compute_adjacency_matrix, extract_dwt_features
from ..data.caching import save_to_cache, load_from_cache # <-- Import caching functions
from .config import CHANNELS_29, DEVICE, EPOCHS

class CustomEEGDataset(Dataset):
// ... existing code ...
        print(f"Loading data for {model_type}...")
        for subj_id in tqdm(self.subject_ids):
            
            # --- Caching Logic ---
            cached_data = load_from_cache(subj_id, self.model_type)
            if cached_data is not None:
                processed_epochs, label = cached_data
            else:
                raw = load_subject_data(subj_id)
                if raw is None:
                    continue

                raw_p = preprocess_raw_data(raw)
                processed_epochs = segment_data(raw_p)
                label = self.labels_dict[subj_id]
                save_to_cache((processed_epochs, label), subj_id, self.model_type)
            # --- End Caching Logic ---

            if self.model_type == "mha_gcn":
// ... existing code ...
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
// ... existing code ...
                    for j in range(i, i + num_windows):
                        adj_matrices.append(
                            compute_adjacency_matrix(processed_epochs[j].get_data(copy=False)[0])
                        )
// ... existing code ...
                    self.labels.append(label)
            else:
                for epoch_item in processed_epochs:
                    epoch_data = epoch_item
                    self.data.append(epoch_data)
                    self.labels.append(label)

// ... existing code ...
```
*Note: The feature extraction for MHA-GCN is complex and happens *after* segmentation. The cache will store the segmented `epochs` list. This is a good balance, as preprocessing and segmentation are the most expensive repeated steps.*

### 3. Add Configurable Debug/Development Mode

**Task:** To allow for faster testing and debugging, add a "debug mode" that runs the LOOCV loop on a small subset of subjects.

**File to Edit:** `notebooks/clarity_eeg_analysis.py`

**Instructions:**

1.  Add a new boolean configuration variable `DEBUG_MODE` at the top of the notebook.
2.  Before starting the LOOCV loop, if `DEBUG_MODE` is `True`, slice the `subject_ids_all` list to a small number (e.g., 3 subjects).

**Proposed Code Edit (`notebooks/clarity_eeg_analysis.py`):**

```python
// ... existing code ...
# CELL 1: Imports
// ... existing code ...
# ---

# %% [markdown]
# ### Cell 2: Setup & Configuration
# All parameters are imported from `src/clarity/training/config.py`.

# %%
# --- Development Flags ---
DEBUG_MODE = True # If True, runs on a small subset of subjects for quick testing.
# -------------------------

np.random.seed(SEED)
torch.manual_seed(SEED)
// ... existing code ...
subject_ids_all = [str(i) for i in range(1, NUM_SUBJECTS + 1)]
labels_dict = {i: 1 if int(i) <= 24 else 0 for i in subject_ids_all}

if DEBUG_MODE:
    subject_ids_all = subject_ids_all[:3]
    print("!!! RUNNING IN DEBUG MODE on 3 subjects !!!")

# %% [markdown]
# ### Cell 3: Main Experiment Loop (LOOCV)
// ... existing code ...
```

</rewritten_file> 