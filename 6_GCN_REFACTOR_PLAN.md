# Module 6: GCN Refactoring with PyTorch Geometric

This document outlines the detailed plan for refactoring the `MHA_GCN` model and its surrounding pipeline to use the `torch-geometric` library. This is a critical performance and scalability upgrade.

## Guiding Principles
- **Performance:** The primary goal is to leverage `PyG` for batched graph processing, eliminating the current Python loop in the training and evaluation functions.
- **Maintainability:** The refactored code should be cleaner, more readable, and easier to maintain.
- **Stability:** The final implementation must not introduce regressions. All existing and new tests must pass, and the codebase must be free of linting and type-checking errors.

---

## Task Checklist
- [ ] **1. Environment Setup:** Add and install `torch-geometric`.
- [ ] **2. Model Refactoring:** Update `MHA_GCN` to use `PyG` layers and data structures.
- [ ] **3. Dataset Refactoring:** Update `CustomEEGDataset` to produce `PyG` `Data` objects.
- [ ] **4. Training Loop Update:** Modify `train_model` and `evaluate_model` to handle `PyG` batches.
- [ ] **5. Notebook Integration:** Update the analysis notebook to use the correct `PyG` `DataLoader`.
- [ ] **6. Testing & Validation:**
    - [ ] Update existing GCN tests.
    - [ ] Add new tests for `PyG` data conversion.
    - [ ] Run the full test suite and ensure all tests pass (unit and integration).
    - [ ] Run `ruff check .` and `pyright` to ensure no new linter or type errors are present.

---

## Detailed Implementation Plan

### 1. Environment Setup
1.  **Modify `requirements.txt`**: Add `torch-geometric` to the requirements file.
2.  **Install Dependencies**: Run `pip install -r requirements.txt`. For an M1/M2 Mac, if this fails, consult the official `PyG` installation guide to install the necessary dependencies (`torch-scatter`, `torch-sparse`, etc.) using pre-compiled wheels for the `mps` architecture before retrying.

### 2. Model Refactoring (`src/clarity/models/mha_gcn.py`)
1.  **Imports**: Add `from torch_geometric.nn import GCNConv, global_mean_pool`.
2.  **Remove `SimpleGCNConv`**: Delete the entire class definition for `SimpleGCNConv`.
3.  **Update `MHA_GCN.__init__`**:
    - Replace `self.gcn1 = SimpleGCNConv(...)` with `self.gcn1 = GCNConv(...)`.
    - Replace `self.gcn2 = SimpleGCNConv(...)` with `self.gcn2 = GCNConv(...)`. The `in_channels` and `out_channels` arguments remain the same.
4.  **Update `MHA_GCN.forward`**:
    - Change the method signature from `forward(self, node_features, adj_matrix)` to `forward(self, x, edge_index, batch)`.
    - Update GCN calls to `x = self.gcn1(x, edge_index)`.
    - After the MHA layer, replace the manual `torch.mean` with `graph_embedding = global_mean_pool(attn_output, batch)`.
    - Ensure the final return remains a tuple of `(logits, attention_weights)`.

### 3. Dataset Refactoring (`src/clarity/training/loop.py`)
1.  **Imports**: Add `from torch_geometric.data import Data`.
2.  **Update `_process_gcn_data`**:
    - **Adjacency to Edge Index**: Convert the `avg_adj` numpy matrix into a `torch.long` tensor in COO format (`edge_index`).
        ```python
        # Inside the loop in _process_gcn_data
        adj_tensor = torch.tensor(avg_adj, dtype=torch.float32)
        edge_index = adj_tensor.nonzero().t().contiguous()
        ```
    - **Instantiate `Data` Object**: Instead of appending a tuple, create and append a `PyG` `Data` object.
        ```python
        # node_features is the dwt_feature_stack
        node_features = torch.tensor(np.array(dwt_feature_stack), dtype=torch.float32)
        graph_data = Data(x=node_features, edge_index=edge_index, y=torch.tensor(label, dtype=torch.long))
        self.data.append(graph_data)
        ```
3.  **Update `__getitem__`**:
    - For `model_type == 'mha_gcn'`, the method should simply `return self.data[idx]`. Remove all other logic for this case.

### 4. Training Loop Update (`src/clarity/training/loop.py`)
1.  **Imports**: Add `from torch_geometric.loader import DataLoader as PyGDataLoader`.
2.  **Remove Manual GCN Batching**: In `train_model` and `evaluate_model`, delete the `for i in range(inputs[0].shape[0]):` loop that was used to process GCN graphs individually.
3.  **Update Data Handling**:
    - The `data` object from the `PyGDataLoader` is the batch. Move it to the device: `data = data.to(DEVICE)`.
    - Unpack the batch for the model call: `logits, weights = model(data.x, data.edge_index, data.batch)`.
    - The labels are on `data.y`.
    - The loss can now be calculated on the full batch output directly.

### 5. Notebook Integration (`notebooks/clarity_eeg_analysis.py`)
1.  **Imports**: Add `from torch_geometric.loader import DataLoader as PyGDataLoader`.
2.  **Conditional DataLoader**: In the main LOOCV loop, instantiate the correct `DataLoader` based on the `model_name`.
    ```python
    if model_name == "mha_gcn":
        train_loader = PyGDataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = PyGDataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    else:
        # Keep the existing standard DataLoader
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    ```

### 6. Testing and Validation
1.  **Update `tests/unit/models/test_models.py`**:
    - Rewrite `test_mha_gcn_forward` to construct a batched input using `PyGDataLoader` from several `Data` objects.
    - The model should be called with `model(batch.x, batch.edge_index, batch.batch)`.
    - Assert that the output logits have the correct shape `(batch_size, num_classes)`.
2.  **Final Checks**:
    - Run `./run_tests.sh` and ensure **all 19 tests pass**.
    - Run `ruff check . --fix` and `pyright` to ensure the codebase is clean. 