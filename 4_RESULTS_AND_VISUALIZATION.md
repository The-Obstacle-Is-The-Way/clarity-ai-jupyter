# Module 4: Results Analysis and Visualization

This module focuses on enhancing the final part of the analysis pipeline: interpreting and reporting results. The goal is to move beyond average accuracy and provide deeper insights into model performance and decision-making, as recommended by the audit.

## Checklist

- [ ] **Add Confusion Matrix to Evaluation**
- [ ] **Integrate Statistical Significance Testing**
- [ ] **Visualize GCN Attention/Adjacency for Interpretability**

---

## Implementation Details

### 1. Add Confusion Matrix to Evaluation

**Task:** A confusion matrix is essential for understanding classification performance, especially in a multi-class scenario. We will compute and plot a confusion matrix aggregated over all folds of the LOOCV.

**Files to Edit:**
*   `notebooks/clarity_eeg_analysis.py`
*   `src/clarity/training/loop.py`

**Instructions:**

1.  **Modify `evaluate_model`:** The function should return the predictions and true labels along with the metrics.

    **Proposed Code Edit (`src/clarity/training/loop.py`):**
    ```python
    // In evaluate_model function
    # ... after computing metrics
    # return accuracy, precision, recall, f1 # OLD
    return (accuracy, precision, recall, f1), (all_preds, all_labels) # NEW
    ```

2.  **Update Notebook Loop:**
    *   In the main LOOCV loop in `notebooks/clarity_eeg_analysis.py`, collect all predictions and labels from each fold.
    *   After the loop, use `sklearn.metrics.confusion_matrix` to compute the final matrix.
    *   Use `seaborn.heatmap` to plot the confusion matrix for a clear visual representation.

**Proposed Code Edit (`notebooks/clarity_eeg_analysis.py`):**

```python
// ... existing code ...
# In Cell 3, main LOOCV loop
# ...
all_fold_preds = []
all_fold_labels = []

# ... inside the for loop ...
    print("Evaluating...")
    metrics, preds_and_labels = evaluate_model(model, test_loader, model_type=MODEL_TO_RUN)
    acc, prec, rec, f1 = metrics
    preds, labels = preds_and_labels
    all_fold_preds.extend(preds)
    all_fold_labels.extend(labels)

# ... In Cell 4: Results
# After printing average metrics...

from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(all_fold_labels, all_fold_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=DEPRESSION_LEVELS.keys(),
            yticklabels=DEPRESSION_LEVELS.keys())
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title(f'Aggregated Confusion Matrix for {MODEL_TO_RUN}')
plt.show()

```

---

### 2. Integrate Statistical Significance Testing

**Task:** To rigorously compare two models (e.g., `BaselineCNN` vs. `MHA_GCN`), we need to check if the difference in their performance is statistically significant. A paired t-test on the fold-by-fold accuracy scores is appropriate for this.

**File to Edit:** `notebooks/clarity_eeg_analysis.py`

**Instructions:**

This requires running the LOOCV twice, once for each model, and saving the accuracy scores from each fold for both models. Then, a t-test can be performed.

1.  Structure the notebook to run the full loop for a list of models (`['cnn', 'mha_gcn']`).
2.  Store the list of accuracies for each model.
3.  Use `scipy.stats.ttest_rel` to perform a paired t-test on the two lists of accuracies.
4.  Report the t-statistic and p-value.

**Proposed Code Edit (`notebooks/clarity_eeg_analysis.py`):**

```python
# This would be a new cell at the end of the notebook or a separate analysis script

from scipy.stats import ttest_rel

# Assume results_cnn['accuracy'] and results_gcn['accuracy'] are available
# after running the LOOCV for both models.

# Example:
# results_cnn = {'accuracy': [0.6, 0.7, ...], ...}
# results_gcn = {'accuracy': [0.7, 0.75, ...], ...}

if 'results_cnn' in locals() and 'results_gcn' in locals():
    t_stat, p_value = ttest_rel(results_cnn['accuracy'], results_gcn['accuracy'])

    print("\n--- Model Comparison ---")
    print(f"Paired t-test between CNN and MHA-GCN accuracies:")
    print(f"T-statistic: {t_stat:.4f}")
    print(f"P-value: {p_value:.4f}")

    if p_value < 0.05:
        print("The difference in performance is statistically significant.")
    else:
        print("The difference in performance is not statistically significant.")

```

---

### 3. Visualize GCN Attention/Adjacency for Interpretability

**Task:** To understand what the `MHA_GCN` model is learning, we can visualize the learned graph structure. This can be done by averaging the adjacency matrices for each class or by inspecting the attention weights from the MHA layer.

**Files to Edit:**
*   `src/clarity/models/mha_gcn.py`: Modify the forward pass to return attention weights.
*   `notebooks/clarity_eeg_analysis.py`: Add a new visualization cell.

**Instructions:**

1.  **Return Attention Weights:** In `mha_gcn.py`, modify the `forward` method of the `MHA_GCN` model to also return the attention weights from the `self.mha` layer.

    **Proposed Code Edit (`src/clarity/models/mha_gcn.py`):**
    ```python
    // ... in MHA_GCN forward pass
    # attn_output, _ = self.mha(...) # OLD
    attn_output, attn_weights = self.mha(x_mha_input, x_mha_input, x_mha_input) # NEW
    # ...
    # return out # OLD
    return out, attn_weights # NEW
    ```

2.  **Add Visualization Cell:** In the notebook, after training and evaluating a GCN model on a single subject (or a representative one), get the attention weights. The attention matrix will have a shape like `(num_nodes, num_nodes)`. Use a heatmap to visualize it.

**Proposed Code Edit (`notebooks/clarity_eeg_analysis.py`):**

```python
# A new cell for GCN visualization

if model and MODEL_TO_RUN == 'mha_gcn':
    # Get a sample from the test set
    sample_data = next(iter(test_loader))
    dwt, adj, label = sample_data
    
    model.eval()
    with torch.no_grad():
        # Get model output and attention weights for the first item in the batch
        output, attention_weights = model(dwt[0].to(DEVICE), adj[0].to(DEVICE))

    # Squeeze to get a 2D matrix (29x29)
    attention_matrix = attention_weights.squeeze().cpu().numpy()

    plt.figure(figsize=(10, 8))
    sns.heatmap(attention_matrix, cmap='viridis', xticklabels=CHANNELS_29, yticklabels=CHANNELS_29)
    plt.title(f'Learned Attention Weights for Subject {test_subject_ids[0]}')
    plt.show()
``` 