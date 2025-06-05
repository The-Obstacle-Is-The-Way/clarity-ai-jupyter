# Module 5: Documentation and Reproducibility

This final module covers essential steps for improving the project's documentation, ensuring the environment is stable and reproducible, and cleaning up the repository structure.

## Checklist

- [ ] **Pin Dependencies for Full Reproducibility**
- [ ] **Enhance In-Notebook Markdown Explanations**
- [ ] **Update `README.md` with Results and References**
- [ ] **Clean Up Repository Structure**

---

## Implementation Details

### 1. Pin Dependencies for Full Reproducibility

**Task:** The `requirements.txt` file does not have pinned versions, which can lead to reproducibility issues if a dependency releases a breaking change. We will create a "frozen" requirements file.

**File to Create:** `requirements.lock.txt`

**Instructions:**

1.  After ensuring your virtual environment (`.venv`) has all the necessary packages installed from `requirements.txt`, generate a pinned version file.
2.  Update the `README.md` to recommend using this file for a stable setup, while keeping `requirements.txt` for more flexible development.

**Command to Generate the File:**

```bash
# Make sure your virtual environment is active
source .venv/bin/activate

# Install the dependencies first
pip install -r requirements.txt

# Generate the locked file
pip freeze > requirements.lock.txt
```

**`README.md` Update:** Add a note in the setup section explaining the two files.

> **For a stable, reproducible environment, use `requirements.lock.txt`:**
> `pip install -r requirements.lock.txt`
>
> **For setting up a new development environment, use `requirements.txt`:**
> `pip install -r requirements.txt`

---

### 2. Enhance In-Notebook Markdown Explanations

**Task:** The audit suggested that while the notebook is clean, it could benefit from more explanatory text to guide users.

**File to Edit:** `notebooks/clarity_eeg_analysis.py`

**Instructions:**

Go through the notebook and add more detailed markdown cells.

*   **Above Cell 1 (Imports):** Add a more detailed introduction. Explain the goal (LOOCV on MODMA), the models being tested, and what the outcome will be.
*   **Above Cell 3 (LOOCV Loop):** Explain what Leave-One-Out Cross-Validation is and why it's a good choice for this dataset size (i.e., robust evaluation on small datasets).
*   **Above Cell 5 (Visualization):** Explain what Differential Entropy represents in the context of EEG (a measure of signal complexity in a frequency band) and why visualizing it as a topomap is useful for understanding brain activity patterns.

---

### 3. Update `README.md` with Results and References

**Task:** The `README.md` should be a comprehensive entry point. As suggested by the audit, we should add a summary of the expected results and key academic references.

**File to Edit:** `README.md`

**Instructions:**

1.  **Add a "Results" Section:** After the "Setup" section, add a new H2 section called "Expected Results". Briefly summarize the performance of the baseline models (e.g., "The Baseline CNN achieves ~XX% accuracy, while the MHA-GCN model reaches ~YY% in a binary classification task.").
2.  **Add a "References" Section:** At the bottom of the file, add a new H2 section called "References". List the key papers that inspired this project's methodology, particularly the ones mentioned in the `AUDIT.md` (Liu et al., 2024; Zhang et al., 2025, etc.).

---

### 4. Clean Up Repository Structure

**Task:** The codebase audit revealed a redundant and empty `src/models` directory. It should be removed to keep the project structure clean.

**File/Directory to Remove:** `src/models/`

**Instructions:**

1.  Verify that the `src/models/` directory is indeed empty and not used for any dynamic file generation.
2.  Delete the directory.

**Command to Remove the Directory:**

```bash
# Ensure the directory is empty first
ls -la src/models/

# If empty, remove it
rmdir src/models/
```
This simple step improves the project's clarity by removing unused clutter. 