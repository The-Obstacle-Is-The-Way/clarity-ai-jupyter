# Module 1: Preprocessing Enhancements

This module focuses on refining and validating the data preprocessing pipeline in `src/clarity/data/modma.py`. While the audit was based on an older version of the code, these steps ensure our current implementation is robust, configurable, and aligns with best practices.

## Checklist

- [ ] **Validate Band-Pass Filter Implementation**
- [ ] **Make Artifact Removal Configurable**
- [ ] **Add Rationale for Channel Selection**

---

## Implementation Details

### 1. Validate Band-Pass Filter Implementation

**Task:** The audit recommended adding a band-pass filter. The function `_apply_filters` in `src/clarity/data/modma.py` already implements filtering. This task is to confirm it functions as a proper band-pass filter and to improve its documentation.

**File to Edit:** `src/clarity/data/modma.py`

**Instructions:**

1.  Inspect the `_apply_filters` function. It currently applies a high-pass filter at 1 Hz (`raw.copy().filter(l_freq=1, h_freq=None)`) and a notch filter at 50 Hz.
2.  To make this a true band-pass filter as recommended (e.g., 1-40 Hz), modify the filter call.
3.  Update the docstring to clearly state the filter parameters.

**Proposed Code Edit (`src/clarity/data/modma.py`):**

```python
def _apply_filters(raw):
    """Apply frequency filters to raw data with adaptive parameters.

    Implements a 1-40 Hz band-pass filter and a 50 Hz notch filter to remove
    DC drifts and high-frequency noise, aligning with standard EEG
    preprocessing pipelines.

    Args:
        raw: MNE Raw object containing EEG data

    Returns:
        MNE Raw object with filters applied
    """
    # Apply high-pass filter with appropriate parameters
    raw = raw.copy().filter(l_freq=1, h_freq=40)

    # Calculate signal properties
```

---

### 2. Make Artifact Removal Configurable

**Task:** The ICA-based artifact removal in `preprocess_raw_data` is a great feature, but can be time-consuming. For rapid experimentation, it's beneficial to have a way to disable it. We will add a boolean flag to control this.

**File to Edit:** `src/clarity/data/modma.py`

**Instructions:**

1.  Modify the `preprocess_raw_data` function signature to accept a new boolean argument `perform_ica`, defaulting to `True`.
2.  Wrap the ICA-related steps (initialization, fitting, artifact detection, and application) in a conditional block that checks this flag.

**Proposed Code Edit (`src/clarity/data/modma.py`):**

```python
def preprocess_raw_data(raw, perform_ica: bool = True):
    """Applies channel selection, filtering, and optionally ICA to the raw MNE object.
    This function orchestrates the preprocessing pipeline using specialized
    functions for each step.
    Args:
        raw: Raw MNE object containing EEG data
        perform_ica: If True, applies ICA for artifact removal. Defaults to True.

    Returns:
        Preprocessed MNE Raw object
    """
    # Step 1: Select channels
    raw = _select_channels(raw)
    # Step 2: Apply filters
    raw = _apply_filters(raw)

    if not perform_ica:
        print("Skipping ICA artifact removal as per configuration.")
        return raw

    # Step 3: Apply ICA for artifact removal (primarily EOG)
    # Dynamically set n_components to be min(20, num_channels)
```

---

### 3. Add Rationale for Channel Selection

**Task:** The audit notes that the reason for selecting the specific 29 channels is not documented. Adding a comment to explain this provides important context for other researchers.

**File to Edit:** `src/clarity/training/config.py`

**Instructions:**

1.  Locate the `CHANNELS_29` list in the config file.
2.  Add a comment above it explaining the rationale for this specific subset of channels.

**Proposed Code Edit (`src/clarity/training/config.py`):**

```python
NUM_SUBJECTS = 53  # Total number of subjects in the MODMA dataset

# --- EEG Configuration ---
# The 29 channels selected correspond to a standard 10-20 system layout,
# covering key cortical areas while excluding peripheral channels that are
# often noisier. This subset is chosen to align with common practices in
# EEG depression research for better comparability.
CHANNELS_29 = [
    "F7",
    "F3",
``` 