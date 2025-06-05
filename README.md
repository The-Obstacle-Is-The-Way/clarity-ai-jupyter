# Psychiatry Digital Twin - MODMA EEG Analysis

This project contains a JupyterLab environment for analyzing the [MODMA dataset](http://modma.lzu.edu.cn/data_sources/sharing/), with a focus on classifying depression levels from EEG signals. The analysis is implemented in `notebooks/clarity_eeg_analysis.py`, a Python script structured to be run as a notebook in modern IDEs like VS Code or JupyterLab.

It implements several deep learning models based on recent research papers, including a baseline CNN and an MHA-GCN, with a full Leave-One-Out Cross-Validation (LOOCV) training and evaluation pipeline.

---

## Setup and Installation

### 1. Create a Python Environment

It is highly recommended to use a virtual environment to manage dependencies.

```bash
# Create the virtual environment
python3 -m venv .venv

# Activate it
source .venv/bin/activate

# On Windows, use:
# .venv\\Scripts\\activate
```

### 2. Install Dependencies

Install all required Python libraries using the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### 3. Download the MODMA Dataset

You must request access to the MODMA dataset directly from the researchers.

-   **Register and request access** at the [official dataset portal](http://modma.lzu.edu.cn/data_sources/sharing/).
-   Once you receive the download link, place the archive in the `data/` directory.
-   Unpack the archive. The expected file structure is `data/MODMA/EEG_128channel_resting/sub01/rest.set`.

*Note: The `data/` directory is included in `.gitignore` and should not be committed to version control.*

### 4. Launch JupyterLab

With your environment activated and the data in place, you can launch JupyterLab.

```bash
jupyter lab
```

Navigate to `notebooks/clarity_eeg_analysis.py` to open and run the analysis. VS Code's Jupyter extension will also render this file as a notebook automatically. 