# Psychiatry Digital Twin - MODMA EEG Analysis

This project contains a JupyterLab environment for analyzing the [MODMA dataset](http://modma.lzu.edu.cn/data_sources/sharing/), with a focus on classifying depression levels from EEG signals. The analysis is implemented in `notebooks/clarity_eeg_analysis.py`, a Python script structured to be run as a notebook in modern IDEs like VS Code or JupyterLab.

It implements several deep learning models based on recent research papers, including a baseline CNN and an MHA-GCN, with a full Leave-One-Out Cross-Validation (LOOCV) training and evaluation pipeline.

---

## License and Terms of Use

This project is intended for **academic and non-commercial research purposes only**.

The [MODMA dataset](http://modma.lzu.edu.cn/data_sources/sharing/) used in this research is subject to its own End User License Agreement (EULA). Users of this codebase are responsible for obtaining the dataset themselves and adhering to all of its terms, which strictly prohibit commercial use.

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
# .venv\Scripts\activate
```

### 2. Install Dependencies

Install all required Python libraries using the `requirements.txt` file. For a fully reproducible environment with pinned versions, use `requirements.lock.txt`.

```bash
# For a stable, reproducible environment
pip install -r requirements.lock.txt

# Or for a flexible development environment
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

---

## Core Scientific References

The methodologies implemented in this repository are grounded in the following key research papers. Users are encouraged to consult them for a deeper understanding of the scientific context.

1.  **MODMA Dataset:**
    Cai, H., Yuan, Z., Gao, Y., et al. A multi-modal open dataset for mental-disorder analysis. *Sci Data* **9**, 178 (2022). [https://doi.org/10.1038/s41597-022-01211-x](https://doi.org/10.1038/s41597-022-01211-x)

2.  **Depression Level Classification & Sample Confidence:**
    Zhang, Z., Xu, C., Zhao, L., Hou, H., & Meng, Q. (2025). Cross-Subject Depression Level Classification Using EEG Signals with a Sample Confidence Method. *arXiv preprint arXiv:2503.13475*.

3.  **Graph-based EEG Analysis:**
    Liu, W., Jia, K., & Wang, Z. (2024). Graph-based EEG approach for depression prediction: integrating time-frequency complexity and spatial topology. *Frontiers in Neuroscience*, *18*, 1367212. [https://doi.org/10.3389/fnins.2024.1367212](https://doi.org/10.3389/fnins.2024.1367212)

4.  **Machine Learning Fairness in EEG:**
    Dang, V. N., Cascarano, A., Mulder, R. H., et al. Fairness and bias correction in machine learning for depression prediction across four study populations. *Sci Rep* **14**, 7848 (2024). [https://doi.org/10.1038/s41598-024-58427-7](https://doi.org/10.1038/s41598-024-58427-7)

5.  **Multimodal Detection (GCN and Transformer):**
    Jia, X., Chen, J., Liu, K., Wang, Q., & He, J. (2025). Multimodal depression detection based on an attention graph convolution and transformer. *Mathematical Biosciences and Engineering*, *22*(3), 652-676. [https://doi.org/10.3934/mbe.2025024](https://doi.org/10.3934/mbe.2025024) 