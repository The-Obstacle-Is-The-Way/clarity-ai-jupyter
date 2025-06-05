MODMA EEG JupyterLab Setup Dossier

⸻

1 · Why MODMA + JupyterLab?

MODMA pairs 128-channel clinical EEG with a 3-lead wearable recording from the same participants, giving you a lab-grade / consumer-grade comparison set that’s perfect for Clarity AI’s diagnostic pipeline. ￼ ￼

⸻

2 · All the links you’ll ever need

Purpose	Link / Source
Official dataset portal & registration	modma.lzu.edu.cn (“Data > Application” tab) – account + approval required  ￼ ￼
Core research papers	• ArXiv pre-print (2020) • Sci Data paper (2022)  ￼ ￼
Reference GitHub repos	UAIS-LANZHOU/MODMA-Dataset (mirrors metadata) • RespectKnowledge/EEG_Speech_Depression_MultiDL (end-to-end example)  ￼ ￼
Example downstream study	ML depression screening framework using 3-lead subset  ￼
JupyterLab install docs	jupyter.org/install + Homebrew & pip notes  ￼ ￼


⸻

3 · Local dev prerequisites
	•	OS: macOS 13 +, Ubuntu 22.04 LTS, or Windows 11 + WSL2
	•	Python ≥ 3.10 (tested with 3.11)
	•	System packages: git, wget (or curl), 10 GB free disk (MODMA ≈ 6 GB)
	•	Python libs (add to requirements.txt or environment.yml):

jupyterlab
mne
numpy pandas scipy matplotlib seaborn
scikit-learn
torch torchvision torchaudio  # optional: for deep-learning demos
tqdm ipywidgets


	•	Extra tools (nice-to-have): nbgitpuller, jupyterlab-lsp, black, ruff

⸻

4 · Installation checklist (copy into README)
	1.	Create env

python -m venv .venv        # or conda create -n modma python=3.11
source .venv/bin/activate
pip install -r requirements.txt


	2.	Install / verify JupyterLab

pip install jupyterlab      # or brew install jupyterlab
jupyter lab --version

If build fails, upgrade pip, setuptools, wheel and retry. ￼

	3.	Clone the repo & scaffold notebooks

git clone https://github.com/<you>/clarity-eeg.git
cd clarity-eeg
mkdir -p notebooks data


	4.	Download MODMA
	•	Register > request access > approved email link.
	•	Move downloaded archive into data/ and unpack:

tar -xf MODMA_EEG.tar.gz -C data/


	•	Keep raw data out of Git (add data/** to .gitignore).

	5.	Launch JupyterLab

jupyter lab

Open notebooks/01_modma_eeg_demo.ipynb.

⸻

5 · Minimal working example (notebook outline)

# 01_modma_eeg_demo.ipynb
1. Setup & imports
   import mne, numpy as np, matplotlib.pyplot as plt
2. Load a 128-channel recording
   raw = mne.io.read_raw_fif("../data/128EEG/subj01_raw.fif", preload=True)
3. Quick visual QC
   raw.plot(n_channels=32, duration=10)
4. Load corresponding 3-lead wearable
   ...
5. Align epochs & compare PSDs
   mne.time_frequency.psd_welch(...)
6. (Optional) Load pretrained weights
   model = torch.load("../models/modma_cnn.pt")
   preds = model(raw_tensor)
7. Simple depression-vs-control ROC/AUC
   from sklearn.metrics import roc_auc_score

Use .mat loaders (scipy.io.loadmat or mne.io.read_epochs_eeglab) if files ship in MATLAB format. ￼

⸻

6 · Troubleshooting crib sheet

Symptom	Fast fix
pip install jupyterlab throws subprocess-exited-with-error	Upgrade pip setuptools wheel, then retry. ￼
pip command not found on macOS	Use python3 -m pip or ensure Homebrew/Pyenv shim in $PATH. ￼
Notebook kernel can’t see newly-installed libs	Restart kernel or install with %pip install <pkg> inside the notebook. ￼
Large .mat file won’t load	Verify MATLAB version ≥ 7.3 (HDF5) or convert using EEGLAB, then reload with MNE. ￼ ￼


⸻

7 · Quick action checklist
	•	Register & download MODMA EEG
	•	Commit requirements.txt / environment.yml
	•	Add notebooks/01_modma_eeg_demo.ipynb stub
	•	Update README with sections above
	•	Push branch, open PR, tag co-founder for review

⸻

Next idea → Once the demo runs, spin up a JupyterHub on GCP so clinicians can explore patient-level EEG without local installs—happy to map that out when you’re ready!


Absolutely—here’s a copy-and-paste, link-enabled list of top recent MODMA EEG papers (including 2025 and a few foundational references), so you can save, share, or run with it in any doc or chat window.

⸻

🧠 Top Papers Using the MODMA EEG Dataset

⸻

1. A Multi-modal Open Dataset for Mental-disorder Analysis
Hanshu Cai, Yiwen Gao, Shuting Sun, et al.
Nature Scientific Data, 2022
This is the foundational paper introducing the MODMA dataset, describing its design, subjects, and multi-modal data.
https://www.nature.com/articles/s41597-022-01211-x

⸻

2. Machine Learning Fairness for Depression Detection Using EEG Data
Angus Man Ho Kwok, Jiaee Cheong, Sinan Kalkan, Hatice Gunes
arXiv preprint, January 2025
Assesses ML fairness and bias mitigation strategies in depression detection with MODMA and other EEG datasets.
https://arxiv.org/abs/2501.18192

⸻

3. Cross-Subject Depression Level Classification Using EEG Signals with a Sample Confidence Method
ZhongYi Zhang, ChenYang Xu, LiXuan Zhao, HuiRang Hou, QingHao Meng
arXiv preprint, March 2025
Proposes a sample confidence method to address label subjectivity and class imbalance using MODMA and other EEG datasets.
https://arxiv.org/abs/2503.13475

⸻

4. Graph-based EEG Approach for Depression Prediction: Integrating Time-Frequency Complexity and Spatial Topology
Wei Liu, Kebin Jia, Zhuozheng Wang
Frontiers in Neuroscience, April 2024
Presents a graph-based method for depression prediction using MODMA and PRED+CT datasets.
https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2024.1367212/full

⸻

5. Multimodal Depression Detection Based on an Attention Graph Neural Network
AIMS Press, March 2025
Develops a multimodal depression detection model integrating EEG and speech, using the MODMA dataset.
https://www.aimspress.com/aimspress-data/mbe/2025/3/PDF/mbe-22-03-024.pdf

⸻

You can paste this list anywhere and all the links will be clickable. If you want a version that’s markdown/table ready, just say the word!
