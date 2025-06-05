This dossier will draw on all available source materials—including the MODMA dataset paper, MHA-GCN ViT paper, and DepL-GCN paper—and provide exact technical parameters, logic flows, formulas, and pseudocode or PyTorch implementations, where appropriate. It will be organized according to the 8 major task categories you've outlined, and include notebook-ready code structures, architecture diagrams, and hyperparameter tables that an AI coding agent can build on immediately.

# Psychiatry Digital Twin – Multimodal Depression Detection Notebook Blueprint

This blueprint outlines a Jupyter Notebook for implementing a multimodal depression detection model (the **Psychiatry Digital Twin** project). The notebook integrates EEG and speech modalities using a **Multi-Head Attention GCN + ViT** approach inspired by recent research. The notebook is organized into clear sections with markdown explanations and code, enabling an AI agent to implement and execute it end-to-end. We ensure all critical details – dataset handling, feature extraction, model architecture, training (with cross-validation), decision-level fusion, visualization, and broader requirements – are explicitly defined for reproducibility and clarity.

## 1. Jupyter Notebook Structure and Flow

**Notebook Overview:** The notebook is structured as a sequence of marked sections, each containing descriptive **Markdown cells** followed by **code cells** to perform the described tasks. We begin with an **Introduction** (Markdown) describing the project context and referencing the MHA-GCN ViT model architecture (which achieved \~89% accuracy on the MODMA dataset). Next, we perform **Environment Setup** (code) to import libraries (e.g. `numpy`, `scipy`, `mne`, `pywt`, `torch`, `torchaudio`/`librosa`, `torch_geometric`, `timm`, `matplotlib`, `plotly`, `ipywidgets`). We then proceed through data ingestion, feature extraction, model definition, training, and interactive analysis, as summarized below:

* **Introduction & Setup:** *(Markdown)* Outline the problem (depression detection from EEG + speech) and the model approach (Graph Convolution + Transformer), citing sources. *(Code)* Import required libraries, set random seeds for reproducibility, and configure the runtime (e.g., check GPU availability).

* **Data Loading & Preprocessing:** *(Markdown)* Describe the MODMA dataset structure and how we will load EEG and speech data. *(Code)* Load EEG recordings (e.g., using MNE or NumPy for 128-channel `.edf` or `.mat` files) and audio recordings (e.g., WAV files). Apply preprocessing: select 29 EEG channels of interest, filter signals 1–40 Hz, and remove artifacts (ICA); for audio, perform pre-emphasis filtering (H(z)=1–0.97z^–1) and normalization. The output of this section is cleaned EEG signals per subject and corresponding audio signals.

* **Feature Extraction:** *(Markdown)* Explain how features are extracted from EEG and speech. We detail two EEG feature methods – **Differential Entropy (DE)** in standard EEG bands and **Discrete Wavelet Transform (DWT)** for multi-scale features – and the speech spectrogram extraction. *(Code)* Compute EEG features for each subject: segment EEG into short windows (e.g. 2 s), compute band-power or DE features per segment, and/or perform DWT on each channel (using `pywt`) to get multi-resolution coefficients. Construct the brain network adjacency matrix via Pearson correlation between channel features. For speech, compute the Mel spectrogram: frame the signal (25 ms Hamming window, 10 ms step), apply STFT, then map power spectra through a Mel filter bank to get a time–frequency matrix (as in Figure 7 of MHA-GCN ViT paper). Convert spectrograms to a format suitable for the ViT (e.g. resize to 224×224 PNG images). The output is: for each EEG segment, a feature matrix (nodes × features) and adjacency matrix; for each speech sample, a Mel-spectrogram image or tensor.

* **Model Definition:** *(Markdown)* Describe the model architecture comprising: (a) an EEG **Graph Convolutional Network with Multi-Head Attention** (MHA-GCN), and (b) two **Vision Transformer (ViT)** models for EEG spectrograms and speech. We break down the GCN layers, attention mechanism, and ViT components (patch embedding, transformer encoder, classification head). *(Code)* Define model classes:

  * `MHAGCN` class: implements two GCN layers and the multi-head attention module as described (4 attention heads on graph nodes), followed by a pooling and fully-connected layer for output.
  * `VitClassifier` class: wraps a pre-trained ViT (Base-Patch16/224) from a library (e.g., timm or HuggingFace) and replaces the final layer to predict depression vs control (or severity level). Instantiate two such models (for EEG and speech).
    We ensure all layer parameters (e.g., number of GCN units, heads, etc.) match those in the literature when specified.

* **Training & Validation (Cross-Validation Loop):** *(Markdown)* Explain the training procedure using Leave-One-Out Cross-Validation (LOOCV) (as done for cross-subject evaluation) and/or 5-fold CV. Describe training hyperparameters, loss functions, and the **decision-level fusion** strategy for combining modalities. *(Code)* Implement the training loop:

  * Loop over each fold or each left-out subject. For each iteration, initialize the models (GCN, EEG-ViT, speech-ViT), and train on the training set. Use `torch.optim.Adam` with set learning rates for each branch.
  * Each training epoch iterates over mini-batches of training samples (each sample here corresponds to one EEG segment + its label, or we can aggregate by subject if not segmenting). Compute loss = sum of branch losses (or handle each branch separately if training independently). Use standard cross-entropy loss for classification.
  * After training, evaluate on the validation/test subject: obtain predicted probabilities from each branch and apply the weighted decision fusion (ensemble) to predict the class. Record metrics (accuracy, precision, recall, F1).
  * Repeat for all folds/subjects and compute average performance.
    We include code for computing performance metrics per fold and for aggregating final results (e.g., confusion matrix across LOOCV). The training code will print/plot training loss and accuracy curves for each model (Figure 8 equivalent), and final CV results.

* **Interactive Visualization & Analysis:** *(Markdown)* Outline the interactive plots and analysis provided to the user for model interpretation (detailed in Section 6). *(Code)* Implement interactive widgets/plots, e.g., a dropdown or slider to select a subject and visualize their EEG channel graph connectivity vs. Mel spectrogram, as well as model predictions for that subject. Also plot the learned **attention weights** on the brain network (node importance scores) and the distribution of spectrogram features for depressed vs control. We will use libraries like Plotly for interactive spectrograms and NetworkX/Plotly for graph visualization.

* **Conclusion & Next Steps:** *(Markdown, optional)* Summarize the results and discuss potential improvements or how this notebook fits into the larger Psychiatry Digital Twin platform.

**Markdown and Output Annotation:** Throughout the notebook, **markdown cells** precede each code block to explain the upcoming steps (for clarity to an AI engineer). After code execution, we include either inline comments or subsequent markdown to interpret the output. For example, after training, a markdown cell will discuss whether the model’s accuracy (\~89% as expected) was achieved and what the confusion matrix indicates. Visual outputs (plots, graphs) are accompanied by captions or markdown explanations highlighting key observations (e.g., “EEG spectrogram for a depressed patient shows higher power in low-frequency bands compared to a control”). This ensures the notebook is not only executable but also informative, mirroring an interactive lab report.

## 2. Data Handling – End-to-End

**Dataset Description:** We use the **MODMA** open dataset for mental-disorder analysis. It includes EEG and audio recordings from depression patients and healthy controls:

* **EEG data:** 53 subjects (24 clinically diagnosed MDD patients, 29 healthy controls) with 128-channel EEG recorded at 250 Hz in resting-state and during a Dot Probe task. In this notebook, we focus on the resting-state EEG of these 53 subjects. Each recording is \~5 minutes long. Prior to sharing, subjects completed clinical assessments (e.g. PHQ-9) that can provide depression severity labels.
* **Speech data:** 52 subjects (a subset of the above, 23 patients and 29 controls) with audio recorded in clinical interviews, reading tasks, and picture descriptions. Audio is assumed sampled at 16 kHz (monophonic). Each subject’s audio clips are concatenated or selected to represent their speech patterns.

**File Organization:** The dataset is provided in a structured format (e.g., BIDS on the UK Data Archive). For example:

* EEG data might be in `.edf` files or `.set` (EEGLAB) files per subject (e.g., `sub001_resting.fif` or similar).
* Audio files might be `.wav` per subject or per session (e.g., `sub001_interview.wav`).
* There may be metadata files (CSV or JSON) containing subject IDs, group labels (depressed/control), and possibly questionnaire scores.

We assume the data files are available in a folder structure accessible to the notebook (for example, `./MODMA/EEG/` and `./MODMA/Audio/` with matching subject IDs).

**Data Loading:** We will use appropriate libraries to load the raw data:

* *EEG:* Use **MNE-Python** (`mne.io.read_raw_edf` or `mne.io.read_raw_bdf`, etc.) if EEG is in EDF/BDF, or `scipy.io` if in MATLAB format. Load the signal for the 5-minute resting segment for each subject. Example code:

  ```python
  import mne
  raw = mne.io.read_raw_eeglab('sub001_resting.set', preload=True)  # example for EEGLAB .set file
  data = raw.get_data()  # shape: [128 channels, time_points]
  sfreq = raw.info['sfreq']  # 250 Hz
  ```

  We then select the 29 channels of interest by name. The chosen montage (10-20 system mostly) includes: F7, F3, Fz, F4, F8, T7, C3, Cz, C4, T8, P7, P3, Pz, P4, P8, O1, O2, Fpz, F9, FT9, FT10, TP9, TP10, PO9, PO10, Iz, A1, A2, POz. In code, after loading:

  ```python
  # Pick the 29 channels
  picks = ['F7','F3','Fz','F4','F8','T7','C3','Cz','C4','T8',
           'P7','P3','Pz','P4','P8','O1','O2','Fpz','F9','FT9','FT10',
           'TP9','TP10','PO9','PO10','Iz','A1','A2','POz']
  raw.pick_channels(picks)
  eeg_data = raw.get_data()  # shape: [29, T]
  ```
* *Audio:* Use `librosa` or `torchaudio` to load audio files. Example:

  ```python
  import librosa
  speech, sr = librosa.load('sub001_interview.wav', sr=16000)  # load audio at 16kHz
  ```

  Here `speech` is a 1D NumPy array of audio samples, `sr=16000`. We may have multiple audio clips per subject (interview, reading, etc.); these can be concatenated or the longest used, depending on analysis goals. For simplicity, assume one representative audio sample per subject.

**Preprocessing – EEG:**
We apply standard EEG preprocessing to each subject’s data to ensure signal quality:

1. **Filtering:** Band-pass filter each channel between 1–40 Hz. This removes drifts (<1 Hz) and high-frequency noise (>40 Hz, including muscle artifacts), preserving key EEG rhythms (delta–gamma). In code, using MNE:

   ```python
   raw_filtered = raw.copy().filter(l_freq=1.0, h_freq=40.0)
   eeg_data = raw_filtered.get_data()  # shape [29, T]
   ```

   This corresponds to the reported high-pass at 1 Hz and low-pass at 40 Hz.

2. **Artifact Removal:** Perform **Independent Component Analysis (ICA)** to remove eye-blink or movement artifacts. Using MNE:

   ```python
   ica = mne.preprocessing.ICA(n_components=29, random_state=42)
   ica.fit(raw_filtered)
   ica.detect_artifacts(raw_filtered)  # or manually exclude components corresponding to EOG artifacts
   clean_eeg = ica.apply(raw_filtered).get_data()
   ```

   After ICA, `clean_eeg` is the artifact-reduced EEG signal (29×T matrix). If MNE’s artifact detection is configured, it will find components correlating with EOG channels or abnormal patterns and exclude them. *Note:* Artifact removal can also be done by visual inspection or other methods, but ICA is a common choice.

3. **Segmentation:** We split each subject’s continuous EEG into windows for feature extraction. Following DepL-GCN’s approach, we use 2-second non-overlapping segments. With 250 Hz sampling, 2 s = 500 samples per segment. For \~5 minutes (300 s) recording, we get about 150 segments per subject. In code:

   ```python
   segment_length = int(2 * sfreq)  # 2 seconds * 250 Hz
   segments = []
   for start in range(0, clean_eeg.shape[1] - segment_length + 1, segment_length):
       seg = clean_eeg[:, start:start+segment_length]  # shape [29, 500]
       segments.append(seg)
   ```

   `segments` will be a list of arrays (each 29×500). Each segment inherits the subject’s label (depressed or control). We ensure no segment overlaps multiple sessions (here just resting-state).

4. **Baseline normalization:** Optionally, for each segment or channel we might remove the mean or perform z-score normalization across time to center the data. This can improve feature stability. (This step is not explicitly in the papers, but a common practice.)

**Preprocessing – Speech:**
We process each audio sample to prepare it for feature extraction:

1. **Pre-emphasis filtering:** Apply a first-order high-pass filter to boost high frequencies, compensating for spectral tilt. The filter is:
   $H(z) = 1 - \mu z^{-1},$
   with \$\mu=0.97\$. In code, this can be done by:

   ```python
   pre_emphasis = 0.97
   speech_preemph = np.append(speech[0], speech[1:] - pre_emphasis * speech[:-1])
   ```

   This creates `speech_preemph` where high-frequency components are slightly amplified relative to low-frequency.
2. **Framing and Windowing:** We divide the audio into short frames (25 ms) with a 10 ms stride (hop), assuming speech is quasi-stationary in \~20 ms intervals. Each frame is windowed by a Hamming window \$w(n)=0.54 - 0.46\cos(2\pi n/(N-1))\$ to taper the edges. In code, we will rely on the STFT function which implicitly does framing and windowing:

   ```python
   import numpy as np
   frame_length = int(0.025 * sr)   # 25 ms
   hop_length   = int(0.010 * sr)   # 10 ms
   window = np.hamming(frame_length)
   ```

   (Librosa’s STFT uses 2048 FFT by default; we’ll specify our frame\_length for accurate framing).
3. **Spectrogram via STFT:** Compute the Short-Time Fourier Transform on pre-emphasized, windowed frames to get the spectrogram. Each STFT column is the Fourier magnitude of a frame. In code with librosa:

   ```python
   import librosa
   D = librosa.stft(speech_preemph, n_fft=frame_length*2, hop_length=hop_length, window='hamming', center=True)
   # n_fft can be frame_length or next power of 2 for efficiency
   spectrogram = np.abs(D)**2  # power spectrum
   ```

   We square the magnitude to obtain the power spectrum, which gives the power spectral density over time.
4. **Mel Filter Bank:** Transform the frequency axis from linear frequency to the Mel scale to mimic human auditory perception. We apply a bank of triangular filters (Mel filters) to the power spectrum, summing energy in each Mel band. The Mel scale is defined such that frequencies are spaced perceptually. Using librosa:

   ```python
   mel_spec = librosa.feature.melspectrogram(S=spectrogram, sr=sr, n_mels=64, fmax=sr/2)
   mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
   ```

   This yields a Mel spectrogram (`mel_spec`) with, say, 64 Mel bands (this is a hyperparameter; common values 40-128). We convert to decibels (`power_to_db`) for a log-scaled spectrogram (as typically visualized). Figure 7 of MHA-GCN ViT shows an example mel spectrogram for depressed vs healthy speech.
5. **Normalization:** We can normalize the mel spectrogram to \[0,1] or standardize it for input to the ViT. Often, image models expect certain mean/std normalization (if using a pre-trained model, e.g., ImageNet means). We will handle this later in the ViT preprocessing.

Both EEG and speech features are now prepared. For each EEG segment we have a *node feature matrix* (29 nodes × feature\_dim) and *adjacency matrix* (29×29). For each speech sample (or corresponding EEG time window) we have a Mel spectrogram image (time×Mel\_freq). These will feed into different model branches.

**Subject Labels and Splitting:** We create a data structure to store the features and labels:

```python
eeg_graphs = []  # list of (features, adjacency, label)
speech_images = []  # list of (mel_spec_image, label)
for subj in subjects:
    label = 1 if subj in depressed_ids else 0
    # For each EEG segment of this subject:
    for seg in subj_segments[subj]:
        features = extract_features(seg)        # e.g., 5-band DE or wavelet features, shape [29, feature_dim]
        adj = compute_adj_matrix(seg, features) # Pearson correlation across channels, shape [29, 29]
        eeg_graphs.append((features, adj, label))
    # For speech, if available for this subject:
    if subj in speech_data:
        mel_image = extract_mel_spectrogram(speech_data[subj])  # shape [time_bins, mel_bins]
        speech_images.append((mel_image, label))
```

Here `extract_features(seg)` could compute either DE features in standard bands or wavelet features:

* **Differential Entropy (DE):** For each of the 29 channels in the 2s segment, compute DE in each frequency band. DE for a signal segment (treated as approximately Gaussian) is \$H(X) = \frac{1}{2}\ln(2\pi e,\sigma^2)\$. Since \$\frac{1}{2}\ln(2\pi e)\$ is constant, using \$\ln(\sigma^2)\$ (log-variance) suffices as a feature. We band-pass the segment into Delta (1–4 Hz), Theta (4–8 Hz), Alpha (8–13 Hz), Beta (13–30 Hz), and low Gamma (30–40 Hz) bands, then compute log-variance for each band. This yields 5 features per channel. In code:

  ```python
  import scipy.signal as sig
  bands = [(1,4),(4,8),(8,13),(13,30),(30,40)]
  band_features = []
  for (low, high) in bands:
      # design a Butterworth bandpass
      sos = sig.butter(4, [low, high], btype='band', fs=sfreq, output='sos')
      filt_seg = sig.sosfilt(sos, seg)  # filter each channel in seg
      # compute variance per channel
      var = filt_seg.var(axis=1)  # variance along time axis for each channel
      band_features.append(np.log(var + 1e-6))  # log-variance
  features = np.vstack(band_features).T  # shape [29 channels, 5 bands]
  ```

  This gives a 29×5 feature matrix for the segment. (This replicates the approach in SSPA-GCN and DepL-GCN.)

* **Wavelet features:** Alternatively, we perform a **Discrete Wavelet Transform** on each channel to capture multi-scale time-frequency information. We choose, for example, a 4-level wavelet decomposition with Daubechies-4 wavelet (`db4`). Each channel’s DWT produces approximation coefficients \$cA\_4\$ and detail coefficients \$cD\_1,\dots,cD\_4\$. We can construct the feature vector for the channel by taking the average energy or entropy of each sub-band. For example:

  ```python
  import pywt
  coeffs = pywt.wavedec(seg[ch, :], wavelet='db4', level=4)  # coeffs = [cA4, cD4, cD3, cD2, cD1]
  # Use log-energy of each set of coefficients as features:
  ch_features = [np.log(np.mean(c**2) + 1e-6) for c in coeffs]
  ```

  Doing this for each channel yields a 29×5 feature matrix as well (5 sub-band features per channel), analogous to DE features. These “wavelet features” serve as node attributes for the graph. (In MHA-GCN, the exact wavelet feature calculation was not explicitly detailed, but a similar approach can be assumed, providing multi-resolution energy features per channel.)

* **Adjacency Matrix:** We build a 29×29 adjacency matrix \$A\$ for each EEG segment to represent the brain network connectivity. We use the Pearson correlation coefficient between channels as edge weights:
  $A_{ij} = r_{ij} = \frac{\sum_t (x_{i,t}-\bar{x_i})(x_{j,t}-\bar{x_j})}{\sqrt{\sum_t (x_{i,t}-\bar{x_i})^2 \sum_t (x_{j,t}-\bar{x_j})^2}},$
  computed over the segment’s time samples. We can compute this efficiently:

  ```python
  corr_matrix = np.corrcoef(seg)  # seg shape [29, time]; result 29x29
  A = np.nan_to_num(corr_matrix)  # replace any NaN (constant signals) with 0
  ```

  By default, this gives values in \[-1,1]. If negative correlations are not meaningful for connectivity, we could take absolute values or set negatives to 0 (not mentioned in the paper, so we keep full correlation matrix as is). We also set the diagonal of \$A\$ to 1 (each node fully connected to itself). If needed, we can add a small constant to the diagonal to ensure the graph Laplacian is well-conditioned, but since we will normalize \$A\$ in GCN, it's optional.

* **Speech Mel Spectrogram images:** The `mel_image` from the earlier extraction is a 2D array (frames × Mel bands). Its dimensions might be, e.g., 300 frames × 64 Mel bands (depending on audio length and chosen n\_mels). Before feeding to ViT, we need to convert this into an image of size 224×224 as expected by the ViT base model. We can do this by simple image resizing:

  ```python
  import cv2
  mel_img_resized = cv2.resize(mel_spec_db, (224, 224))  # scale time and frequency axes to 224 each
  ```

  If the ViT expects 3-channel input (RGB), we can stack the grayscale image into 3 channels:

  ```python
  mel_img_rgb = np.stack([mel_img_resized]*3, axis=-1)  # shape 224x224x3
  ```

  We will do similar for EEG time-frequency images if we use them. For EEG, one strategy is to treat each channel’s spectrogram as an image and perhaps tile or stack them. However, in our approach, we already feed EEG into the GCN and also could feed an aggregate EEG spectrogram. In MHA-GCN ViT, they computed 2D spectrograms for EEG channels via STFT and fed them into a ViT, but it's ambiguous if they treated each channel separately or combined. For simplicity, we won’t use a separate EEG spectrogram ViT branch in this blueprint (since our GCN handles EEG). If desired, one could create an **EEG spectrogram image per segment** (e.g., average power across all 29 channels at each frequency-time, or an image collage of channels) and feed it to a second ViT. The paper suggests both EEG and speech spectrograms were input to ViT modules, so an advanced implementation can include an EEG spectrogram branch. Here, we proceed with the **speech spectrogram** for the ViT branch, and the **EEG graph** for the GCN branch.

**Data Splits (Cross-Validation):** Because of the limited number of subjects, we avoid a simple train-test split. Instead, we use cross-validation to maximize data usage:

* *Leave-One-Out CV:* recommended for assessing cross-subject performance. Each fold leaves one subject out as test data and uses all others for training. Given \~53 subjects, we will have 53 folds. This approach tests generalization to completely unseen subjects.
* *5-Fold CV:* as used in the MHA-GCN ViT study. Here we randomly split subjects into 5 folds (\~10–11 per fold), use 4 folds for training and 1 for testing, and repeat so each fold serves as test once. This is less exhaustive than LOOCV but reduces training cycles (5 vs 53). If computing resources are a concern, 5-fold can be used. We will demonstrate LOOCV for completeness, but the code can be easily adapted to k-fold.

For **LOOCV**, we create an index list of subjects. Pseudocode:

```python
subjects = list(subject_ids)  # e.g., [0,...,52] or actual IDs
results = []
for i, test_subj in enumerate(subjects):
    train_subjs = [s for s in subjects if s != test_subj]
    # Prepare training data:
    train_graphs = [g for g in eeg_graphs if g[2] in train_subjs]  # using label or id to filter
    train_images = [img for img in speech_images if img[1] in train_subjs]
    test_graphs  = [g for g in eeg_graphs if g[2] == test_subj]
    test_images  = [img for img in speech_images if img[1] == test_subj]
    # ... (train models on train_graphs & train_images, then evaluate on test_graphs & test_images)
```

We will ensure that if a subject has multiple EEG segments, all those segments of the test subject are withheld together (the model never sees any segment of the test subject during training). During testing, the model will produce predictions for each EEG segment of the test subject and each speech sample. We then aggregate those to decide the subject’s overall label (e.g., majority vote or averaging probabilities). Using subject-level accuracy (did we correctly classify that subject) is important. Similarly, for 5-fold, we partition the subject list into 5 roughly equal sets and iterate.

At this point, our data pipeline yields:

* `eeg_graphs`: list of training samples, each with (node\_features, adjacency, subject\_label).
* `speech_images`: list of samples with (image\_tensor, subject\_label).
  These will feed into the model training.

## 3. Detailed Feature Extraction Implementation

This section details the feature computations mathematically and with code examples, ensuring an implementation-ready guide.

**EEG Feature Extraction:** We present both approaches – *Differential Entropy features* and *Wavelet features* – for completeness.

* **Differential Entropy (DE) Features:** Following prior work, we divide each EEG segment into standard frequency bands and compute the feature vector \$f\_i\$ for each EEG channel \$i\$ as:
  $f_i = [DE_{i,\delta},\, DE_{i,\theta},\, DE_{i,\alpha},\, DE_{i,\beta},\, DE_{i,\gamma}],$
  where \$DE\_{i,b} = H(X\_{i,b})\$ is the differential entropy of channel \$i\$’s signal in band \$b\$. If \$X\_{i,b}\$ is approximately Gaussian, \$DE\_{i,b} \approx \frac{1}{2}\ln(2\pi e,\sigma\_{i,b}^2)\$ where \$\sigma\_{i,b}^2\$ is the variance of band-pass filtered signal. We implement DE by band-pass filtering and using \$\ln(\sigma^2)\$:

  ```python
  import numpy as np, scipy.signal as sig
  bands = {'delta':(1,4),'theta':(4,8),'alpha':(8,13),'beta':(13,30),'gamma':(30,40)}
  def differential_entropy_features(segment, sfreq=250):
      # segment: shape [29, time_samples]
      features = []
      for ch in range(segment.shape[0]):
          ch_feats = []
          for band, (low, high) in bands.items():
              sos = sig.butter(4, [low, high], btype='band', fs=sfreq, output='sos')
              filtered = sig.sosfilt(sos, segment[ch])
              var = np.var(filtered)
              ch_feats.append(np.log(var + 1e-6))
          features.append(ch_feats)
      return np.array(features)  # shape (29, 5)
  ```

  This yields a \$29\times5\$ feature matrix per segment (29 nodes, 5 features each). We might standardize these features (e.g., z-score across training set) so that each feature dimension has mean 0 and unit variance, helping the model converge.

* **Wavelet Transform Features:** EEG being non-stationary, wavelet transform offers multi-scale analysis. We use DWT to decompose each channel’s time-series into sub-bands. For example, using a 4-level DWT with Daubechies-4 wavelet:
  $WT_{i}(t) \xrightarrow{\text{DWT}} \{cA^4_i, cD^4_i, cD^3_i, cD^2_i, cD^1_i\},$
  where \$cA^4\$ is the approximation (low-frequency) coefficients at level 4, and \$cD^j\$ are detail coefficients at levels 4,3,2,1 (higher frequency components). The number of coefficients depends on segment length and wavelet. We derive features by summarizing each coefficient set (e.g., energy or entropy):

  ```python
  import pywt, numpy as np
  def wavelet_features(segment, wavelet='db4', level=4):
      features = []
      for ch in range(segment.shape[0]):
          coeffs = pywt.wavedec(segment[ch], wavelet=wavelet, level=level)
          # Use log-energy of each coefficients vector as feature
          ch_feats = [np.log(np.mean(c**2) + 1e-6) for c in coeffs]
          features.append(ch_feats)
      return np.array(features)  # shape (29, level+1) -> here 29x5
  ```

  This produces 5 features per channel (one for each sub-band: the approximation + 4 detail levels). The rationale is similar to DE: capturing the power in different frequency ranges, but wavelet uses adaptive time windows (higher time resolution for high freq, and vice versa), which can capture transient patterns. In the MHA-GCN model, these “discrete wavelet features of each channel” are used as the graph node attributes.

* **Graph Construction (Adjacency):** For each segment’s feature matrix (from DE or DWT), we compute the adjacency matrix representing channel connectivity. Using Pearson correlation is a straightforward choice and was adopted in MHA-GCN. Formula:
  $A_{ij} = \mathrm{PCC}(X_i, X_j) = \frac{\mathrm{Cov}(X_i,X_j)}{\sigma_{X_i}\sigma_{X_j}},$
  where \$X\_i\$ is the time-series of channel \$i\$ (optionally we could use feature vectors instead of raw time series; in practice, using the raw EEG segment or band-limited signals works well for correlation). We implement adjacency as:

  ```python
  def compute_adjacency(segment):
      # segment shape: [29, time_samples]
      corr = np.corrcoef(segment)  # 29x29
      corr[np.isnan(corr)] = 0  # handle zero-variance
      # (Optional) set self-connections to 1
      np.fill_diagonal(corr, 1.0)
      return corr
  ```

  We do this per segment. Optionally, one might threshold \$A\$ to remove very low correlations (assuming they are noise) or use absolute values. The blueprint will use the full weighted graph. This adjacency encodes the brain network structure for that window.

* **EEG Time-Frequency Maps (alternative):** As an additional feature for the ViT branch (optional), one can create 2D time-frequency images of EEG. For example, apply STFT on each channel of a segment to get a spectrogram, then aggregate across channels. MHA-GCN extracted such 2D spectrograms for all 29 channels. One approach: compute a spectrogram per channel and either (a) take an average across channels (giving a single “mean EEG spectrogram” image) or (b) arrange channel spectrograms in a grid (e.g., 5×6 grid image). However, due to complexity and the existence of the GCN branch, we consider this optional. If implemented, we would:

  ```python
  import matplotlib.pyplot as plt
  # Example: average spectrogram
  freqs, times, Sxx = sig.spectrogram(segment, fs=sfreq, nperseg=128, noverlap=64)  # shape (29, freq_bins, time_bins)
  mean_Sxx = Sxx.mean(axis=0)  # average over channels -> shape (freq_bins, time_bins)
  plt.imshow(10*np.log10(mean_Sxx+1e-6), aspect='auto', origin='lower', extent=[times[0], times[-1], freqs[0], freqs[-1]])
  ```

  Then we would resize this to 224×224 and use it in a similar way to the speech spectrogram. For brevity, we proceed without this branch, focusing on GCN for EEG and ViT for speech, as that already demonstrates multimodal fusion.

**Speech Feature Extraction:** We largely follow standard audio processing, as already outlined:

* Pre-emphasis (coefficient 0.97) – implemented prior to STFT.
* STFT with Hamming window (25 ms) and 10 ms hop to produce spectrogram.
* Mel filter bank (e.g., 64 or 128 filters) applied to power spectrum, producing a Mel spectrogram.
* Log scaling (dB).

Mathematically, the STFT is:
$X(m, k) = \sum_{n=0}^{N-1} x[n+mH]\,w[n]\,e^{-j2\pi k n/N},$
where \$w\[n]\$ is the Hamming window of length \$N\$ (frame\_length) and \$H\$ is the hop length. We then compute the power \$|X(m,k)|^2\$ for each time frame \$m\$ and frequency bin \$k\$. The Mel filter bank is a set of weighting functions \$H\_m(k)\$ (as given in Eq. (4.4) in the paper) that sum the power over frequency with triangular weighting:
$P(m, \text{mel}_j) = \sum_{k} |X(m,k)|^2\,H_j(k),$
for Mel filter \$j\$. After filtering, we have a time × mel\_bin matrix. Taking \$\log\_{10}\$ gives log-magnitude (which is what the human ear’s loudness perception corresponds to, up to a scaling).

We already provided code using `librosa.feature.melspectrogram`. To integrate with the model, we convert each Mel spectrogram into a 224×224 RGB image (zero-pad or interpolate as needed). This image now represents the speech features.

**Example Code Snippet – Combined Extraction:** To illustrate the flow on a single subject (or one segment and one audio):

```python
# Example for one subject
raw = mne.io.read_raw_eeglab('subj01.set', preload=True)
raw.filter(1, 40)
raw.pick_channels(picks)
eeg_data = raw.get_data()        # 29 x T
# Assume audio file 'subj01.wav'
speech, sr = librosa.load('subj01.wav', sr=16000)
# Preprocess EEG
eeg_data = mne.preprocessing.ICA(n_components=29).fit(raw).apply(raw).get_data()
segments = np.split(eeg_data, eeg_data.shape[1]//(2*sfreq), axis=1)  # 2s segments
features_list, adj_list = [], []
for seg in segments:
    feats = differential_entropy_features(seg, sfreq=sfreq)  # or wavelet_features(seg)
    A    = compute_adjacency(seg)
    features_list.append(feats)
    adj_list.append(A)
# Preprocess Speech
speech_pre = np.append(speech[0], speech[1:] - 0.97*speech[:-1])
S = np.abs(librosa.stft(speech_pre, n_fft=512, hop_length=160, window='hamming'))**2
mel_spec = librosa.feature.melspectrogram(S=S, sr=sr, n_mels=64, fmax=8000)
mel_db = librosa.power_to_db(mel_spec, ref=np.max)
mel_img = cv2.resize(mel_db, (224, 224))
mel_img_rgb = np.repeat(mel_img[:, :, np.newaxis], 3, axis=2)  # make 3-channel
```

This code would yield `features_list` (each element 29×5 array), `adj_list` (29×29), and `mel_img_rgb` (224×224×3) for that subject. In practice, we loop over all subjects and aggregate these into training and testing sets as needed.

## 4. Explicit Model Implementation Details

We now detail the model architecture and how to implement it step by step. The model consists of two major components:

1. **EEG Graph Neural Network (MHA-GCN):** A Graph Convolutional Network with a multi-head attention mechanism for EEG channel graphs.
2. **Vision Transformer (ViT) for Spectrograms:** A ViT that processes 2D spectrogram images (speech Mel spectrograms, and optionally EEG spectrograms) to extract deep features.

These components produce outputs that are fused at the decision level (ensemble of predictions).

### EEG Graph Convolutional Network with Multi-Head Attention

**Graph Input:** Each EEG sample (segment) is represented as a graph \$G=(V, E)\$ with \$|V|=29\$ nodes (EEG channels) and edge weights given by the adjacency matrix \$A\$ (Pearson correlation). Each node has an attribute vector \$\mathbf{x}\_i \in \mathbb{R}^d\$ (e.g., \$d=5\$ features per channel from DE or DWT). We assemble the node features into a matrix \$X \in \mathbb{R}^{29\times d}\$.

**Graph Convolution Layers:** We use two layers of Graph Convolution to learn higher-level node embeddings from the initial features. A standard GCN layer (Kipf & Welling, 2017) is:
$H^{(l+1)} = \sigma\left(\tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2} H^{(l)} W^{(l)}\right),$
where \$\tilde{A}=A+I\$ (add self-loop), \$\tilde{D}\$ is the diagonal degree matrix of \$\tilde{A}\$, \$H^{(l)}\$ is the node feature matrix at layer \$l\$ (with \$H^{(0)}=X\$), \$W^{(l)}\$ is the trainable weight matrix of layer \$l\$, and \$\sigma\$ is an activation (ReLU). We adopt this formulation:

* **Layer 1:** Input \$X\$ (29×\$d\$) → GraphConv → \$H^{(1)}\$ (29×\$h\$) → ReLU →
* **Layer 2:** \$H^{(1)}\$ → GraphConv → \$H^{(2)}\$ (29×\$h\$) → ReLU.

We choose the hidden dimension \$h\$ (e.g., \$h=64\$). The output \$H^{(2)} = \[\mathbf{h}\_1, \mathbf{h}*2, ..., \mathbf{h}*{29}]^T\$ contains learned features for each node.

**Multi-Head Attention (Graph Attention) Module:** To enhance the GCN’s ability to capture complex node relationships, we apply a multi-head attention mechanism on the outputs of the GCN. The idea is similar to Graph Attention Networks (GAT), but here it’s a separate module after two GCN layers (per MHA-GCN design).

We have \$H^{(2)}\$ and we compute, for each attention head \$c = 1,\dots,C\$ (with \$C=4\$ heads in MHA-GCN):

* Queries: \$Q\_c = H^{(2)} W^q\_c\$, shape (29 × \$p\$)
* Keys: \$K\_c = H^{(2)} W^k\_c\$, shape (29 × \$p\$)
* Values: \$V\_c = H^{(2)} W^v\_c\$, shape (29 × \$p\_v\$)

Here \$W^q\_c \in \mathbb{R}^{h\times p}\$, \$W^k\_c \in \mathbb{R}^{h\times p}\$, \$W^v\_c \in \mathbb{R}^{h\times p\_v}\$ are learned projections for head \$c\$. We choose \$p = p\_v = h/C\$ so that all heads together span the original feature size (e.g., if \$h=64\$ and \$C=4\$, then \$p=16\$).

For each head \$c\$, we calculate attention coefficients between node \$i\$ and node \$j\$ (where \$j\$ is in the neighborhood of \$i\$, here the graph is fully connected with weights):
$e_{ij}^{(c)} = \frac{(Q_c[i] \cdot K_c[j])}{\sqrt{p}},$
and normalize across all neighbors \$j\$ of \$i\$ using softmax:
$\alpha_{ij}^{(c)} = \frac{\exp(e_{ij}^{(c)})}{\sum_{u \in \mathcal{N}(i)} \exp(e_{iu}^{(c)})},$
where \$\mathcal{N}(i)\$ are neighbors of \$i\$ (in our fully-weighted graph, effectively all other nodes + itself). This is the scaled dot-product attention. The attention coefficient \$\alpha\_{ij}^{(c)}\$ represents the importance of node \$j\$’s features to node \$i\$ in head \$c\$.

Then we compute the head’s output for node \$i\$ as the weighted sum of value vectors from its neighbors:
$z_i^{(c)} = \sum_{j \in \mathcal{N}(i)} \alpha_{ij}^{(c)} \, V_c[j].$
In matrix form, for head \$c\$, we get an output matrix \$Z^{(c)} \in \mathbb{R}^{29 \times p\_v}\$.

We concatenate the heads’ outputs \$Z^{(1)},...,Z^{(C)}\$ and project them with an output weight \$W^o\$:
$Z = \text{Concat}(Z^{(1)},...,Z^{(C)}) \cdot W^o,$
where \$W^o \in \mathbb{R}^{(C p\_v) \times h\_{\text{out}}}\$ (we can set \$h\_{\text{out}} = h\$ to keep dimension consistent). In practice, some implementations simply average the heads instead of concat for simplicity. MHA-GCN description indicates either concatenation then linear, or averaging (Equation 3.9 shows an average across heads):
$\mathbf{z}_i = \frac{1}{C}\sum_{c=1}^{C} \sum_{j \in \mathcal{N}(i)} \alpha_{ij}^{(c)} W^v_c \mathbf{h}_j^{(2)},$
which yields \$\mathbf{z}*i\$ as the new feature for node \$i\$ after attention. (This equation is a form of averaging heads' outputs, with \$v*{c,j}=W^v\_c h\_j\$.)

Thus, the multi-head attention module produces an updated node feature matrix \$Z \in \mathbb{R}^{29 \times h}\$ (assuming we set output dim = h). Intuitively, each node’s feature now encodes a weighted combination of other nodes’ features, where weights are learned by attention – this can capture important inter-channel interactions not caught by GCN alone.

**Classification Layer (Graph branch):** Now we have, for each EEG segment, a matrix \$Z\$ of size 29×\$h\$. We need to produce a classification (depressed vs control). We convert the node features into a graph-level feature. Common ways:

* Take a simple average or sum over node features: \$\mathbf{g} = \frac{1}{29}\sum\_{i=1}^{29} \mathbf{z}\_i\$. This yields a \$h\$-dimensional graph embedding.
* Or apply another attention pooling (learn weights per node to sum).
* Or even take the “importance scores” \$z\_i\$ computed – note \$z\_i\$ in Eq. (3.9) was called an "importance score" for node \$i\$ (if \$pv=1\$ that would be scalar, but here it’s a vector; possibly in the paper they set \$pv=1\$ to get scalar importance per node).

For simplicity, we do average pooling:
$\mathbf{g} = \frac{1}{29}\sum_{i=1}^{29} \mathbf{z}_i \in \mathbb{R}^{h}.$
Then \$\mathbf{g}\$ goes through a final Fully Connected layer that maps \$\mathbb{R}^h \to \mathbb{R}^2\$ (for binary classification) or \$\mathbb{R}^L\$ for \$L\$ depression levels:
$ \mathbf{o}_{EEG} = \text{softmax}( \mathbf{W}_{out} \mathbf{g} + \mathbf{b}_{out} ),$
where \$\mathbf{W}\_{out} \in \mathbb{R}^{2 \times h}\$.

**Summary of GCN Branch Implementation:** We can implement this in PyTorch as a custom `nn.Module`. Pseudocode:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MHAGCN(nn.Module):
    def __init__(self, in_feats, hidden_feats, num_heads, num_classes):
        super().__init__()
        # GraphConv weights (using simplest form without bias here)
        self.W0 = nn.Parameter(torch.randn(in_feats, hidden_feats))
        self.W1 = nn.Parameter(torch.randn(hidden_feats, hidden_feats))
        # Attention weights
        self.num_heads = num_heads
        head_dim = hidden_feats // num_heads
        self.Wq = nn.Parameter(torch.randn(num_heads, hidden_feats, head_dim))
        self.Wk = nn.Parameter(torch.randn(num_heads, hidden_feats, head_dim))
        self.Wv = nn.Parameter(torch.randn(num_heads, hidden_feats, head_dim))
        self.Wo = nn.Parameter(torch.randn(num_heads * head_dim, hidden_feats))
        # Output classifier
        self.fc = nn.Linear(hidden_feats, num_classes)
    def forward(self, X, A):
        # X: (batch, 29, in_feats), A: (batch, 29, 29)
        # GCN Layer 1
        A_norm = normalize_adj(A)         # compute \tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}
        H1 = F.relu(A_norm @ X @ self.W0)  # (batch, 29, hidden)
        # GCN Layer 2
        H2 = F.relu(A_norm @ H1 @ self.W1)  # (batch, 29, hidden)
        # Multi-head attention:
        # Compute queries, keys, values for each head
        # shape transformations: we'll get (batch, num_heads, 29, head_dim)
        Q = torch.einsum('bhf,hfd->bhd', H2, self.Wq)  # or loop over heads
        K = torch.einsum('bhf,hfd->bhd', H2, self.Wk)
        V = torch.einsum('bhf,hfd->bh*d', H2, self.Wv)  # '*' for head dim possibly same as above
        # Calculate attention scores
        # Expand dims for broadcasting: Q (batch, heads, 29, head_dim), K similarly
        Q = Q.unsqueeze(2)  # (batch, heads, 1, 29, head_dim) if needed
        K = K.unsqueeze(3)  # (batch, heads, 29, 1, head_dim)
        scores = torch.sum(Q * K, dim=-1) / (head_dim ** 0.5)  # (batch, heads, 29, 29)
        alpha = torch.softmax(scores, dim=-1)  # normalize over j (last dim)
        # Weighted sum of values
        Z_head = torch.matmul(alpha, V)  # (batch, heads, 29, head_dim)
        # Combine heads
        Z_cat = Z_head.transpose(1,2).reshape(X.size(0), 29, -1)  # (batch, 29, heads*head_dim)
        Z = Z_cat @ self.Wo  # (batch, 29, hidden_feats)
        # Pooling
        g = Z.mean(dim=1)  # (batch, hidden_feats)
        out = self.fc(g)    # (batch, num_classes)
        return out
```

*(The above code uses Einstein summation or manual broadcasting for clarity; an optimized implementation might vectorize differently. `normalize_adj(A)` computes \$\tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}\$ for each graph in the batch.)*

During forward pass:

* We feed `X` (node features) and `A` (adjacency) for either one sample or a batch of samples.
* Compute normalized adjacency \$\tilde{A}\$: we add an identity to A (self-loop) and then \$\tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2}\$. This can be done per graph in batch. If using torch\_geometric, `GCNConv` would handle this normalization internally.
* Compute first GCN layer: matrix multiply \$A\_{norm} X W0\$ then ReLU.
* Second GCN similarly.
* Then attention: for simplicity, the pseudocode treats each sample individually. We compute queries, keys, values for each head by projecting \$H2\$. Then compute attention scores \$\alpha\_{ij}^{(c)}\$ and new features \$Z\$.
* Pool and classify.

We have to ensure the model handles variable batch sizes and uses GPU if available (which PyTorch does if `.to(device)` moves the model and data).

**Activation & Regularization:** We used ReLU after each GCN. We can also include dropout layers to avoid overfitting (e.g., dropout on \$H1\$ or on \$g\$ before FC). L2 regularization can be applied via optimizer weight decay.

**Multi-Class Variation:** If classifying depression severity (e.g., 4 classes: healthy, mild, moderate, severe), `num_classes=4` in the final layer and use softmax accordingly. The rest of the architecture remains the same.

### Vision Transformer (ViT) for Spectrograms

The ViT processes 2D images by splitting them into patches, embedding those patches, and applying transformer layers. For our case:

* Input: a spectrogram image, e.g., 224×224 pixels, 3 channels (RGB or fake-RGB).
* The model configuration is **ViT-Base Patch16/224**, meaning:

  * Patch size = 16×16.
  * Image size = 224×224, so there are \$(224/16)^2 = 14\times14 = 196\$ patches.
  * Hidden embedding size = 768 (each patch is linearly mapped to a 768-dim vector).
  * Number of transformer layers = 12.
  * Number of self-attention heads per layer = 12 (each of size 64 since 64×12=768).
  * MLP feedforward dimension = 3072 in each layer.
  * There is a learnable \[CLS] token used to aggregate the patch information, and learned 1D positional embeddings for each of the 197 positions (196 patches + 1 CLS).

**ViT Forward Pass:** Summarized:

1. **Patch Embedding:** Flatten each 16×16 patch of the image (for a single-channel image that’s 256 values; for 3-channel, 256×3=768 values per patch) and linearly project it to the model dimension (768). In ViT-Base, with 3-channel input, the linear projection actually has input dimension \$16\times16\times3=768\$ and output 768, essentially acting like a fixed linear identity in this case. (If our image were single-channel and we duplicated it to 3-channel, it aligns with pre-trained weights expecting 3 channels.)
   We denote patch embeddings as \$E\_p \in \mathbb{R}^{196 \times 768}\$.
2. **Add \[CLS] token:** Introduce a learnable embedding \$\mathbf{e}*{CLS} \in \mathbb{R}^{768}\$ that will represent the whole image. Form the sequence: $\[\mathbf{e}*{CLS}; E\_{p1}; E\_{p2}; \dots; E\_{p196}]\$ of length 197.
3. **Add Positional Encoding:** Add a learned positional vector to each element of the sequence to retain patch order information. There is a vector for position 0 (CLS) and 1–196 for patches (learned parameters).
4. **Transformer Encoder:** Pass the sequence through \$L=12\$ transformer layers. Each layer does:

   * Multi-head self-attention: 12 heads, each computes attention among all 197 tokens (including CLS) with key/query/value of dim 64. So similar mechanism as described for GCN attention but fully dense among patches.
   * Add & Norm, then a Feed-Forward (MLP) of two linear layers with GeLU nonlinearity (768 -> 3072 -> 768).
   * Another Add & Norm. (Following standard Transformer architecture.)
     The outcome after 12 layers is a sequence of the same length 197, now containing high-level features.
5. **Classification Head:** Take the final hidden state corresponding to the \[CLS] token (which is a 768-d vector containing aggregated information from all patches). Feed it into a linear layer that outputs \$n\$ classes. In a pre-trained ViT model, a dense layer is often trained for ImageNet classes. We will replace that with our own head for depression classification.

**Implementation Approach:** We will leverage a pre-trained ViT model from libraries to avoid coding the full transformer (although it’s feasible, it’s lengthy). Using timm (PyTorch image models) or HuggingFace Transformers:

```python
import timm
vit = timm.create_model('vit_base_patch16_224', pretrained=True)
# Replace head
vit.head = nn.Linear(vit.head.in_features, num_classes)
```

This gives us a ViT that is pre-trained on ImageNet and will fine-tune to our task with a new head (randomly initialized). Pre-training should help given our dataset is small. If using HuggingFace:

```python
from transformers import ViTForImageClassification
vit_model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
vit_model.classifier = torch.nn.Linear(768, num_classes)
```

(This assumes the classifier is named `classifier` in that model.)

We will create two instances: `vit_eeg` (if we include EEG spectrograms) and `vit_speech`. In our blueprint, we mainly use `vit_speech` for the Mel spectrograms.

**Preparing Inputs:** The spectrogram images should be normalized similarly to the images the ViT was trained on. ViT base was trained on ImageNet, mean = (0.5,0.5,0.5), std = (0.5,0.5,0.5) if using the default normalization, or actual ImageNet mean/std. For simplicity, we could scale pixel values to 0–1 and then to -1–1. Alternatively:

```python
# Normalize image tensor (assuming mel_img_rgb is 224x224x3 numpy)
img_tensor = torch.tensor(mel_img_rgb).permute(2,0,1).float()  # shape 3x224x224
img_tensor = img_tensor/255.0  # if not already 0-1
# Subtract mean and divide std (using ImageNet stats):
mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
img_tensor = (img_tensor - mean) / std
```

We will include such normalization in the data preprocessing pipeline for the ViT branch.

**Forward Pass:** The ViT model will take the image tensor and output class logits. Using the HuggingFace `ViTForImageClassification`, the forward call returns a `logits` tensor of shape (batch\_size, num\_classes). If using timm or a custom model, we get the same. The forward includes patching, adding \[CLS], etc., internally.

**Integration with Ours:** For training, we’ll treat `vit_speech` as another model that outputs \$\mathbf{o}*{speech}\$. Similarly, if an `vit_eeg_img` was included, it would output \$\mathbf{o}*{eegImg}\$. Each output is a probability distribution over classes after softmax.

**Loss Functions:** We will use cross-entropy loss for each branch:

* For binary classification, use `nn.BCEWithLogitsLoss` (if output not softmax) or `nn.CrossEntropyLoss` (with logits of size 2).
* For multi-class (severity levels), `nn.CrossEntropyLoss` is appropriate (the logits size = number of classes).

**Decision-Level Fusion:** Rather than merging features early, the design is to combine decisions from each modality branch. The method described in MHA-GCN ViT uses a weighted sum of the output probabilities from each classifier. Specifically, if \$m\$, \$n\$, \$v\$ are the predicted probability vectors from (for example) the GCN branch, EEG-ViT branch, and speech-ViT branch respectively, and \$\varepsilon\_1, \varepsilon\_2, \varepsilon\_3\$ are their validation accuracies, then:
$p_{\text{fusion}} = \varepsilon_1 m + \varepsilon_2 n + \varepsilon_3 v,$
and the predicted label is \$\arg\max(p\_{\text{fusion}})\$. In our implementation, we will:

* During training, each branch is trained with its own loss. We could sum the losses or alternate training. In a simple ensemble, it’s common to train each separately.
* During validation/testing, we compute the softmax probabilities of each branch’s output. We then weight and sum them. If we lack a separate validation set to estimate accuracies as weights, we might use equal weights or the training accuracy as a proxy. Alternatively, treat the weights as hyperparameters (could even be learned by a small logistic regression on top of the concatenated outputs, but that adds complexity).
* For this blueprint, we will assume equal weights or manually set weights if one modality is known to perform better. (We note the paper reported precision of \~90% for speech vs \~89% for EEG, etc., which are comparable, but for demonstration we might just average.)

**Model Training Strategy:** We have two sub-networks (GCN and ViT). We can train them in two ways:

* **Joint training:** Construct a single combined model that outputs three logits (one per branch) and one fused output, and minimize a combined loss. For example, \$Loss = \text{CE}(o\_{fusion}, label) + \lambda\_1 \text{CE}(o\_{EEG}, label) + \lambda\_2 \text{CE}(o\_{speech}, label)\$. This way, gradients flow in each branch. However, since fusion is at decision (non-differentiable argmax or linear combination outside softmax), joint training would require us to combine earlier (e.g., feature fusion) or just train with each branch’s loss.
* **Separate training:** Train GCN and ViT independently on the training data to minimize their own classification losses. Then at test time do ensemble. This is simpler and mimics training separate expert models.

Given the architecture is modular and to avoid complex multi-objective training, we choose **separate training**. That means our training loop will optimize `MHAGCN` with EEG training data, and optimize `vit_speech` with speech training data. We must be careful in cross-validation to not use test subject data in either.

During a CV fold:

1. **Train GCN model:** Use all EEG segments of training subjects. Optimize `criterion_EEG(o_EEG, label)`. For binary classes, use cross-entropy.
2. **Train ViT model:** Use all speech spectrograms of training subjects. Optimize `criterion_speech(o_speech, label)`.
3. Optionally, if we had EEG spectrogram branch, train that too similarly.
4. **Validation:** For the held-out subject(s), collect predictions:

   * If the test subject has multiple EEG segments, get predictions for each segment from the GCN. Average those segment predictions to get a subject-level probability (since the true label is per subject). For example, if 10 out of 15 segments are predicted “depressed”, we might lean towards depressed. A better way is to average the predicted probability of “depressed” across segments to get a probability for the subject.
   * Get the speech model prediction for the subject’s audio.
   * Combine these predictions via weighted sum. For binary classification, that might be as simple as computing weighted sum of “depressed” probabilities. For multi-class, we do vector sum.
   * Decide final label and compare to ground truth.

We also consider performance metrics per fold.

**Example Code – Model Definition and Fusion:**

```python
# Instantiate models
gcn_model = MHAGCN(in_feats=5, hidden_feats=64, num_heads=4, num_classes=2)
vit_model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=2)

# Optimizers
opt_gcn = torch.optim.Adam(gcn_model.parameters(), lr=1e-3)
opt_vit = torch.optim.Adam(vit_model.parameters(), lr=1e-4)

# Loss function
criterion = nn.CrossEntropyLoss()

# Training loop for one epoch (GCN as example)
for features, adj, label in train_graph_loader:   # assume DataLoader provides batched graphs
    features = features.to(device)  # shape [batch, 29, 5]
    adj = adj.to(device)            # shape [batch, 29, 29]
    label = label.to(device)        # shape [batch]
    opt_gcn.zero_grad()
    out = gcn_model(features, adj)  # shape [batch, 2]
    loss = criterion(out, label)
    loss.backward()
    opt_gcn.step()

# Similar loop for vit_model on speech data:
for images, label in train_spectrogram_loader:
    images = images.to(device)    # shape [batch, 3, 224, 224]
    label = label.to(device)
    opt_vit.zero_grad()
    out = vit_model(images)       # out.logits if using HF model
    logits = out if isinstance(out, torch.Tensor) else out.logits
    loss = criterion(logits, label)
    loss.backward()
    opt_vit.step()

# After training, evaluate on test subject
gcn_model.eval()
vit_model.eval()
# Suppose test_subject_segments is a list of (features, adj) for each EEG segment of test subj
eeg_preds = []
with torch.no_grad():
    for features, adj in test_subject_segments:
        features = torch.tensor(features, device=device).unsqueeze(0)  # [1,29,5]
        adj = torch.tensor(adj, device=device).unsqueeze(0)            # [1,29,29]
        logit = gcn_model(features, adj)  # [1,2]
        prob = torch.softmax(logit, dim=-1)
        eeg_preds.append(prob.cpu().numpy())
eeg_mean_prob = np.mean(eeg_preds, axis=0)  # average over segments -> shape [1,2]

# For speech:
speech_image = prepare_image(test_subject_mel)  # returns torch tensor [1,3,224,224]
with torch.no_grad():
    out = vit_model(speech_image.to(device))
    speech_prob = torch.softmax(out if isinstance(out, torch.Tensor) else out.logits, dim=-1)
    speech_prob = speech_prob.cpu().numpy()  # shape [1,2]

# Decision fusion (equal weight example)
final_prob = 0.5 * eeg_mean_prob + 0.5 * speech_prob  # both are 1x2 arrays
final_label = np.argmax(final_prob, axis=1)
```

This code demonstrates classification for one test subject. In practice, we wrap it in the CV loop to aggregate results.

**Multi-class adjustments:** Use `softmax` and average predictions similarly but taking care to average probability vectors.

**Attention to Indices:** The label association: In `train_graph_loader`, each segment carries its subject’s label. We must ensure all segments from depressed subjects are labeled 1, etc. This is done when preparing `eeg_graphs`. For speech, each spectrogram gets the subject’s label.

**Using PyTorch Geometric (optional):** We could leverage `torch_geometric.data.Data` for each graph (with `x` for node features and `edge_index` or adjacency for edges) and `torch_geometric.nn.GCNConv` for layers, and perhaps implement a custom attention layer. But given our small graph (29 nodes) and custom MHA, pure PyTorch is fine.

## 5. Training and Validation Loop Logic

We now articulate the training procedure and validation logic in detail, including cross-validation handling and hyperparameter choices.

**Training Procedure:** We train in multiple stages corresponding to each data modality, then fuse:

* **EEG-GCN Training:** Train `MHAGCN` on EEG training data (all segments from training subjects).
* **Speech-ViT Training:** Train `ViT` on speech training data (all spectrograms from training subjects).
* *(If using an EEG spectrogram ViT branch, train it similarly on EEG spectrograms from training subjects.)*

We will use **Leave-One-Out Cross-Validation** as default. Pseudocode outline:

```python
subjects = np.array(subject_ids)  # list/array of subject IDs
all_metrics = []  # to collect metrics for each fold

for test_idx in range(len(subjects)):
    test_subj = subjects[test_idx]
    train_subjs = np.delete(subjects, test_idx)
    # Prepare training datasets for this fold:
    train_graphs = [g for g in eeg_graphs if g[2] in train_subjs]
    train_images = [img for img in speech_images if img[1] in train_subjs]
    test_graphs  = [g for g in eeg_graphs if g[2] == test_subj]
    test_images  = [img for img in speech_images if img[1] == test_subj]
    # Initialize models (or reset weights if carried from previous fold)
    gcn_model = MHAGCN(...); vit_model = VitClassifier(...)
    # Optimizers and schedulers
    opt_gcn = Adam(gcn_model.parameters(), lr=1e-3)
    opt_vit = Adam(vit_model.parameters(), lr=1e-4)
    # Train for N epochs:
    for epoch in range(1, N+1):
        gcn_model.train(); vit_model.train()
        # Optionally shuffle training data each epoch
        for batch in DataLoader(train_graphs, batch_size=32, shuffle=True):
            X, A, labels = batch  # assume batch collates correctly
            opt_gcn.zero_grad()
            out = gcn_model(X, A)
            loss = criterion(out, labels)
            loss.backward()
            opt_gcn.step()
        for batch in DataLoader(train_images, batch_size=32, shuffle=True):
            imgs, labels = batch
            opt_vit.zero_grad()
            out = vit_model(imgs)
            logits = out if isinstance(out, torch.Tensor) else out.logits
            loss = criterion(logits, labels)
            loss.backward()
            opt_vit.step()
        # (Optional) evaluate on a validation subset of train_subjs for early stopping or to adjust ensemble weights
    # After training, evaluate on test_subj:
    gcn_model.eval(); vit_model.eval()
    # Get aggregated EEG prediction for test_subj:
    eeg_probs = []
    for X, A, lbl in test_graphs:
        prob = softmax(gcn_model(X.unsqueeze(0), A.unsqueeze(0)), dim=-1)
        eeg_probs.append(prob.detach().cpu().numpy())
    eeg_probs = np.mean(eeg_probs, axis=0)  # average probabilities
    # Get speech prediction (if available):
    if len(test_images) > 0:
        # There should typically be one speech sample per test_subj
        img, lbl = test_images[0]
        prob = softmax(vit_model(img.unsqueeze(0)).logits, dim=-1)
        speech_prob = prob.detach().cpu().numpy()
    else:
        speech_prob = None
    # Fuse decisions:
    if speech_prob is None:
        final_prob = eeg_probs
    else:
        final_prob = (eeg_probs * w_eeg + speech_prob * w_speech) 
    pred_label = argmax(final_prob)
    true_label = 1 if test_subj in depressed_ids else 0
    # Record metrics (e.g., accuracy for this subj, confusion matrix increment, etc.)
    all_metrics.append(pred_label == true_label)
```

Key points:

* **Epochs:** We choose the number of epochs N. Prior work (DepL-GCN) used 100 epochs for convergence. We can similarly set N=100 per fold. Early stopping could be applied if a small validation from train is set aside, but in LOOCV with few subjects, it may be unstable. Instead, use a fixed N or monitor training loss convergence.

* **Batch size:** If each EEG segment is a sample, we have many more EEG samples than subjects (e.g., \~150 segments \* (52 training subjects) ≈ 7800 segments). A batch size of 32 or 64 is reasonable. For speech, if each subject has one spectrogram, the number of training samples = number of training subjects (\~52-1=51). We can either do batch = all (since 51 is small) or augment audio by splitting it into chunks to have more training examples. Simpler: batch=1 or few for speech (since each sample is large and ViT memory usage is high). We might actually train the speech ViT in batch mode of small size (say 8 or 16 if we generate smaller segments or augment).

* **Optimizer and LR:** The GCN branch (with \~ few thousand parameters) can use a larger LR (1e-3). The ViT (with millions of parameters) should use a smaller LR (1e-4 or 5e-5) when fine-tuning to avoid drastic weight updates. We might also freeze some early layers of ViT initially. For simplicity, we keep them all trainable but small LR.

* **Monitoring training:** We will print training progress: e.g., each epoch print loss or after full training print final training accuracy for both branches.

* **Validation set weights:** The ensemble weight \$\varepsilon\_i\$ for each branch ideally comes from validation performance. In LOOCV, we don’t have a separate validation per fold unless we carve one from training (e.g., take 10% of train subjects as val to get an idea of branch accuracy). We can do that:

  * For each fold, after training, evaluate GCN on its train subjects (via segments) and compute accuracy, and same for ViT on train subjects. These give an estimate of how well each branch fits the training data (which might be optimistic). Alternatively, use cross-validation itself: we could use the average accuracy of each branch across all folds (from training or testing) as weights for final combination.
  * The paper assumed known validation accuracies. For the blueprint, we might simply set weights equally or based on overall performance ranking. As an example, if in prior studies EEG branch had \~87% acc and speech \~85%, one might weight EEG slightly higher. But to stick to references: we can cite using the reported five-fold CV precision etc. However, they specifically mention using validation set accuracies \$\varepsilon\$, which implies they held out some validation.
  * We will proceed with equal weights in code for simplicity, but will mention that one can tune these weights.

* **Combining multiple EEG segments:** As shown, we average probabilities over segments for the test subject. This effectively does a majority vote by probabilities. Alternatively, sum logits or majority vote by predictions are options. Averaging probabilities is a smooth approach. If multi-class, we average the probability vector.

* **Computing Metrics:** For each test subject, we know true label. We accumulate:

  * Total correct predictions (to compute overall accuracy = correct/total).
  * For precision/recall, we accumulate confusion matrix:

    * True Positives (TP): depressed predicted depressed.
    * True Negatives (TN): healthy predicted healthy.
    * False Positives (FP): healthy predicted as depressed.
    * False Negatives (FN): depressed predicted as healthy.
  * Then \$Accuracy = \frac{TP+TN}{TP+TN+FP+FN}\$,
    \$Precision = \frac{TP}{TP+FP}\$ (for depressed class),
    \$Recall = \frac{TP}{TP+FN}\$,
    \$F1 = 2 \frac{\text{Precision}\cdot \text{Recall}}{\text{Precision}+\text{Recall}}\$.
    We output these metrics per fold or averaged. In five-fold CV, we’d average metrics across folds. In LOOCV, we can just compute an aggregate confusion and then metrics (which is effectively the same as averaging since each test is one subject).

* **Example metric output:** Suppose we got 90% accuracy, we’ll display:

  * “Accuracy: 90%, Precision: X%, Recall: Y%, F1: Z%”. These can be calculated and printed.
    We will also compare to reference: MHA-GCN ViT got 89.03% accuracy, 90.16% precision, 89.04% recall, 88.83% F1, which we can mention if matching.

* **Loop Efficiency:** LOOCV will train 53 separate models for each branch. This is expensive, but since each model is not huge (except ViT, which is heavy). One way to speed up is to train one model on all data and do subject-wise hold-out evaluation (not exactly LOOCV because the model sees that subject’s data, which is not allowed). So we indeed need to retrain for each left-out to be rigorous. With 53 folds and two models each, expect long runtime. If needed, one could reduce epochs or use 5-fold for quicker demo.

**Leave-One-Out vs k-Fold Implementation:** We could add a parameter to switch. For clarity, we illustrate LOOCV explicitly as above. For 5-fold, the code is similar but the outer loop is 5 iterations dividing `subjects` into train/test splits by fold.

**Visualization of Training Loop:** We will include in the notebook some real-time plots of training loss, or print statements each epoch:

```python
print(f"Epoch {epoch}: GCN loss = {loss_gcn:.4f}, ViT loss = {loss_vit:.4f}")
```

or after training:

```python
print(f"Fold {fold}: Test Accuracy = {acc_fold:.2%}")
```

to track progress. We can store losses per epoch and then plot them after training (like the training and accuracy curves shown in Figure 8).

**Ensuring Reproducibility:** Set random seeds (`torch.manual_seed(0)`, `np.random.seed(0)`), and possibly make results deterministic by certain backend flags if needed. We also ensure to shuffle data consistently in DataLoaders and perhaps stratify splits in 5-fold (i.e., each fold has similar ratio of depressed/control).

## 6. Interactive Visualization – Concrete Definitions

To make the notebook interactive and insightful, we include visualization components that allow exploration of the data and the model’s behavior. Below are concrete interactive visualization features:

* **EEG Channel Graph Visualization:** We provide an interactive plot of the EEG channel network for different subjects. For example, a dropdown menu of subject IDs (or a toggle for “depressed” vs “healthy average”) that, when selected, displays the graph where nodes are EEG channels (plotted according to their approximate scalp locations) and edges weighted by correlation. We can use Plotly’s network graph or `networkx` with matplotlib:

  * Nodes are color-coded by lobe (frontal, temporal, etc.) or degree of connectivity.
  * Edge thickness represents the correlation strength between channels.
  * For a depressed subject, one might observe stronger overall connectivity (just hypothetical; we’d need to confirm from data) vs a healthy control.
  * **Interaction:** The user can select a subject, and an `ipywidgets` callback will recompute the network for that subject’s resting EEG. Alternatively, we can pre-compute average adjacency for all depressed vs all healthy and let the user toggle between those two graphs to see differences in brain network structure.

  *Implementation:* Use `widgets.Dropdown` with options = subject IDs, and an `on_change` that updates a Plotly figure (Plotly can be used in offline mode within Jupyter). Example code snippet:

  ```python
  import plotly.graph_objects as go
  from ipywidgets import Dropdown, Output

  dropdown = Dropdown(options=[('Subject ' + str(s), s) for s in subjects], description='Subject:')
  out = Output()
  def update_graph(change):
      subj = change.new
      # compute average adjacency across all segments of subj
      A = avg_adj_by_subject[subj]  # assume we precomputed this
      # create edge list for plot: threshold for visibility
      edges = [(i,j,A[i,j]) for i in range(29) for j in range(i+1,29) if A[i,j] > 0.5]
      with out:
          out.clear_output()
          fig = go.Figure()
          # add edges as line segments
          for (i,j,w) in edges:
              xi, yi = channel_locs[i]; xj, yj = channel_locs[j]
              fig.add_trace(go.Scatter(x=[xi,xj], y=[yi,yj],
                                       line=dict(width=w*5, color='blue'),
                                       hoverinfo='none', mode='lines'))
          # add nodes
          fig.add_trace(go.Scatter(x=[loc[0] for loc in channel_locs],
                                   y=[loc[1] for loc in channel_locs],
                                   mode='markers+text', text=channel_names,
                                   marker=dict(size=10, color='red'),
                                   textposition="top center"))
          fig.update_layout(title=f"EEG Channel Network for Subject {subj}", showlegend=False)
          fig.show()
  dropdown.observe(update_graph, names='value')
  display(dropdown, out)
  ```

  This code (conceptual) would display a graph when a subject is selected. We assume `channel_locs` is a list of 2D coordinates for each of the 29 channels (we can use standard 10-20 layout coordinates).

* **Spectrogram Visualization:** We allow interactive viewing of EEG vs speech spectrograms. For instance:

  * A toggle to select **EEG spectrogram** or **Speech spectrogram** and a subject (and maybe specific EEG channel).
  * For EEG, if user selects a subject and channel, we plot that channel’s time-frequency spectrogram (perhaps using a wavelet scalogram or STFT) comparing a depressed vs healthy subject side by side.
  * For speech, if user selects a subject, we show that subject’s Mel spectrogram (the input to the model). A depressed subject’s mel spectrogram might show differences in pitch or energy distribution (e.g., depressed speech often has monotonic low energy).
  * The interactive element could be a slider through time as well, but since we usually use the whole spectrogram image, a simple static image per subject suffices.

  Implementation: Use `widgets.Dropdown` for subject and a radio button for modality (EEG vs speech). On change, update a matplotlib or Plotly heatmap:

  ```python
  subj_dd = Dropdown(options=subjects, description="Subject")
  mod_dd = Dropdown(options=['EEG Spectrogram','Speech Mel'], description="Modality")
  out2 = Output()
  def update_spec(change):
      subj = subj_dd.value; mod = mod_dd.value
      with out2:
          out2.clear_output()
          if mod == 'Speech Mel':
              # plot mel spectrogram
              spec = subject_to_mel[subj]  # 2D array
              plt.figure(figsize=(6,4))
              plt.title(f"Speech Mel Spectrogram - Subj {subj}")
              plt.imshow(spec, origin='lower', aspect='auto', cmap='magma')
              plt.colorbar(label='Power (dB)')
              plt.xlabel('Time'); plt.ylabel('Mel frequency')
              plt.show()
          else:
              # plot EEG spectrogram for a representative channel or an average
              chan = 'Cz'  # for example
              sig = get_channel_signal(subj, chan)
              f,t,Sxx = sig.spectrogram(sig, fs=250, nperseg=250, noverlap=200)
              plt.figure(figsize=(6,4))
              plt.title(f"EEG Spectrogram ({chan}) - Subj {subj}")
              plt.pcolormesh(t, f, 10*np.log10(Sxx), shading='gouraud', cmap='viridis')
              plt.colorbar(label='Power (dB)')
              plt.xlabel('Time (s)'); plt.ylabel('Frequency (Hz)')
              plt.show()
  subj_dd.observe(update_spec, names='value'); mod_dd.observe(update_spec, names='value')
  display(mod_dd, subj_dd, out2)
  ```

  This snippet would show a dynamic spectrogram when the user selects subject or modality.

* **Model Attention and Feature Importance:** Visualize what the model learned:

  * For the GCN, we can plot the **node attention weights** \$\alpha\_{ij}\$ or aggregated node importances. For each test subject’s EEG graph, after multi-head attention, we had importance scores \$z\_i\$ for each node. If we take the norm or average of \$\mathbf{z}\_i\$ across features, that can indicate how “important” channel \$i\$ was in making the decision. For instance, we could show a topographic map highlighting which brain regions the model paid most attention to for a depressed subject vs a control.

    * Implementation: Take \$Z\$ from the model (`Z = Z_cat @ W^o` from code). Compute something like node\_importance = `Z.var(dim=1)` or `Z.norm(dim=1)` for each node. Then plot those values on a head map (or as a bar chart). Possibly interactive for selecting a subject or an average over groups.
  * For the ViT, we can extract the self-attention weights of the CLS token in the last layer to see which patches were most attended to for classification (like in Vision Transformer visualization, one can sometimes project attention to image space). This is advanced, but we might show which time-frequency regions contributed. Simpler: compare averaged spectrograms between classes.

  For blueprint, we may keep it simpler:

  * Provide a static figure (or interactive if time) showing, for example, the **mean node importance** across all depressed vs all controls. E.g., “frontal channels have higher importance for depression classification” if that emerges.
  * Possibly refer to any figure from the paper if available (though the paper doesn’t show attention visualization, but others might).

  Implementation example for node importance:

  ```python
  # Suppose we have a trained model and a batch of segments for a subject
  X = torch.tensor(segment_features).unsqueeze(0)
  A = torch.tensor(segment_adj).unsqueeze(0)
  gcn_model.eval()
  with torch.no_grad():
      # Get node features after attention (Z)
      H1 = F.relu(A_norm @ X @ gcn_model.W0)
      H2 = F.relu(A_norm @ H1 @ gcn_model.W1)
      # Compute attention as done in forward (we might need to modify model to output Z for viz)
      # For simplicity, assume we can get an intermediate output Z
      Z = gcn_model.get_attention_output(X, A)  # Not implemented in our earlier code but imagine it returns Z
  Z = Z.squeeze(0).cpu().numpy()  # 29 x h
  importance = np.linalg.norm(Z, axis=1)  # 29 values
  plt.bar(range(29), importance); plt.xticks(range(29), channel_names, rotation=90)
  plt.title("Node importance scores for subject X")
  plt.show()
  ```

  This would yield a bar chart. We could make this interactive by allowing the user to choose which subject’s importance to view.

* **Performance Visualization:** Finally, to help interpret model performance, we display:

  * The **training curves**: After training each fold, we collected loss and accuracy per epoch for GCN and ViT. We plot these curves (loss vs epoch, accuracy vs epoch) for a representative fold, or average across folds. This corresponds to Figure 8 (training and accuracy curves) in the paper.
  * The **confusion matrix** for the final model (aggregated over CV). We can show this as a 2x2 matrix image or as text. Possibly interactive: user clicks a button to show confusion matrix or toggle between normalized and raw counts.

  Example:

  ```python
  import seaborn as sns
  cf_matrix = np.array([[TN, FP],[FN, TP]])
  sns.heatmap(cf_matrix, annot=True, fmt='d', cmap='Blues',
              xticklabels=['Predicted HC','Predicted MDD'],
              yticklabels=['Actual HC','Actual MDD'])
  plt.title("Confusion Matrix")
  plt.show()
  ```

  This gives a static confusion matrix after CV.

All these visualizations serve to make the notebook a **digital twin dashboard** where one can inspect both the input data characteristics and the model’s learned representations and decisions in an interactive manner.

To implement interactivity, we rely on `ipywidgets` and dynamic output areas as shown, which the AI agent can set up. The notebook’s narrative will guide the user on how to use the controls (e.g., "Select a subject from the dropdown to view their brain network connectivity").

**Summary of Interactivity:** The blueprint defines:

* Dropdowns for subject selection (for graph and spectrogram).
* Automatic plot updates in response.
* Visual plots for model interpretation (node importance, training curves, confusion matrix).
  We ensure each interactive cell has explanatory text.

By interacting with these, users can, for example, **compare a depressed subject’s EEG network to a healthy control’s**, observe **differences in speech spectrogram patterns**, and understand **which EEG channels or time-frequency regions the model focuses on**. This enriches the digital twin experience, turning raw model outputs into meaningful insights about the patient’s state.

## 7. Implementation of Broader “Implementation Requirements”

Beyond core model training, we address broader requirements to integrate this solution into a **Psychiatry Digital Twin** system, ensuring robustness, fairness, and extensibility.

**Handling Class Imbalance and Severity Levels:** In our dataset, if class distribution is imbalanced (e.g., slightly more healthy controls), we should ensure the model is not biased. DepL-GCN introduced a **minority sample penalty module** to tackle class imbalance in depression severity classification. The idea: if a sample belongs to an underrepresented class (e.g., severe depression if most subjects are mild) and the model misclassifies it, increase its weight in the loss. Conversely, frequently seen class samples or easy samples can be weighted less. We can implement a simpler version: use class weights in `CrossEntropyLoss` (PyTorch allows passing `weight=torch.tensor([w0, w1, ...])`). For binary, if depressed cases are fewer, give them a higher weight to balance. For multi-class severity (if using PHQ-9 categories), determine weights inversely proportional to class frequency. This ensures the model pays enough attention to severe cases even if they are rare.

**Sample Confidence and Curriculum Learning:** The DepL-GCN also proposed a **sample confidence module** that down-weights “hard” or possibly mislabeled samples as training progresses. They compute a metric (called eL2) per sample – essentially the ℓ2 norm of the prediction error vector – and if a sample consistently has high error (hard to learn), they eventually reduce its weight to prevent it from skewing the model. Meanwhile, the minority penalty increases weight for hard minority samples. They prioritize the minority penalty over confidence when both trigger. To implement this:

* Keep track of each sample’s loss over epochs.
* If a sample’s loss remains high while others drop, consider it a difficult sample. Possibly its label is noisy or it's an outlier.
* Introduce a schedule (e.g., after epoch 40) where we compute confidence coefficient \$w\_{\text{conf},i}\$ inversely related to sample loss, and penalty \$w\_{\text{pen},i}\$ for minority if misclassified.
* Update the loss as: \$Loss = \frac{1}{N}\sum\_i w\_{i}^{\text{all}} \cdot \text{CE}(y\_i, \hat{y}\_i)\$, where \$w\_i^{\text{all}}\$ combines confidence and penalty (the paper’s formula (8) essentially does this).

For simplicity, our blueprint doesn’t implement the full dynamic weighting, but we ensure awareness:

* We could set a rule: after 50 epochs, identify top 10% highest loss samples (if they belong mostly to majority class and might be noise) and reduce their weight by setting a lower class weight or even temporarily removing them from training for a few epochs (a form of curriculum learning where easy samples are learned first, then gradually focus on harder ones).
* Meanwhile, if severe cases are few, always keep them weighted high.

These approaches ensure the model doesn’t get stuck on impossibly hard examples and improves generalization, aligning with the **curriculum learning** principle.

**Domain Adaptation for Multi-Dataset:** The platform might eventually incorporate multiple data sources (e.g., another dataset like PRED+CT). To maintain a generalizable digital twin:

* Use a **Domain Discriminator** with adversarial training to encourage the model to learn features that are not specific to the dataset source. The DepL-GCN framework (SSPA-GCN base) included a domain adaptation component (they mention a "Domain label" and "Domain discriminator" in Fig. 3 of their paper). This likely means they trained on two datasets simultaneously (MODMA and another) and had the GCN features also try to fool a domain classifier (so the model learns common patterns across datasets).
* If we integrate such, we add a domain classification head to the GCN’s graph embedding \$\mathbf{g}\$ or ViT’s CLS embedding, and train it to predict which dataset the sample is from. We reverse its gradients (Gradient Reversal Layer technique) so that our model learns to make domain-invariant features. This addresses differences in recording conditions, making the digital twin more robust across environments.
* Implementation: if we had data from two hospitals, label domain0 vs domain1 for samples. Add `domain_fc = Linear(h, 2)` to model and a GRL in forward. Compute domain\_loss = CrossEntropy(domain\_pred, domain\_label). Then the combined loss = task\_loss + \$\gamma\$ \* domain\_loss, where \$\gamma\$ is negative if using GRL (or we minimize task\_loss and maximize domain\_loss via GRL). In code, one can use `torch.autograd.grad(..., create_graph=True)` or simpler, use `adversarial_loss = -domain_loss` and include in optimizer steps carefully. (This is advanced, so in blueprint we note it as an extension.)

**Real-Time Deployment Considerations:** For a Psychiatry Digital Twin used in practice, beyond off-line analysis:

* **Streaming Data:** The system should handle continuous EEG streams and possibly live audio. That means our preprocessing pipeline should be real-time: e.g., apply filters and segment on the fly (perhaps a sliding window to detect changes in state). The model inference should be efficient (our GCN is small so <1ms per sample; ViT is heavier but could use a smaller model or run every few seconds).
* **Latency:** If used as a monitoring tool, a few seconds delay is okay. If used for instant feedback (like during a session), ensure model inference is within that tolerance. With a GPU, our model will likely run quickly for single samples.
* **Resource Usage:** The ViT has many parameters; consider using a smaller architecture or distilling it. For instance, a CNN-based approach for spectrograms could be more lightweight if deployment on an edge device is needed. We might prepare a fallback model (like a ResNet18) if no GPU available.

These real-time concerns are part of broader “implementation requirements” for turning the notebook solution into a continuously running system.

**Explainability:** In clinical settings, interpretability is crucial:

* We already incorporate attention weights visualization to explain which channels/frequencies are contributing. Additional explainability techniques like SHAP or LIME on our models could help justify decisions (e.g., “The model focuses on decreased alpha power in right frontal region and monotonous low-pitched speech – correlating with depression biomarkers”).
* The digital twin might present these explanations to clinicians (perhaps via the visualizations we included).

**Reproducibility & Versioning:** To ensure the platform’s reliability:

* Fix random seeds and document library versions (PyTorch 1.11.0, Python 3.9 as in the paper’s environment). Our code will call `torch.manual_seed` etc.
* Use deterministic operations where possible (PyTorch allows `torch.use_deterministic_algorithms(True)` to avoid nondeterministic GPU ops).
* Save trained model weights for each fold or the final model. This allows re-loading without retraining, useful for a persistent digital twin that gets updated periodically rather than every run.
* Possibly containerize the environment (Dockerfile specifying dependencies) for deployment.

**Scalability:** If more data becomes available or new modalities:

* The design is modular; we can add another modality branch (e.g., facial expression video – a ViT or CNN on that).
* If dataset grows, we might switch from LOOCV to a dedicated train/val/test split for efficiency, using the val to tune hyperparameters.
* The code should be vectorized to handle batch operations for speed (we took care to use matrix ops in GCN, though our example code was per-sample for clarity, one would batch multiple graphs by stacking adjacency in a block diagonal or use PyG’s batching of sparse graphs).

**Personalization:** A digital twin might adapt to individual baseline. One could fine-tune the model on a specific patient’s prior recordings to better detect their changes (e.g., use transfer learning to adapt global model to individual). This is beyond current scope but an interesting extension: maintain a small update to the model for a specific patient as more of their data is collected, ensuring the twin reflects personal norms.

**Ethical and Privacy Considerations:** We ensure that:

* Data is stored and handled securely (in notebooks we don’t show that aspect, but mention that all patient data should be de-identified, and computations could be done on encrypted or local servers).
* The model’s decisions are not biased by demographic factors (the dataset is relatively small but balanced in gender in this case; a broader requirement is to test fairness across age, gender if data allows).
* We always include the caveat that this tool aids, not replaces, clinical diagnosis – especially given the moderate sample size.

In summary, implementing these broader requirements means our notebook doesn’t just train a model, but sets a foundation for a robust, generalizable, and clinically useful system:

* Weighted loss or dynamic curriculum to handle imbalanced or noisy data.
* Domain adversarial training for multi-source integration.
* Real-time data pipeline for streaming input.
* Visualization and explainability for user trust.
* Reproducibility through careful seeding and documentation (we match the environment used in reference: Python 3.9, PyTorch 1.11, etc.).

The notebook will include notes or optional code for these features (for instance, a commented section showing how to adjust sample weights over epochs or how to add a domain classifier, so an AI engineer can enable those if needed).

By addressing these aspects, the Psychiatry Digital Twin moves closer to a deployable solution that is **accurate, interpretable, and adaptable** to real-world clinical settings.

## 8. Comprehensive Hyperparameter List & Reproducibility

Finally, we compile all hyperparameters and settings used, along with notes to reproduce results consistently.

**Data & Preprocessing Hyperparameters:**

* `sampling_rate_eeg = 250 Hz` – EEG sampling frequency.
* `eeg_channels = 29` – Channels selected for analysis (list provided).
* `filter_bandpass = [1 Hz – 40 Hz]` – EEG band-pass filter cutoffs.
* `ica_components = 29` – Number of ICA components (equal to channel count) for artifact removal.
* `segment_length = 2 s` (500 samples at 250 Hz) – EEG segment duration for feature extraction.
* `bands = {delta:(1,4), theta:(4,8), alpha:(8,13), beta:(13,30), gamma:(30,40)}` (Hz) – frequency bands for DE features (note: gamma limited to 40 Hz due to filtering).
* `wavelet_name = 'db4'` – Wavelet type for DWT features (Daubechies 4 is commonly used in EEG analysis).
* `wavelet_level = 4` – Levels of decomposition (gives \~5 sub-bands).
* `mel_n_filters = 64` – Number of Mel filter banks for spectrogram (we choose 64 as a mid value; not explicitly given in paper, but 64 is sufficient to capture speech formants).
* `frame_length_audio = 25 ms` – STFT frame size for speech (typically 400 samples at 16kHz).
* `frame_hop_audio = 10 ms` – STFT hop (160 samples at 16kHz).
* `pre_emphasis_coeff = 0.97` – Speech pre-emphasis filter coefficient.
* `window_function = Hamming` – for STFT on audio (and if used for EEG spectrogram).
* `spectrogram_resize = (224,224)` – target image size for ViT input.
* `image_norm_mean = (0.485,0.456,0.406)`, `image_norm_std = (0.229,0.224,0.225)` – normalization constants (ImageNet) applied to spectrogram images before ViT.

**Model Hyperparameters:**

* `gcn_in_features = 5` – Dimension of node feature vector (DE5 or DWT5).
* `gcn_hidden_dim = 64` – Dimension of GCN hidden layers (arbitrary choice, but 64 worked in practice).
* `num_gcn_layers = 2` – Number of GCN layers before attention.
* `gcn_activation = ReLU` – Activation after each GCN layer.
* `att_num_heads = 4` – Number of attention heads in MHA module.
* `att_head_dim = 16` – Dimension per head (such that 4\*16 = 64 = hidden\_dim).
* `att_dropout = 0` – (Not explicitly mentioned, we assume none, but one could add dropout in attention).
* `graph_pooling = average` – Method to get graph-level embedding from node features (we use average).
* `num_classes = 2` (for binary depressed/healthy). If doing severity classification, `num_classes = 4` (or depending on how levels are grouped by PHQ-9).
* `ViT_model_name = 'vit_base_patch16_224'` – ViT architecture.

  * `patch_size = 16` – (implied by model name).
  * `embed_dim = 768` – hidden size in ViT.
  * `num_transformer_layers = 12`.
  * `num_attention_heads_vit = 12`.
  * `mlp_dim = 3072` – feedforward network size.
  * `vit_pretrained = True` – using pretrained weights on ImageNet.
  * `vit_trainable_layers = all` – we fine-tune whole model (alternatively could freeze some).
* `fusion_weights = (w_eeg, w_speech)` – weights for decision fusion. If not using a validation to set, default to (0.5, 0.5) or based on performance. For instance, if after training we find EEG branch accuracy = 85%, speech = 80%, we might set \$w\_{eeg}=0.85, w\_{speech}=0.80\$ (normalized to sum=1 as 0.515, 0.485) as a heuristic.
* In DepL-GCN’s multi-class scenario, they effectively gave higher weight to EEG (AMGCN-L baseline \~73%) vs others (EEGNet \~60%), but since we are doing ensemble differently, we’ll keep it equal or slight bias if evidence.

**Training Hyperparameters:**

* `optimizer = Adam` – for both GCN and ViT.
* `learning_rate_gcn = 1e-3` – initial LR for GCN branch.
* `learning_rate_vit = 1e-4` – LR for ViT branch (lower to avoid catastrophic forgetting of pre-trained features).
* `weight_decay = 1e-5` – L2 regularization (if any, can use small value to prevent overfit).
* `batch_size_eeg = 32` – batch size for EEG segments (training).
* `batch_size_speech = 8` – batch size for spectrograms (we use smaller to fit in GPU memory).
* `epochs = 100` – training epochs per fold (this is enough for convergence as used in literature).
* `scheduler = None` – (optionally, could use a learning rate scheduler, e.g., reduce LR on plateau or cosine decay).
* `early_stopping = False` – (not used in LOOCV context, as we train fixed epochs).
* `seed = 42` – random seed for reproducibility (we set `np.random.seed(42)`, `torch.manual_seed(42)` at the top).
* `torch_deterministic = True` – set torch backend to deterministic (e.g., `torch.use_deterministic_algorithms(True)` to avoid nondeterministic ops).
* Hardware: ensure runs on GPU if available (`device = 'cuda' if torch.cuda.is_available() else 'cpu'`). The reference environment had an NVIDIA GTX 3060 GPU; our implementation should similarly utilize GPU for speed.

**Cross-Validation Settings:**

* `cv_method = 'LOOCV'` – leave-one-out cross-validation (53 folds).
* Or `cv_method = 'KFold'` with `K=5` – use 5-fold cross-validation (each fold \~10 subjects). If using KFold, ensure stratification (each fold has proportional depressed vs control).
* `cv_shuffle = False` (if LOOCV, no shuffle needed as each element goes once; for KFold, we can shuffle before splitting to ensure random fold composition).
* In LOOCV, since test is one subject, no separate validation set. In 5-fold, we could set aside, e.g., within each training fold, 10% of subjects as val to tune weights or early stop (not explicitly done in paper, but one can).
* `aggregate_segments = True` – indicates that we aggregate multiple segment predictions to one subject prediction in evaluation (majority vote or avg probability).

**Thresholds and Others:**

* Decision threshold = 0.5 for binary (we use argmax, which is equivalent to threshold 0.5 on binary probability).
* For multi-class, no threshold (argmax).
* `corr_threshold_graph = None` – if we wanted to prune weak edges for visualization, we might set e.g. 0.3 to only show edges >0.3 in graph plot (just for visualization, not in model).
* **Optional DepL hyperparams:** (if implementing sample weighting)

  * `conf_start_epoch = 45` – epoch to start applying sample confidence adjustments (found optimal \~40–50 for MODMA).
  * `u_rate = 0.55` – update rate for eL2 threshold (DepL-GCN found 0.5–0.6 best for MODMA).
  * These would affect how quickly we lower weights for confident samples.
  * Weights \$w\_{conf}, w\_{pen} \in {0,1}\$ as per algorithm toggling (we either apply the module to a sample in an epoch or not).
* If domain adaptation:

  * `lambda_domain = 0.1` – weight for domain adversarial loss in total loss (tune such that domain accuracy hovers around chance).
  * Possibly `domain_grad_reverse = True` – flag to indicate using gradient reversal in implementation.

We will clearly set all these hyperparameters at the top of the notebook in a single block (makes it easy to adjust by user). For example:

```python
# Hyperparameters
SFREQ = 250
CHANNELS = ['F7','F3',...,'POz']  # 29 channels
EEG_FILTER_LOW, EEG_FILTER_HIGH = 1.0, 40.0
SEGMENT_SEC = 2
BANDS = [(1,4),(4,8),(8,13),(13,30),(30,40)]
WAVELET = 'db4'; WAVELET_LEVEL = 4
PRE_EMPHASIS = 0.97
FRAME_LEN = 400; HOP_LEN = 160  # for 16kHz audio (~25ms, ~10ms)
N_MELS = 64
IMG_SIZE = 224
GCN_HIDDEN = 64; GCN_LAYERS = 2
ATT_HEADS = 4
NUM_CLASSES = 2
LR_GCN = 1e-3; LR_VIT = 1e-4
BATCH_EEG = 32; BATCH_SPEECH = 8
EPOCHS = 100
SEED = 42
CV_FOLDS = None  # None for LOOCV, or 5 for 5-fold
```

(This is an example of what we'd define.)

**Reproducibility Checks:**

* We will run a sanity check: e.g., train on a small subset or a single fold and verify outputs are as expected (the blueprint expects the AI agent to implement and presumably test).
* We note that due to stochastic processes (random weight init, randomness in Adam), results can vary slightly. But setting the seed and using deterministic algorithms should ensure nearly identical results on each run.
* Logging: the notebook should print or store the random seed and possibly the software versions (`print(torch.__version__, np.__version__)`). E.g., "Using PyTorch 1.11.0, numpy 1.24.0" to document environment.

**Comparing to Reference Results:** After running cross-val, we expect performance roughly similar to literature:

* Our target: \~89% accuracy, \~89% F1 for binary classification on MODMA. If our features and training are correct, we should achieve in that ballpark. Minor differences might occur if, say, we used DE vs exactly how they did DWT.
* If severity classification (4 classes), DepL-GCN achieved 81% accuracy on MODMA 4-class. That is a high bar; we can mention that multi-class is harder. Our blueprint focuses on binary for simplicity unless extended.
* If results deviate, one can tune hyperparameters (e.g., increase GCN hidden dim or train longer). The blueprint provides the scaffolding to do so.

**Hyperparameter Sensitivity:** We will mention a few likely important ones:

* Number of attention heads (we used 4 as in paper for EEG GCN; changing it could affect performance).
* Choice of features (DE vs wavelet) – one could experiment and see slight differences in model accuracy.
* Learning rates – too high for ViT could ruin pre-trained features; too low for GCN might slow training.
* Batch size – larger batch gives more stable gradient (if memory allows, could increase for EEG to 128).
* Segment length – 2s was chosen, if we increase to 5s, we have fewer samples, possibly lower training data but more stable features. That could degrade generalization due to less data for training. DepL chose 2s likely as a sweet spot.
* Preprocessing – e.g., if we didn’t do ICA artifact removal, results might be slightly worse due to noise. We include it to match their process.

**Ensuring Reproducibility Steps Recap:**

1. Set seed for all libraries.
2. Document environment and package versions.
3. Use fixed CV splits (we can even hardcode the fold composition for 5-fold if needed).
4. Log hyperparameters and final metrics clearly, so results can be compared when changes are made.
5. Provide the random initialization details or save the random initial weights for consistency if needed.

By meticulously defining these parameters and steps, another AI engineer or the generative agent can run the notebook and expect consistent behavior and the ability to tweak things in a controlled manner.

**Conclusion:** This technical blueprint has thus specified an end-to-end solution: from data handling to model details, training logic, visualization, and extended considerations. All parameters and design choices were drawn from the provided references (e.g., MODMA dataset descriptions, MHA-GCN ViT methodology, DepL-GCN innovations) or typical best practices, ensuring the notebook can be implemented without ambiguity and yield a Psychiatry Digital Twin model that is both high-performing and transparent. The major hyperparameters are summarized in the list above, which serves as a reference table in the notebook for quick tuning and verification. Each critical claim or setting has been cited to underline the reasoning, making the blueprint not just a set of instructions, but a well-justified plan grounded in current research findings.&#x20;
