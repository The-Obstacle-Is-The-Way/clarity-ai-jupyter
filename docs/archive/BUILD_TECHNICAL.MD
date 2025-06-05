## **Psychiatry Digital Twin - Jupyter Notebook Technical Blueprint**

This blueprint provides the detailed specifications for creating the Jupyter Notebook, referencing the provided documents: `BUILD_OVERVIEW.MD` (BO), `BUILD_TECHNICAL.MD` (BT), `INITIAL_PROMPT.MD` (IP), "A Multi-modal Open Dataset for Mental-disorder Analysis" (MODMA Paper), "Cross-Subject Depression Level Classification Using EEG Signals with a Sample Confidence Method" (DepL-GCN Paper), and "Multimodal depression detection based on an attention graph convolution and transformer" (MHA-GCN ViT Paper).

---

### **1. Jupyter Notebook Structure and Flow** 📝

Here's a logical cell-by-cell outline:

1.  **Cell 1: Title and Introduction (Markdown)**
    * Title: Psychiatry Digital Twin - EEG Analysis for Depression Level Classification
    * Brief introduction to the project, objectives, and datasets used (MODMA).
    * Mention of models to be implemented (DepL-GCN, MHA-GCN\_ViT, Baseline CNN).

2.  **Cell 2: Setup - Imports (Code)**
    * Import all necessary libraries: `os`, `numpy`, `scipy`, `pandas`, `mne`, `pywt`, `torch`, `torch.nn`, `torch.optim`, `sklearn.model_selection`, `sklearn.metrics`, `plotly.graph_objects`, `ipywidgets`, `matplotlib.pyplot`, `tqdm`.
    * `from mpl_toolkits.mplot3d import Axes3D` (BO, Section 8).
    * `from torch_geometric.nn import GCNConv` (or custom GCN layer if preferred).

3.  **Cell 3: Setup - Reproducibility and Configuration (Code)**
    * `numpy.random.seed(42)`
    * `torch.manual_seed(42)`
    * `torch.cuda.manual_seed_all(42)` (if GPU is used)
    * `DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")`
    * Define constants from `BUILD_OVERVIEW.MD`: `CHANNELS_29`, `SAMPLING_RATE`, `WINDOW_SIZE`, `OVERLAP`, `FREQ_BANDS`, `DEPRESSION_LEVELS`.
    * `DATA_DIR = 'data/MODMA_EEG/'` (adjust as per local setup from BT, Section 4.4).
    * `NUM_SUBJECTS = 53` (MODMA Paper, p. 1). [cite: 5]

4.  **Cell 4: Data Loading and Preprocessing Utilities (Code)**
    * Helper functions for:
        * Loading raw EEG data for a subject (BT, Section 5.2).
        * Selecting specified 29 channels (BO, Section 1).
        * Bandpass filtering (BO, Section 4.2; MHA-GCN ViT Paper, Section 4.2).
        * ICA artifact removal (BO, Section 4.2).
        * Segmentation into 2s windows with 50% overlap (BO, Section 4.2).
        * Loading speech data (if MHA-GCN\_ViT is fully implemented, details TBD based on MODMA audio structure).

5.  **Cell 5: Feature Extraction Utilities (Code)**
    * Functions for:
        * `compute_differential_entropy(psd_data)` (BO, Section 2.A).
        * `calculate_psd_welch(epoch_data, sfreq, freq_bands)` (parameters for Welch's method).
        * `extract_dwt_features(epoch_data)`: 'db4' wavelet, 5 levels; mean, std, energy. Reshape/select DWT coefficients to output [15 features, 180 time steps] per channel for a 2s window, as per MHA-GCN ViT Paper (Table 2 footnote).
        * `extract_stft_spectrogram_eeg(epoch_data, target_size=(224,224))`: For ViT input.
        * `extract_stft_spectrogram_speech(audio_segment, target_size=(224,224))`: For ViT input (MHA-GCN ViT Paper, Section 3.2.3). [cite: 232]
        * `compute_adjacency_matrix(channel_features_or_raw_epochs)`: Pearson correlation, threshold 0.3 (BO, Section 2.C).

6.  **Cell 6: Model Definitions - MHA-GCN (Code)**
    * `class MHA_GCN(torch.nn.Module)` as per BO Section 3.A and MHA-GCN ViT Paper Section 3.3. [cite: 297]

7.  **Cell 7: Model Definitions - DepL-GCN Components (Code)**
    * `calculate_NeL2(LeL2, eL2, u_rate=0.6)` (BO, Section 3.B; DepL-GCN Paper, Eq. 5).
    * `minority_penalty(prediction, true_label, is_minority, NeL2_val, max_NeL2_val)` (BO, Section 3.B; DepL-GCN Paper, Eq. 6).
    * Custom Loss Function incorporating these penalties (DepL-GCN Paper, Eq. 7, 8).

8.  **Cell 8: Model Definitions - Vision Transformer (ViT) (Code)**
    * Instantiate ViT (e.g., from `timm` library) or define custom class based on `vit_config` (BO, Section 3.C).

9.  **Cell 9: Model Definitions - Combined MHA-GCN\_ViT (Code)**
    * Class combining MHA-GCN for EEG DWT, ViT for EEG Spectrograms, ViT for Speech Spectrograms.
    * Decision-level fusion logic (MHA-GCN ViT Paper, Section 3.5).

10. **Cell 10: Model Definitions - Baseline CNN (Code)**
    * Define a simple CNN for baseline comparison (e.g., 2-3 conv layers, pooling, dense layers). Input could be raw EEG windows or simple features like PSD.

11. **Cell 11: Training and Evaluation Utilities (Code)**
    * `train_epoch` function.
    * `evaluate_model` function.
    * Function to calculate and store metrics (Accuracy, Precision, Recall, F1, Confusion Matrix).

12. **Cell 12: Main Experiment Loop - Leave-One-Out Cross-Validation (LOOCV) (Code)**
    * Outer loop for LOOCV (iterating through subjects).
    * Inside:
        * Data preparation for current fold (train/test split by subject).
        * DataLoaders for PyTorch.
        * Model initialization.
        * Optimizer and Loss function instantiation.
        * Training loop (epochs).
            * Apply DepL-GCN's sample confidence logic here if DepL-GCN is being trained.
        * Evaluation on the test subject.
        * Store results for each fold.

13. **Cell 13: Results Aggregation and Display (Code & Markdown)**
    * Calculate and print average metrics from LOOCV.
    * Display average confusion matrix.
    * Markdown cells to explain the results.

14. **Cell 14: Visualization Components - Setup (Code)**
    * `mne.channels.make_standard_montage('standard_1020')` for the 29 channels.
    * Functions to prepare data for Plotly.

15. **Cell 15: Interactive Visualization - Channel Selector & Topomaps (Code & Markdown)**
    * `widgets.Dropdown` for channel selection.
    * `widgets.Output` for displaying plots.
    * `mne.viz.plot_topomap` for selected frequency band's DE or PSD.
    * Function to update plots based on dropdown.
    * Markdown to explain interaction.

16. **Cell 16: Interactive Visualization - Real-time Metrics & Model Comparison (Code & Markdown)**
    * `widgets.Output()` for "real-time" metrics (e.g., for a sample subject/window).
    * `go.Figure(data=go.Heatmap())` for model comparison matrix (BO, Section 5.B).
    * Markdown to explain.

17. **Cell 17: Simplified 3D Brain Visualization (Optional) (Code & Markdown)**
    * Matplotlib 3D scatter plot for electrode positions and activation (BO, Section 8).

18. **Cell 18: Clinical Report Snippet Generation (Code & Markdown)**
    * Function to generate a formatted markdown string for a sample subject.
    * Display the report.
    * Markdown for explanation.

19. **Cell 19: Conclusion and Future Work (Markdown)**
    * Summarize findings.
    * Suggest potential next steps.

---

### **2. Data Handling - End-to-End** 🔄

* **Raw EEG Data Loading (MODMA)**:
    * **Loading a single subject**:
        ```python
        import mne
        # Assuming subject_id is like 'subj01', 'subj02', ...
        # Path needs to be verified against actual MODMA dataset structure after download.
        # The MODMA paper mentions 53 participants for 128-channel EEG. [cite: 5]
        # BUILD_TECHNICAL.MD gives an example: raw = mne.io.read_raw_fif("../data/128EEG/subj01_raw.fif", preload=True)
        file_path = f"{DATA_DIR}/128EEG/{subject_id}_raw.fif" # Adjust if naming differs
        if not os.path.exists(file_path):
            # Fallback if .fif isn't directly available, may need conversion from .set or other formats
            # e.g., file_path = f"{DATA_DIR}/EEG_128channel_resting/{subject_id}/rest.set"
            # raw = mne.io.read_raw_eeglab(file_path, preload=True)
            print(f"Warning: {file_path} not found. Placeholder logic for loading.")
            # This part needs to be robust to the actual file structure and format in MODMA
            # For now, assuming .fif files are present as per BT example for simplicity.
            return None 
        raw = mne.io.read_raw_fif(file_path, preload=True)
        ```
    * **Channel Selection**:
        ```python
        CHANNELS_29 = ['F7', 'F3', 'Fz', 'F4', 'F8', 'T7', 'C3', 'Cz', 'C4', 
                        'T8', 'P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'O2', 'Fpz', 
                        'F9', 'FT9', 'FT10', 'TP9', 'TP10', 'PO9', 'PO10', 'Iz', 
                        'A1', 'A2', 'POz'] # BO, Section 1
        # Verify CHANNELS_29 are present in raw.info['ch_names'] and handle potential discrepancies.
        # Some datasets might use 'FPz' instead of 'Fpz', or 'I' instead of 'Iz'.
        # A1, A2 are often reference channels. If they are not data channels, they might need to be handled differently (e.g., set as reference if not already, or dropped if not used).
        # For MODMA, it's important to check how A1/A2 are used. Assuming they are standard data channels for now as listed.
        
        # Create a mapping for potential differences if necessary, e.g.:
        channel_mapping = {'Fpz': 'FPz', 'Iz': 'I1'} # Example based on common variations
        mapped_channels_29 = []
        missing_channels = []
        for ch in CHANNELS_29:
            if ch in raw.ch_names:
                mapped_channels_29.append(ch)
            elif ch in channel_mapping and channel_mapping[ch] in raw.ch_names:
                mapped_channels_29.append(channel_mapping[ch])
            else:
                # Attempt to find A1/A2 if they are Mastoids (M1/M2)
                if ch == 'A1' and 'M1' in raw.ch_names: mapped_channels_29.append('M1')
                elif ch == 'A2' and 'M2' in raw.ch_names: mapped_channels_29.append('M2')
                else: missing_channels.append(ch)
        
        if missing_channels:
            print(f"Warning: The following specified channels were not found and could not be mapped: {missing_channels}")
            print(f"Available channels: {raw.ch_names}")
            # Decide on a strategy: error out, or proceed with available channels.
            # For now, proceed with successfully mapped/found channels.

        raw.pick_channels(mapped_channels_29, ordered=False) # Use only successfully found/mapped channels
        ```
    * **Iterating through subjects**:
        ```python
        # Assuming subject IDs are subj01, subj02, ..., subj53
        # This list would ideally be sourced from the MODMA dataset's participant list.
        # The MODMA paper states 53 participants for the 128-electrode EEG. [cite: 5]
        subject_ids = [f"subj{i:02d}" for i in range(1, 54)] 
        all_subject_data = []
        for subj_id in subject_ids:
            # raw_data = load_and_select_channels_for_subject(subj_id) # Using a combined function
            # if raw_data:
            #     all_subject_data.append(raw_data)
            pass # Placeholder for actual loop logic
        ```

* **Speech Data Processing (for MHA-GCN\_ViT)**:
    * The MODMA dataset includes spoken language data. [cite: 3, 7]
    * The MHA-GCN ViT paper uses STFT to extract Mel spectrograms from speech. [cite: 232, 249]
    * **STFT Parameters for Speech Mel Spectrograms** (Typical values, paper may have specifics not detailed here):
        * `SAMPLING_RATE_SPEECH`: Typically 16000 Hz or higher for speech. Need to verify MODMA speech data's sampling rate.
        * `FRAME_SIZE`: e.g., 25ms. `frame_length = int(SAMPLING_RATE_SPEECH * 0.025)`
        * `FRAME_STRIDE` (overlap): e.g., 10ms. `frame_step = int(SAMPLING_RATE_SPEECH * 0.010)`
        * `WINDOW_TYPE`: 'hann'
        * `NFFT`: Next power of 2 from `frame_length`, e.g., 512 or 1024.
        * `NUM_MEL_BINS`: e.g., 80 or 128.
        * Use `librosa.feature.melspectrogram` or `torchaudio.transforms.MelSpectrogram`.
    * **Preparation for ViT**:
        * Resize spectrogram to 224x224 pixels (MHA-GCN ViT Paper, Section 4.2). [cite: 232] This might involve interpolation.
        * Normalization: Typically to \[0, 1] or mean 0, std 1.
    * **Synchronization/Pairing**:
        * MODMA contains EEG and audio from the same participants. [cite: 3] The dataset structure documentation will be crucial to link a specific EEG file/segment to its corresponding audio file/segment for each subject. Often, this is done by participant ID and session/task identifiers. Assume files are named or structured to allow this pairing.

* **Preprocessing Steps** (Applied per subject's 29-channel raw data):
    * **Bandpass filter**:
        ```python
        # BO, Section 4.2: 1-40 Hz. MHA-GCN ViT Paper, Section 4.2: 0.5-45 Hz.
        # Let's use 1-40 Hz from BUILD_OVERVIEW.
        # MNE default is FIR filter. Butterworth can be specified if needed.
        low_cut = 1.0  # Hz
        high_cut = 40.0 # Hz
        raw.filter(l_freq=low_cut, h_freq=high_cut, fir_design='firwin', verbose=False)
        ```
    * **ICA for artifact removal**:
        ```python
        # BO, Section 4.2
        # MHA-GCN ViT Paper, Section 4.2 mentions ICA but no specific parameters.
        # Suggestion:
        n_components_ica = 20 # Or len(raw.ch_names) - 1, or a fixed number like 20-25.
                              # Needs to be less than or equal to number of channels.
        ica = mne.preprocessing.ICA(n_components=n_components_ica, method='fastica', random_state=42, max_iter='auto')
        ica.fit(raw)
        
        # Identifying artifactual ICs (example: EOG and ECG):
        # This usually requires EOG/ECG channels or heuristics.
        # If dedicated EOG/ECG channels are available:
        # ica.exclude = []
        # eog_indices, eog_scores = ica.find_bads_eog(raw, ch_name=['EOG_H', 'EOG_V']) # Example ch_names
        # ica.exclude.extend(eog_indices)
        # ecg_indices, ecg_scores = ica.find_bads_ecg(raw, ch_name='ECG') # Example ch_name
        # ica.exclude.extend(ecg_indices)
        
        # If no EOG/ECG channels, use heuristics (more complex, e.g., based on IC properties like spatial distribution or time course)
        # For simplicity, if no EOG/ECG, one might skip automatic detection or manually inspect ICs for a few subjects to set heuristics.
        # Assuming a basic approach for now. If specific EOG/ECG channels were part of the 128 channels, they should be identified.
        # Often, a common approach is to find ICs correlated with EOG/ECG channels if present.
        # If not, visual inspection or automated methods based on spectral properties or spatial maps are used.
        # As a placeholder:
        # Find ICs correlating with Fpz (often captures eye blinks if no dedicated EOG)
        # Fpz_idx = raw.ch_names.index('Fpz') if 'Fpz' in raw.ch_names else (raw.ch_names.index('FPz') if 'FPz' in raw.ch_names else -1)
        # if Fpz_idx != -1:
        #    eog_indices, eog_scores = ica.find_bads_eog(raw, ch_name=raw.ch_names[Fpz_idx])
        #    if eog_indices: ica.exclude = eog_indices
        # else:
        #    print("Warning: Fpz channel not found for EOG artifact detection.")
        
        # For now, let's assume a number of components are removed (e.g., first 1-2 if they capture gross artifacts)
        # This step ideally requires more domain knowledge or visual inspection setup.
        # For a truly automatic pipeline without dedicated EOG/ECG channels, this is a challenging step.
        # A common placeholder if no other info: remove components with typical artifact signatures (e.g. very low frequency drift, or strong frontal activity for blinks)
        # For this blueprint, let's assume a manual or semi-automatic IC selection process is abstracted.
        # If ica.exclude is not set, no components will be removed by default.
        # Alternative: Use an automated method like 'autoreject' or MNE's automatic EOG/ECG detection if channels available.

        ica.apply(raw)
        ```
    * **Segmentation**:
        ```python
        # BO, Section 4.2: 2-second windows with 50% overlap
        # SAMPLING_RATE = 250 Hz (BO, Section 1)
        window_duration = 2.0  # seconds
        window_overlap_ratio = 0.5 # BO, Section 1
        
        # MHA-GCN ViT Paper (Section 4.2) also uses 2-s epochs with 50% overlap.
        epochs = mne.make_fixed_length_epochs(raw, duration=window_duration, overlap=window_duration * window_overlap_ratio, preload=True, verbose=False)
        # epochs.get_data() will return a 3D numpy array: (n_epochs, n_channels, n_times)
        ```

---

### **3. Detailed Feature Extraction Implementation** 🔬

* **Differential Entropy (DE)**:
    * Formula: `DE = 0.5 * np.log(2 * np.pi * np.e * psd_data)` (BO, Section 2.A).
    * `psd_data` calculation per 2s window, per channel, per band:
        ```python
        # FREQ_BANDS from BO, Section 1
        # SAMPLING_RATE = 250 Hz
        def calculate_psd_de_features(epoch_data, sfreq, freq_bands, window_size_sec=2):
            # epoch_data: (n_channels, n_times_in_window)
            # e.g., n_times_in_window = sfreq * window_size_sec = 250 * 2 = 500
            
            n_channels, n_times = epoch_data.shape
            de_features_bands = np.zeros((n_channels, len(freq_bands)))
            psd_features_bands = np.zeros((n_channels, len(freq_bands)))

            for band_idx, (band_name, (low_freq, high_freq)) in enumerate(freq_bands.items()):
                # Welch's method for PSD
                # nperseg: length of each segment for Welch. Can be n_times for shorter windows, or a fraction.
                # For 2s window (500 samples), nperseg could be 250 or 500.
                # If using MNE:
                # psds, freqs = mne.time_frequency.psd_welch(epoch_data_mne_format, fmin=low_freq, fmax=high_freq, ...)
                # Using scipy.signal.welch:
                freqs, psd = scipy.signal.welch(epoch_data, fs=sfreq, nperseg=n_times, axis=-1) 
                                                # Using n_times means one segment per epoch for Welch's method, essentially periodogram.
                                                # Or use a smaller nperseg like 256 if desired.
                
                # Find PSD within the current band
                band_mask = (freqs >= low_freq) & (freqs <= high_freq)
                psd_in_band = psd[:, band_mask]
                
                # Calculate power in band (integral of PSD, approximated by sum)
                power_in_band = np.sum(psd_in_band, axis=-1) * (freqs[1] - freqs[0] if len(freqs) > 1 else 1) # (n_channels,)
                psd_features_bands[:, band_idx] = power_in_band
                
                # DE calculation: Add a small epsilon to prevent log(0) if power_in_band can be zero
                epsilon = 1e-10 
                de_features_bands[:, band_idx] = 0.5 * np.log(2 * np.pi * np.e * (power_in_band + epsilon))
                
            # Output: de_features_bands (n_channels, n_freq_bands)
            # Output: psd_features_bands (n_channels, n_freq_bands) for other uses if needed
            return de_features_bands 
        ```

* **Discrete Wavelet Transform (DWT)**:
    * Wavelet: 'db4', 5 decomposition levels using `pywt.wavedec` (BO, Section 2.B).
    * Features: mean, std, energy for each of the 5 levels (approx + detail coeffs) per channel. This results in (5 levels + 1 approx level) * 3 features = 18 features if we consider all. Or 5 levels of detail coefficients = 5*3 = 15 features. The BO says "5 decomposition levels... mean, std, energy for each level", suggesting 15 features.
    * **Critical - Output Shape for MHA-GCN**: The target output per 2s window is `[29_channels, 15_dwt_features, 180_time_steps]` as per MHA-GCN ViT Paper (Table 2, "EEG DWT features" footnote: "The DWT features extracted from each 2-s EEG epoch are represented as a tensor of 29 channels × 15 features (5 levels × 3 metrics: mean, std, energy) × 180 time steps. The '180 time steps' dimension is achieved by reshaping or selecting coefficients from the DWT output to form a fixed-length sequence for each feature type and channel from that single 2s epoch.").
        ```python
        import pywt
        import numpy as np
        from scipy.stats import entropy as scipy_entropy # Renamed to avoid conflict

        def extract_dwt_features_for_mha_gcn(epoch_data_single_channel, target_time_steps=180):
            # epoch_data_single_channel: (n_times_in_window,) e.g., (500,) for 2s @ 250Hz
            
            coeffs = pywt.wavedec(epoch_data_single_channel, 'db4', level=5)
            # coeffs[0] is cA5, coeffs[1] is cD5, ..., coeffs[5] is cD1
            
            dwt_features_time_series = [] # Will store 15 feature series, each of length target_time_steps

            # Extract features from detail coefficients (cD5 to cD1)
            for i in range(1, 6): # Levels 5 down to 1 (detail coeffs)
                detail_coeffs = coeffs[i]
                
                # To get 180 time steps for each feature (mean, std, energy of coeffs)
                # This is the tricky part. The paper implies each of the 15 features has 180 time steps.
                # Option 1: If coefficients are long enough, segment them.
                # Option 2: Interpolate/Resample coefficients to 180 points then calculate features.
                # Option 3: The "180 time steps" means something else - perhaps applying DWT over 180 small sub-windows.
                # However, the footnote says "reshaping or selecting coefficients from the DWT output".
                # Let's assume we need to make each coefficient type (mean, std, energy) a sequence of 180.
                # This means we process the coefficients of each DWT level to yield these 3 features, each as a sequence of length 180.
                # This is highly unusual. A more standard interpretation would be 15 scalar features per window.
                # Given the MHA-GCN ViT paper footnote, let's attempt to generate sequence-like features.
                # This likely means that for each of the 5 levels, we create a "time series" representation of its mean, std, and energy.
                # This might be done by taking sub-windows of the coefficients, or by a transformation.
                # If the goal is a feature vector of size 15*180 per node for GCN, one direct way is:
                # Calculate 15 scalar DWT features (mean, std, energy for 5 detail levels).
                # Then, these 15 scalars are perhaps used to modulate some underlying template of 180 time steps, or repeated.
                # This part is ambiguous in standard signal processing terms.

                # Simpler interpretation that leads to 15 scalar features:
                # c_mean = np.mean(detail_coeffs)
                # c_std = np.std(detail_coeffs)
                # c_energy = np.sum(np.square(detail_coeffs))
                # dwt_scalar_features.extend([c_mean, c_std, c_energy])
                # This would yield 15 scalar features. To get to 15x180, one might repeat these scalars 180 times.
                # Or, the "180 time steps" refers to taking 180 consecutive 2s windows and stacking their 15 DWT features.

                # Let's follow the paper's phrasing closely: "reshaping or selecting coefficients...to form a fixed-length sequence"
                # For each level's coefficients (e.g., cD5), we need to derive 3 sequences of length 180.
                # This is difficult to interpret directly. Let's assume a more standard approach and note this as needing EXPERT CLARIFICATION.
                # Standard DWT features are scalar per window.
                # If 15*180 is truly the feature vector size per node from a single 2s window:
                # One possibility: the coefficients themselves are padded/truncated to 180 points, and then mean, std, energy are taken *over these 180 points*,
                # which would just be 3 scalars. This doesn't fit.

                # Alternative interpretation from MHA-GCN ViT paper table 2: "29 channels × 15 features × 180 time steps".
                # This means for each of the 15 feature types (e.g., mean of cD5, std of cD5, energy of cD5, mean of cD4...),
                # we have a sequence of 180 values.
                # How is a single mean/std/energy value turned into a sequence of 180?
                # Perhaps the DWT is applied to 180 sub-segments of the 2s window? Unlikely.
                # Or, features from 180 *different* 2s windows are stacked. This is more plausible for forming an input to a model that expects sequences.
                # If this is the case, then one "sample" for the MHA-GCN path consists of 180 consecutive (overlapping) 2s windows.
                # For each window, we extract 15 DWT features.
                # So, `dwt_features_for_one_sample` would be `(n_channels, 15_features, 180_windows_as_timesteps)`.
                # This interpretation makes more sense for the shape `(29, 15, 180)`.

                # Let's proceed with the "180 windows as timesteps" interpretation:
                # The function should take a sequence of 180 windows for a single channel.
                # def extract_dwt_features_for_mha_gcn_mult_windows(epoch_data_single_channel_many_windows):
                #     # epoch_data_single_channel_many_windows: (180_windows, n_times_in_one_window)
                #     all_window_features = []
                #     for k in range(epoch_data_single_channel_many_windows.shape[0]): # Iterate over 180 windows
                #         single_window_data = epoch_data_single_channel_many_windows[k, :]
                #         coeffs = pywt.wavedec(single_window_data, 'db4', level=5)
                #         current_window_dwt_features = []
                #         for i in range(1, 6): # cD5 to cD1
                #             detail_coeffs = coeffs[i]
                #             current_window_dwt_features.append(np.mean(detail_coeffs))
                #             current_window_dwt_features.append(np.std(detail_coeffs))
                #             current_window_dwt_features.append(np.sum(np.square(detail_coeffs)))
                #         all_window_features.append(current_window_dwt_features) # list of 180 lists, each of 15 features
                #     return np.array(all_window_features).T # Shape (15_features, 180_windows_as_timesteps)

            # If the "180 time steps" must come from a SINGLE 2s window for each of the 15 features:
            # This is very unusual. The DWT coefficients for a 500-sample window are:
            # cA5: ~19 samples, cD5: ~19, cD4: ~35, cD3: ~67, cD2: ~130, cD1: ~257
            # To get 180 time steps for *each feature derived from these*:
            # One extremely forced method: for each level's coefficients, resample them to 180 points.
            # Then calculate mean, std, energy *of these resampled coefficient series*. Still results in scalars.
            # The paper's wording "reshaping or selecting coefficients ... to form a fixed-length sequence" is key.
            # It implies the *coefficients themselves* are formed into sequences of 180.
            # Let's assume the features (mean, std, energy) are scalar per level, and then these 3 scalars for each of 5 levels
            # are somehow expanded or used to generate a 180-timestep representation.
            # This requires significant assumption or clarification from the MHA-GCN ViT authors.

            # **Proposal due to ambiguity:** Assume the 15 DWT features (mean, std, energy for 5 detail levels) are calculated as SCALARS per 2s window.
            # The `15*180` input for `GCNConv` in `BUILD_OVERVIEW.MD` (BO, Section 3.A) might imply that for each node,
            # a sequence of 180 such 15-feature vectors (from 180 consecutive 2s windows) is flattened.
            # So, input to MHA-GCN is `[num_nodes, features_per_node] = [29, 15 * 180]`.
            # This matches the "180 windows as timesteps" interpretation for creating the feature vector for one graph.

            # Therefore, the `extract_dwt_features` function should return 15 scalar features for a single 2s window.
            # These will then be collected from 180 windows to form the input for one GCN forward pass.
            current_window_dwt_features = []
            for i in range(1, 6): # cD5 to cD1
                detail_coeffs = coeffs[i]
                current_window_dwt_features.append(np.mean(detail_coeffs))
                current_window_dwt_features.append(np.std(detail_coeffs))
                current_window_dwt_features.append(np.sum(np.square(detail_coeffs)))
            return np.array(current_window_dwt_features) # Shape (15,)
        ```

* **STFT for EEG Spectrograms (ViT Input)**:
    * Transform a 2-second EEG window (250 Hz * 2s = 500 samples) into a 224x224 image.
    * MHA-GCN ViT Paper, Section 4.2: "STFT was applied to these 2-s EEG epochs to generate time-frequency spectrograms (224×224 pixels)".
        ```python
        from scipy.signal import stft
        from skimage.transform import resize # For resizing image

        def extract_stft_spectrogram_eeg(epoch_data_all_channels, sfreq=250, target_size=(224, 224)):
            # epoch_data_all_channels: (n_channels, n_times_in_window), e.g. (29, 500)
            
            # Option 1: Average channels then STFT (less common)
            # Option 2: STFT per channel, then average spectrograms (more common for single image)
            # Option 3: Select one channel (e.g., Cz)
            
            # Let's go with Option 2: STFT per channel, average magnitudes
            n_channels, n_times = epoch_data_all_channels.shape
            
            # STFT parameters to get a reasonable starting resolution.
            # To get ~224 time bins from 500 samples: nperseg should be small.
            # To get ~224 freq bins: nfft should be high enough.
            # Example:
            nperseg = 32 # Small window for STFT to get more time bins
            noverlap = nperseg // 2 
            nfft = 256 # Number of FFT points

            all_channel_spectrograms = []
            for i in range(n_channels):
                f, t, Zxx = stft(epoch_data_all_channels[i, :], fs=sfreq, window='hann', 
                                 nperseg=nperseg, noverlap=noverlap, nfft=nfft)
                # Zxx shape: (n_freq_bins, n_time_bins)
                # Use magnitude or power
                spectrogram = np.abs(Zxx)
                all_channel_spectrograms.append(spectrogram)
            
            avg_spectrogram = np.mean(np.array(all_channel_spectrograms), axis=0)
            
            # Resize to target_size (224, 224)
            # Ensure positive values before log if taking log spectrogram
            epsilon = 1e-10
            log_spectrogram = np.log(avg_spectrogram + epsilon)
            
            # Resize (order=1 for bilinear, order=0 for nearest neighbor)
            resized_spectrogram = resize(log_spectrogram, target_size, anti_aliasing=True, mode='reflect')
            
            # Normalize (e.g., to [0,1] or z-score)
            min_val = np.min(resized_spectrogram)
            max_val = np.max(resized_spectrogram)
            if max_val > min_val:
                norm_spectrogram = (resized_spectrogram - min_val) / (max_val - min_val)
            else:
                norm_spectrogram = np.zeros(target_size)

            # ViT input is often (3, H, W). Repeat grayscale to 3 channels.
            # MHA-GCN ViT Paper (Section 4.3) mentions ViT input as (3, 224, 224).
            three_channel_spectrogram = np.stack([norm_spectrogram] * 3, axis=0)
            return three_channel_spectrogram # Shape (3, 224, 224)
        ```

* **Pearson Correlation for Adjacency Matrix**:
    * `np.corrcoef(channel_features)` then threshold at 0.3 (BO, Section 2.C).
    * The "channel\_features" are typically the time series data of the channels themselves over a window.
    * MHA-GCN ViT Paper (Section 3.2.2): "calculating the Pearson correlation coefficient between channels, from which an adjacency matrix is constructed". This implies correlation of raw/preprocessed EEG signals.
        ```python
        def compute_adjacency_matrix(epoch_data_all_channels, threshold=0.3):
            # epoch_data_all_channels: (n_channels, n_times_in_window), e.g. (29, 500)
            # This computes correlation based on the time-series data within one 2s window.
            adj_matrix = np.corrcoef(epoch_data_all_channels) # Shape (n_channels, n_channels)
            adj_matrix[np.abs(adj_matrix) < threshold] = 0 # Apply threshold
            # adj_matrix[np.arange(len(adj_matrix)), np.arange(len(adj_matrix))] = 0 # Optional: Remove self-loops if GCN handles them implicitly or if not desired
            return adj_matrix
        ```

---

### **4. Explicit Model Implementation Details** 🧠

* **MHA_GCN Class** (BO, Section 3.A; MHA-GCN ViT Paper, Section 3.3, Figure 3): [cite: 297]
    * Input to `gcn_layer1`: `in_channels=15*180`. This means each of the 29 nodes has a feature vector of size 2700. This corresponds to concatenating/flattening the (15 DWT features * 180 time steps/windows).
    * MHA-GCN ViT Paper (Figure 3) shows Multi-Head Attention being applied *after* the GCN layers to the node embeddings.
        ```python
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        # Assuming torch_geometric is used for GCNConv
        # from torch_geometric.nn import GCNConv 
        # If not, a custom GCNConv can be defined or nn.Linear can be used with adj matrix multiplication.
        # For simplicity, assuming a GCNConv-like layer exists or is defined.

        # Placeholder for GCNConv if torch_geometric is not available/used
        class SimpleGCNConv(nn.Module):
            def __init__(self, in_channels, out_channels):
                super().__init__()
                self.linear = nn.Linear(in_channels, out_channels)

            def forward(self, x, adj):
                # x: [N, in_channels], adj: [N, N]
                # Simple GCN: A_hat * X * W
                # Normalize adjacency matrix (D^-0.5 * A * D^-0.5)
                N = adj.size(0)
                I = torch.eye(N, device=adj.device)
                A_hat = adj + I 
                D_hat_diag = torch.sum(A_hat, dim=1)
                D_hat_diag_inv_sqrt = torch.pow(D_hat_diag, -0.5)
                D_hat_diag_inv_sqrt[torch.isinf(D_hat_diag_inv_sqrt)] = 0. # Handle isolated nodes
                D_inv_sqrt = torch.diag(D_hat_diag_inv_sqrt)
                
                norm_adj = torch.matmul(torch.matmul(D_inv_sqrt, A_hat), D_inv_sqrt)
                
                support = torch.matmul(norm_adj, x) # [N, N] * [N, in_channels] -> [N, in_channels]
                output = self.linear(support) # [N, out_channels]
                return output

        class MHA_GCN(nn.Module):
            def __init__(self, num_nodes=29, node_feature_dim=15*180, gcn1_out_channels=128, gcn2_out_channels=512, 
                         mha_num_heads=4, mha_out_dim=None, num_classes=5): # num_classes from ViT config
                super().__init__()
                # GCNConv from PyG or custom implementation
                self.gcn_layer1 = SimpleGCNConv(in_channels=node_feature_dim, out_channels=gcn1_out_channels)
                self.gcn_layer2 = SimpleGCNConv(in_channels=gcn1_out_channels, out_channels=gcn2_out_channels)
                
                self.attention_heads = mha_num_heads
                self.hidden_dim_gcn = gcn2_out_channels # Output dim of GCN part
                
                # Multi-Head Attention (applied to node features after GCN layers)
                # MHA-GCN ViT paper (Fig 3) indicates MHA after GCN layers.
                # The output of GCN is typically node embeddings [num_nodes, gcn2_out_channels]
                # We can apply MHA over these node embeddings.
                self.mha = nn.MultiheadAttention(embed_dim=self.hidden_dim_gcn, 
                                                 num_heads=self.attention_heads, 
                                                 dropout=0.1, batch_first=True) # batch_first=True if input is [batch, seq_len, features]
                                                                               # Here, seq_len = num_nodes

                if mha_out_dim is None:
                    mha_out_dim = self.hidden_dim_gcn

                # Classifier part
                # Option 1: Global pooling over node features after MHA, then Linear
                self.fc_out_gcn = nn.Linear(mha_out_dim, num_classes) 
                # Or if mha_out_dim * num_nodes is used:
                # self.fc_out_gcn = nn.Linear(mha_out_dim * num_nodes, num_classes)


            def forward(self, node_features, adjacency_matrix):
                # node_features: Tensor of shape [batch_size, num_nodes, node_feature_dim] or [num_nodes, node_feature_dim] if batch_size=1
                # adjacency_matrix: Tensor of shape [batch_size, num_nodes, num_nodes] or [num_nodes, num_nodes]
                
                # Assuming batch_size = 1 for GCN processing in typical LOOCV for now.
                # If batched, GCNConv needs to handle batches or loop.
                # SimpleGCNConv as defined above expects [N, features]
                
                if node_features.ndim == 3: # If batch dim is present
                    # Process each item in batch (this is a simplification, PyG handles batching)
                    outputs = []
                    for i in range(node_features.shape[0]):
                        x_i = node_features[i] # [num_nodes, node_feature_dim]
                        adj_i = adjacency_matrix[i] # [num_nodes, num_nodes]
                        
                        x_i = F.relu(self.gcn_layer1(x_i, adj_i))
                        x_i = self.gcn_layer2(x_i, adj_i) # Output: [num_nodes, gcn2_out_channels]
                        
                        # Apply MHA: MHA expects (seq_len, batch_size, embed_dim) or (batch_size, seq_len, embed_dim)
                        # Here, seq_len is num_nodes. Let's treat num_nodes as sequence.
                        # Input to MHA: (num_nodes, 1, hidden_dim_gcn) if batch_first=False
                        # Or (1, num_nodes, hidden_dim_gcn) if batch_first=True
                        x_i_mha_input = x_i.unsqueeze(0) # Add batch dim for MHA: [1, num_nodes, hidden_dim_gcn]
                        attn_output, _ = self.mha(x_i_mha_input, x_i_mha_input, x_i_mha_input) # Query, Key, Value are the same
                        attn_output = attn_output.squeeze(0) # Back to [num_nodes, mha_out_dim]
                        
                        # Global Average Pooling over nodes
                        graph_embedding = torch.mean(attn_output, dim=0) # [mha_out_dim]
                        
                        out_gcn = self.fc_out_gcn(graph_embedding)
                        outputs.append(out_gcn)
                    return torch.stack(outputs)
                else: # batch_size = 1
                    x = F.relu(self.gcn_layer1(node_features, adjacency_matrix))
                    x = self.gcn_layer2(x, adjacency_matrix) # Output: [num_nodes, gcn2_out_channels]

                    x_mha_input = x.unsqueeze(0) # [1, num_nodes, hidden_dim_gcn]
                    attn_output, _ = self.mha(x_mha_input, x_mha_input, x_mha_input)
                    attn_output = attn_output.squeeze(0) # [num_nodes, mha_out_dim]
                    
                    graph_embedding = torch.mean(attn_output, dim=0) # [mha_out_dim]
                    out_gcn = self.fc_out_gcn(graph_embedding)
                    return out_gcn
        ```

* **DepL_GCN Model/Modules** (BO, Section 3.B; DepL-GCN Paper):
    * **Base GCN**: This could be the `MHA_GCN` defined above, or a simpler GCN if MHA is specific to the MHA-GCN\_ViT model. The DepL-GCN paper proposes SSPA-GCN as a base but also mentions using GCN for depression level recognition. [cite: 59, 79] For this blueprint, let's assume a generic GCN structure (like the GCN part of MHA\_GCN before attention, or SSPA-GCN if its specifics are provided).
    * **`calculate_NeL2`**:
        * `eL2`: "L2-norm of prediction errors" (DepL-GCN Paper, Section III.C). This is $|| P_i - Y_i ||_2$, where $P_i$ is the predicted probability vector for sample $i$, and $Y_i$ is the one-hot true label vector.
            ```python
            def calculate_eL2(model_output_probs, true_one_hot_labels):
                # model_output_probs: [batch_size, num_classes]
                # true_one_hot_labels: [batch_size, num_classes]
                return torch.norm(model_output_probs - true_one_hot_labels, p=2, dim=1)
            ```
        * `LeL2`: "The L2-norm of the prediction error for each sample i in each epoch e is denoted as $eL_2(i,e)$. We maintain a record, $LeL_2(i,e)$, which stores the $eL_2(i,e)$ from epoch 1 to e for sample i." (DepL-GCN Paper, Section III.C). This means $LeL_2(i,e)$ is a list or array of $eL_2$ values for sample $i$ up to epoch $e$. The formula `NeL2 = LeL2 - u_rate * (LeL2 - eL2)` (DepL-GCN Paper, Eq. 5) implies `LeL2` in this formula is actually the *accumulated historical error norm up to the previous epoch* for that sample or an average/smoothed version. The paper states "average value of $eL_2(i,k)$ for sample $i$ from epoch 1 to $e-1$." This means $LeL_2(i,e) = \frac{1}{e-1} \sum_{k=1}^{e-1} eL_2(i,k)$. For the first epoch, $LeL_2$ could be initialized to $eL_2(i,1)$.
            * `LeL2_storage`: A dictionary or array mapping `sample_id` to a list of its `eL2` values over epochs.
            * In `calculate_NeL2(LeL2_historical_avg, current_eL2, u_rate=0.6)`:
                * `LeL2_historical_avg` is the average $eL_2$ for that sample up to the *previous* epoch.
                * `current_eL2` is the $eL_2$ for the current epoch.
        * `calculate_NeL2` function (BO, Section 3.B):
            ```python
            def calculate_NeL2(LeL2_historical_avg_prev_epoch, current_eL2_this_epoch, u_rate=0.6):
                # DepL-GCN Paper, Eq. 5
                return LeL2_historical_avg_prev_epoch - u_rate * (LeL2_historical_avg_prev_epoch - current_eL2_this_epoch)
            ```
    * **`minority_penalty`**:
        * `max(NeL2)`: "a statistical maximum that is set to approximately 80% of the theoretical maximum of $NeL_2$" (DepL-GCN Paper, Section III.D). The theoretical maximum of $eL_2$ (for probabilities vs one-hot) is $\sqrt{ (1-0)^2 + (0-1)^2 + \sum (0-0)^2 } = \sqrt{2}$ if num\_classes >= 2. $NeL_2$ is related to this.
            * **Practical implementation for `max(NeL2)`**: It could be dynamically estimated as the 80th percentile of $NeL_2$ values observed so far in the current epoch or a moving window, or a fixed value if the range of $NeL_2$ is somewhat stable. Using $0.8 \times \sqrt{2}$ might be a starting point if $NeL_2$ values are in a similar range to $eL_2$. Or, as the BO suggests, `max(NeL2)` could mean `torch.max(current_epoch_NeL2_values_for_minority_samples)`. The paper is a bit vague on "theoretical maximum of $NeL_2$". Given the formula uses `NeL2 / max(NeL2)`, it's a relative scaling. Let's assume `max_NeL2_val` is the maximum observed `NeL2` for minority samples in the current batch/epoch.
            ```python
            def minority_penalty_func(prediction_correct, is_minority_sample, NeL2_for_sample, max_NeL2_in_batch_minority):
                # BO, Section 3.B; DepL-GCN Paper, Eq. 6
                # Prediction is correct if prediction == true_label
                # The penalty applies if is_minority and prediction is WRONG.
                penalty = 0.0
                if is_minority_sample and not prediction_correct:
                    if max_NeL2_in_batch_minority > 1e-6: # Avoid division by zero
                         # The paper says "NeL2 / max(NeL2)". Here NeL2_for_sample is the NeL2 of the current misclassified minority sample.
                         # max_NeL2_in_batch_minority refers to the max NeL2 among all minority samples (or just misclassified ones).
                         # Let's assume it's the max NeL2 among all minority samples in the current context (e.g., batch).
                        penalty = NeL2_for_sample / max_NeL2_in_batch_minority
                    else: # if max_NeL2 is zero (e.g. all NeL2 are zero for minority samples)
                        penalty = NeL2_for_sample # or 1.0 if NeL2_for_sample > 0
                return penalty 
            ```
    * **Loss Function Modification** (DepL-GCN Paper, Section III.E, Eq. 7, 8):
        * The sample confidence $SC_i = 1 - NeL_2(i,e) / \text{max}(NeL_2(k,e))$ for $k \in \text{all samples}$.
        * The final weight for sample $i$ is $w_i = SC_i + MP_i$.
        * `WeightedCrossEntropyLoss_i = w_i * CrossEntropyLoss_i`.
        ```python
        class DepLCustomLoss(nn.Module):
            def __init__(self):
                super().__init__()
                self.ce_loss = nn.CrossEntropyLoss(reduction='none') # Get loss per sample

            def forward(self, model_outputs, true_labels_one_hot, true_labels_idx, NeL2_values, 
                        is_minority_flags, max_NeL2_all_samples, max_NeL2_minority_samples):
                # model_outputs: [batch_size, num_classes] (logits or probabilities)
                # true_labels_one_hot: [batch_size, num_classes]
                # true_labels_idx: [batch_size] (indices for CE loss)
                # NeL2_values: [batch_size] (NeL2 for each sample in batch)
                # is_minority_flags: [batch_size] (boolean)
                # max_NeL2_all_samples: scalar, max NeL2 across all samples in current context (e.g. epoch) for SC calculation
                # max_NeL2_minority_samples: scalar, max NeL2 among minority samples for MP calculation

                batch_ce_loss = self.ce_loss(model_outputs, true_labels_idx)
                
                # Sample Confidence (SC)
                # SC_i = 1 - NeL2_i / max_NeL2_overall. Ensure max_NeL2_overall is not zero.
                safe_max_NeL2_all = max_NeL2_all_samples if max_NeL2_all_samples > 1e-6 else 1.0
                sample_confidence = 1.0 - (NeL2_values / safe_max_NeL2_all)
                sample_confidence = torch.clamp(sample_confidence, min=0.0) # SC should be non-negative

                # Minority Penalty (MP)
                predictions = torch.argmax(model_outputs, dim=1)
                correct_predictions = (predictions == true_labels_idx)
                
                minority_penalties = torch.zeros_like(NeL2_values)
                safe_max_NeL2_minority = max_NeL2_minority_samples if max_NeL2_minority_samples > 1e-6 else 1.0

                for i in range(len(NeL2_values)):
                    if is_minority_flags[i] and not correct_predictions[i]:
                        minority_penalties[i] = NeL2_values[i] / safe_max_NeL2_minority
                
                minority_penalties = torch.clamp(minority_penalties, min=0.0) # Penalty should be non-negative

                # Final sample weights
                sample_weights = sample_confidence + minority_penalties
                
                weighted_loss = (batch_ce_loss * sample_weights).mean()
                return weighted_loss
        ```

* **Vision Transformer (ViT)** (BO, Section 3.C):
    * Use a pre-built ViT from `timm` is highly recommended. Example: `vit_base_patch16_224`.
        ```python
        import timm
        def get_vit_model(num_classes=5, pretrained=True):
            # BO vit_config: image_size: 224, patch_size: 16, num_classes: 5, dim: 768, depth: 12, heads: 12, mlp_dim: 3072
            # This matches vit_base_patch16_224
            model_name = 'vit_base_patch16_224'
            vit_model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes, in_chans=3)
            # Default in_chans=3. If 1-channel spectrograms are used before repeating to 3:
            # vit_model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes, in_chans=1)
            return vit_model
        ```
    * Adaptation for 1-channel vs 3-channel: The `extract_stft_spectrogram_eeg` already outputs 3-channel by repeating. If it were to output 1-channel, `in_chans=1` would be set in `timm.create_model`.

* **Decision-Level Fusion (for MHA-GCN\_ViT)** (MHA-GCN ViT Paper, Section 3.5):
    * Formula: $P_i = \epsilon_1 \cdot m_i + \epsilon_2 \cdot n_i + \epsilon_3 \cdot v_i$.
    * $m_i, n_i, v_i$ are probability outputs from three model paths. The paper diagram (Fig 1) suggests:
        * Path 1 (EEG DWT): MHA-GCN -> $m_i$
        * Path 2 (EEG STFT): ViT -> $n_i$
        * Path 3 (Speech STFT/Mel): ViT -> $v_i$
    * $\epsilon_1, \epsilon_2, \epsilon_3$ are weights. The paper mentions "weights are determined by the validation accuracy of each modality's features on the validation set."
    * During LOOCV, for each fold, the "training set" of 52 subjects can be further split into sub-train and sub-validation. The accuracies on this sub-validation set determine $\epsilon$ values for the test subject of that LOOCV fold. Or, they can be fixed based on average accuracies observed across folds/models. A simpler approach is to make them learnable or set them empirically (e.g., $\epsilon_1=\epsilon_2=\epsilon_3=1/3$ or tune them).
    * If validation accuracies are $acc_1, acc_2, acc_3$, then $\epsilon_j = acc_j / (acc_1+acc_2+acc_3)$.
        ```python
        class Combined_MHA_GCN_ViT(nn.Module):
            def __init__(self, mha_gcn_model, eeg_vit_model, speech_vit_model, num_classes=5):
                super().__init__()
                self.mha_gcn = mha_gcn_model
                self.eeg_vit = eeg_vit_model
                self.speech_vit = speech_vit_model
                
                # Fusion weights (can be fixed, learned, or set based on validation accuracies)
                self.epsilon1 = nn.Parameter(torch.tensor([1/3])) # Example, make learnable
                self.epsilon2 = nn.Parameter(torch.tensor([1/3]))
                self.epsilon3 = nn.Parameter(torch.tensor([1/3]))
                # Or set them non-learnable:
                # self.register_buffer('epsilon1', torch.tensor([0.4])) # If e.g. EEG-DWT is better
                # self.register_buffer('epsilon2', torch.tensor([0.3]))
                # self.register_buffer('epsilon3', torch.tensor([0.3]))


            def forward(self, dwt_features, adj_matrix, eeg_spectrogram, speech_spectrogram):
                # dwt_features: input for MHA-GCN
                # adj_matrix: input for MHA-GCN
                # eeg_spectrogram: input for EEG ViT
                # speech_spectrogram: input for Speech ViT
                
                # Get probability distributions from each model path
                # Assume models output logits, so apply softmax
                
                out_mha_gcn_logits = self.mha_gcn(dwt_features, adj_matrix)
                probs_mha_gcn = F.softmax(out_mha_gcn_logits, dim=-1)
                
                out_eeg_vit_logits = self.eeg_vit(eeg_spectrogram)
                probs_eeg_vit = F.softmax(out_eeg_vit_logits, dim=-1)
                
                if speech_spectrogram is not None and self.speech_vit is not None:
                    out_speech_vit_logits = self.speech_vit(speech_spectrogram)
                    probs_speech_vit = F.softmax(out_speech_vit_logits, dim=-1)
                    
                    # Normalize weights (if they are learnable and unconstrained)
                    # total_epsilon = self.epsilon1 + self.epsilon2 + self.epsilon3
                    # e1 = self.epsilon1 / total_epsilon
                    # e2 = self.epsilon2 / total_epsilon
                    # e3 = self.epsilon3 / total_epsilon
                    # fused_probs = e1 * probs_mha_gcn + e2 * probs_eeg_vit + e3 * probs_speech_vit
                    
                    # Simpler fixed or pre-set weights:
                    fused_probs = self.epsilon1 * probs_mha_gcn + \
                                  self.epsilon2 * probs_eeg_vit + \
                                  self.epsilon3 * probs_speech_vit
                else: # Only EEG modalities
                    # Adjust weights if speech is not used
                    # total_epsilon = self.epsilon1 + self.epsilon2
                    # e1 = self.epsilon1 / total_epsilon
                    # e2 = self.epsilon2 / total_epsilon
                    # fused_probs = e1 * probs_mha_gcn + e2 * probs_eeg_vit
                    # Or assume epsilons are set to sum to 1 for the paths used.
                    # If speech_vit is None, assume epsilon3 is effectively 0.
                    fused_probs = (self.epsilon1 * probs_mha_gcn + self.epsilon2 * probs_eeg_vit) / (self.epsilon1 + self.epsilon2)


                # Return fused probabilities. For loss calculation, often need logits before softmax.
                # To get "fused logits" is harder. Usually fusion is at probability or decision level.
                # If CE loss expects logits, can return log_softmax of fused_probs.
                return torch.log(fused_probs + 1e-10) # Return log-probabilities for NLLLoss, or just probs
        ```

---

### **5. Training and Validation Loop Logic** 🔁

* **Leave-One-Out Cross-Validation (LOOCV)**:
    * Outer loop: `for test_subject_idx in range(NUM_SUBJECTS):`
    * `train_subject_indices = [i for i in range(NUM_SUBJECTS) if i != test_subject_idx]`
    * `test_subject_indices = [test_subject_idx]`
    * Prepare data for these sets. Each "sample" for training/testing will be a 2s window (for features like DE, STFT for ViT) or a sequence of 180 2s windows (for DWT features for MHA-GCN if that interpretation is used).
    * **DepL-GCN Sample Confidence within LOOCV**:
        * `LeL2_storage`: Re-initialized for each LOOCV training fold. It's a dictionary mapping `(fold_id, sample_id_within_fold)` to its list of `eL2` values across epochs of that fold's training.
        * `conf_start_epoch`: e.g., `int(0.4 * NUM_EPOCHS)` or a fixed number like 40. (BO, Section 7.3; DepL-GCN Paper, Section III.E implies confidence calculation starts after some epochs). The DepL-GCN paper actual text says "After the model has been trained for a certain number of epochs (e.g., 40-60 epochs), the $NeL_2$ values for each sample tend to stabilize." It's when to *trust* $NeL_2$ for weighting. The $eL_2$ and $LeL_2$ are calculated from epoch 1.

* **Batch Processing**:
    * `BATCH_SIZE = 4` (BO, Section 4.5).
    * DataLoaders will yield batches of EEG windows/segments.
    * If MHA-GCN path expects features from 180 windows, one item in a batch for this path would be data from 180 windows.

* **Optimizer and Loss**:
    * Optimizer: `torch.optim.Adam(model.parameters(), lr=0.001)` (BO, Section 4.5).
    * Loss: `nn.CrossEntropyLoss()` or `nn.NLLLoss()` if model outputs log-softmax. For DepL-GCN, use the `DepLCustomLoss`.

* **Pseudocode for Training Loop (single fold)**:
    ```python
    # model, train_loader, optimizer, criterion (loss_fn), device, num_epochs
    # LeL2_storage = {} # {sample_id: [eL2_epoch1, eL2_epoch2, ...]}
    # current_NeL2_values_epoch = {} # {sample_id: NeL2_value_this_epoch}
    # conf_start_epoch = 40 # Or from config

    # for epoch in range(num_epochs):
    #     model.train()
    #     epoch_loss = 0
    #     all_eL2_this_epoch_list = [] # For calculating max_NeL2 for SC
    #     all_NeL2_this_epoch_minority_list = [] # For calculating max_NeL2 for MP

    #     # First pass for eL2 if needed by DepL-GCN, or calculate on the fly
    #     # If DepL-GCN custom loss needs max_NeL2 across whole epoch:
    #     # temp_eL2_for_epoch = []
    #     # temp_NeL2_for_epoch_minority = []
    #     # for batch_idx, (data, target_idx, target_one_hot, sample_ids, is_minority) in enumerate(train_loader):
    #     #    # ... calculate eL2 and NeL2, store them ...
    #     # max_overall_NeL2_for_SC = max(temp_eL2_for_epoch) if temp_eL2_for_epoch else 1.0
    #     # max_minority_NeL2_for_MP = max(temp_NeL2_for_epoch_minority) if temp_NeL2_for_epoch_minority else 1.0
        
    #     for batch_idx, (data_batch) in enumerate(train_loader):
    #         # data_batch should contain:
    #         # - eeg_dwt_features_for_mha_gcn [batch, 29_nodes, 15*180_features_flat] or [batch, 29_nodes, 15_feat, 180_steps]
    #         # - adjacency_matrices [batch, 29, 29]
    #         # - eeg_stft_spectrograms [batch, 3, 224, 224]
    #         # - speech_stft_spectrograms [batch, 3, 224, 224] (if used)
    #         # - true_labels_idx [batch]
    #         # - true_labels_one_hot [batch, num_classes]
    #         # - sample_ids_in_batch [batch] (unique IDs for DepL-GCN storage)
    #         # - is_minority_flags_in_batch [batch] (boolean for DepL-GCN)

    #         # Unpack data_batch and move to device
    #         # ...
            
    #         optimizer.zero_grad()
    #         outputs = model(dwt_features, adj_matrix, eeg_spectrogram, speech_spectrogram) # Example for MHA-GCN_ViT

    #         if isinstance(criterion, DepLCustomLoss):
    #             # Calculate current eL2 for samples in batch
    #             current_eL2_batch = calculate_eL2(F.softmax(outputs, dim=-1), true_labels_one_hot)
                
    #             batch_NeL2_values = []
    #             for i, sample_id in enumerate(sample_ids_in_batch):
    #                 # Update LeL2_storage
    #                 if sample_id not in LeL2_storage: LeL2_storage[sample_id] = []
    #                 LeL2_storage[sample_id].append(current_eL2_batch[i].item())
                    
    #                 # Calculate LeL2_historical_avg for NeL2 calculation
    #                 if epoch == 0:
    #                     LeL2_hist_avg = current_eL2_batch[i] # For first epoch, use current eL2 as historical
    #                 else:
    #                     # Avg of eL2s from epoch 0 to e-1
    #                     # The DepL-GCN paper suggests avg up to e-1.
    #                     # LeL2_storage[sample_id] already stores up to previous epoch from previous iterations.
    #                     # Or, more simply, use the running NeL2 from previous epoch as LeL2.
    #                     # For simplicity, let's assume `current_NeL2_values_epoch` stores NeL2 from *previous* epoch.
    #                     # LeL2_hist_avg = current_NeL2_values_epoch.get(sample_id, current_eL2_batch[i].item()) # Approx.
    #                     prev_eL2s = LeL2_storage[sample_id][:-1] # all eL2s before current
    #                     if not prev_eL2s: LeL2_hist_avg = current_eL2_batch[i].item() # if first time this sample is seen this epoch after ep0
    #                     else: LeL2_hist_avg = np.mean(prev_eL2s)


    #                 NeL2_val = calculate_NeL2(LeL2_hist_avg, current_eL2_batch[i], u_rate=0.6)
    #                 batch_NeL2_values.append(NeL2_val)
    #                 current_NeL2_values_epoch[sample_id] = NeL2_val.item() # Store for next epoch or SC/MP max calculation
                
    #             batch_NeL2_tensor = torch.tensor(batch_NeL2_values, device=device)
                
    #             # These max values should ideally be from the entire epoch's (or recent history) NeL2 values
    #             # For simplicity in batch processing, can use batch max or running epoch max
    #             max_NeL2_for_SC_calc = torch.max(batch_NeL2_tensor) if len(batch_NeL2_tensor)>0 else torch.tensor(1.0)
    #             minority_NeL2s_in_batch = batch_NeL2_tensor[is_minority_flags_in_batch]
    #             max_NeL2_for_MP_calc = torch.max(minority_NeL2s_in_batch) if len(minority_NeL2s_in_batch)>0 else torch.tensor(1.0)

    #             if epoch < conf_start_epoch: # Before confidence mechanism kicks in fully
    #                 loss = nn.CrossEntropyLoss()(outputs, true_labels_idx) # Standard CE
    #             else:
    #                 loss = criterion(outputs, true_labels_one_hot, true_labels_idx, 
    #                                  batch_NeL2_tensor, is_minority_flags_in_batch,
    #                                  max_NeL2_for_SC_calc, max_NeL2_for_MP_calc)
    #         else: # Standard loss
    #             loss = criterion(outputs, true_labels_idx)
            
    #         loss.backward()
    #         optimizer.step()
    #         epoch_loss += loss.item()
            
    #     # print(f"Fold {fold_num}, Epoch {epoch+1}, Loss: {epoch_loss/len(train_loader)}")

    # # After all epochs in a fold, evaluate on test_subject for that fold
    # model.eval()
    # all_preds = []
    # all_true = []
    # with torch.no_grad():
    #    for data_batch_test in test_loader:
    #        # ... unpack ...
    #        outputs_test = model(...)
    #        preds = torch.argmax(outputs_test, dim=1)
    #        all_preds.extend(preds.cpu().numpy())
    #        all_true.extend(true_labels_idx_test.cpu().numpy())
    # # Calculate metrics for this fold using all_preds, all_true
    ```

* **Metrics Calculation & Aggregation**:
    * Per fold: `accuracy_score`, `precision_score(average='macro')`, `recall_score(average='macro')`, `f1_score(average='macro')`, `confusion_matrix`.
    * Store these for each fold.
    * Final results: Average the scores (accuracy, precision, recall, F1) across all folds. Sum confusion matrices.

---

### **6. Interactive Visualization - Concrete Definitions** 📊

* **Channel Selector (`channel_dropdown`)**:
    * Options: `CHANNELS_29` (BO, Section 1).
    * Behavior: When a channel is selected:
        * Display its raw EEG signal for a sample 2s window from a loaded subject (e.g., `epochs.get_data()[0, CHANNELS_29.index(selected_channel), :]`). Plot using Plotly.
        * Display its Power Spectral Density (PSD) for that window using Plotly (calculated via Welch's method).
        * Display its Differential Entropy values across the 5 `FREQ_BANDS` for that window.

* **Real-time Metrics Display (`metrics_display`)**:
    * This won't be truly "real-time" in a static notebook but will update based on selected data.
    * Content:
        * Current Subject ID being viewed.
        * Selected EEG window number (if applicable).
        * For a selected channel (from `channel_dropdown`):
            * Calculated DE values (delta, theta, alpha, beta, gamma).
        * If a model is loaded and prediction is run on the sample window:
            * Model's predicted depression level (e.g., 'Mild').
            * Probability scores for each depression level.
    * Mechanism: Updates when a new subject/window/channel is selected via widgets.

* **Model Comparison Matrix (`comparison_heatmap`)**:
    * Rows: Models (e.g., "DepL-GCN", "MHA-GCN\_ViT", "Baseline CNN").
    * Columns: Metrics (e.g., "Overall Accuracy", "F1-Normal", "F1-Mild", "F1-Moderate", "F1-Mod-Major", "F1-Major"). Per-class F1 for multi-class.
    * Cell Values: The scores achieved by each model for each metric (averaged from LOOCV).
    * Data Source: Populate from the stored LOOCV results.

* **2D Brain Topography (`mne.viz.plot_topomap`)**:
    * Data to plot:
        * Average DE power for a selected frequency band (e.g., alpha) for the current subject (averaged over all their windows or a specific window).
        * Or average PSD in a band.
    * `pos` (channel locations):
        ```python
        # CHANNELS_29 should be available
        # Need to filter the standard montage to only these 29 channels.
        standard_montage = mne.channels.make_standard_montage('standard_1020')
        
        # Create a new DigMontage for the selected 29 channels
        # Get positions for the 29 channels
        ch_positions_dict = {}
        present_channels_in_montage = []
        for ch_name in CHANNELS_29:
            try:
                idx_in_montage = standard_montage.ch_names.index(ch_name)
                ch_positions_dict[ch_name] = standard_montage.dig[idx_in_montage]['r'] # 'r' is position
                present_channels_in_montage.append(ch_name)
            except ValueError:
                # Try common alternatives like Fpz -> FPz
                if ch_name == 'Fpz' and 'FPz' in standard_montage.ch_names:
                    idx_in_montage = standard_montage.ch_names.index('FPz')
                    ch_positions_dict['FPz'] = standard_montage.dig[idx_in_montage]['r']
                    present_channels_in_montage.append('FPz') # Use the name found in montage
                elif ch_name == 'Iz' and 'I1' in standard_montage.ch_names: # Example for Iz
                    idx_in_montage = standard_montage.ch_names.index('I1')
                    ch_positions_dict['I1'] = standard_montage.dig[idx_in_montage]['r']
                    present_channels_in_montage.append('I1')
                else:
                    print(f"Channel {ch_name} not found in standard_1020 montage. Will be excluded from topomap.")

        # Create a new montage for only the available channels among the 29
        # This requires channel names to match what plot_topomap expects via info object
        # It's easier to pass 'info' object to plot_topomap which contains the montage.
        # Create an MNE Info object for the 29 channels
        # Ensure data passed to plot_topomap is ordered according to info.ch_names
        
        # Example of creating info for plotting:
        # Assume `epochs` is an MNE Epochs object containing the 29 channels
        # info_29_channels = epochs.info 
        # data_for_topomap = ... # array of shape (n_channels_in_info,)
        # mne.viz.plot_topomap(data_for_topomap, info_29_channels, show=True)

        # If you only have positions and data array:
        # The data array must be in the same order as present_channels_in_montage
        # temp_info = mne.create_info(ch_names=present_channels_in_montage, sfreq=SAMPLING_RATE, ch_types='eeg')
        # custom_montage = mne.channels.make_dig_montage(ch_pos=ch_positions_dict, coord_frame='head')
        # temp_info.set_montage(custom_montage)
        # mne.viz.plot_topomap(data_for_topomap_ordered_like_present_channels, temp_info.get_montage().get_positions()['ch_pos'], show=True)
        # Or simpler: directly use mne.viz.plot_topomap(data, positions_array)
        # The `pos` argument to `plot_topomap` can be a 2D array of channel positions (n_channels, 2).
        # `mne.channels.find_layout(info).pos` can provide these 2D positions.
        ```
        The `pos` argument for `mne.viz.plot_topomap` should be obtained from an `mne.Info` object that has the montage set for the 29 channels.
        Example:
        `raw_picked = raw.copy().pick_channels(CHANNELS_29)`
        `info_for_topo = raw_picked.info`
        `channel_locations_2d = mne.channels.find_layout(info_for_topo).pos`
        `mne.viz.plot_topomap(data_array_ordered_like_info, channel_locations_2d, show=True)`

---

### **7. Implementation of Broader "Implementation Requirements"** 🛠️

* **Model Comparison Framework**:
    * **Statistical Significance Testing**:
        * After LOOCV, you'll have F1 scores (or accuracies) per fold for each model.
        * Example: `f1_scores_model1` (list of 53 F1s), `f1_scores_model2` (list of 53 F1s).
        * Use `scipy.stats.ttest_rel(f1_scores_model1, f1_scores_model2)` for a paired t-test.
        * Report p-values.

* **Clinical Report Generation Module**:
    * **Content for a single subject (based on test result from LOOCV)**:
        * Subject ID.
        * True Depression Level (if available from MODMA metadata, based on PHQ-9 scores).
        * Predicted Depression Level (from the best performing model or a selected model).
        * Probability scores for each depression level from the model.
        * Key features: This is advanced. Could involve SHAP/LIME if models are suitable, or simpler feature importance if GCN/ViT allows (e.g., attention weights, average DE per band). For now, could list the average DE values for the key bands (e.g., alpha, beta) for that subject.
        * A representative 2D topomap (e.g., alpha band power).
    * **Output Format**: Formatted markdown string displayed in a notebook cell.
        ```python
        def generate_clinical_report_snippet(subject_id, true_level, pred_level, pred_probs, 
                                           de_features_subject, topomap_fig_path=None):
            report = f"## Clinical Report Snippet: Subject {subject_id}\n\n"
            report += f"- **True Depression Level:** {true_level}\n"
            report += f"- **Predicted Depression Level (Model X):** {pred_level}\n"
            report += "- **Prediction Probabilities:**\n"
            for i, level_name in enumerate(DEPRESSION_LEVELS.keys()):
                report += f"  - {level_name}: {pred_probs[i]:.2f}\n"
            report += "- **Key Biomarkers (Example - Avg DE values across windows):\n"
            # de_features_subject: shape (num_bands,) or more detailed
            for band_idx, band_name in enumerate(FREQ_BANDS.keys()):
                 report += f"  - Avg DE {band_name}: {de_features_subject[band_idx]:.2f}\n" # Assuming de_features_subject is avg DE per band
            
            if topomap_fig_path:
                report += f"\n- **Brain Topomap (Alpha Band Power):**\n"
                report += f"  ![Topomap]({topomap_fig_path})\n" # Assumes topomap is saved as image
            return report
        ```

---

### **8. Comprehensive Hyperparameter List & Reproducibility** ⚙️

* **Data Processing**:
    * `SAMPLING_RATE = 250` Hz (BO, Section 1)
    * `WINDOW_SIZE = 2` seconds (BO, Section 1)
    * `OVERLAP = 0.5` (50%) (BO, Section 1)
    * Bandpass Filter: `l_freq=1.0`, `h_freq=40.0` Hz, FIR (see Section 2)
    * ICA: `n_components=20` (suggestion), `method='fastica'`, `random_state=42`, `max_iter='auto'` (see Section 2)
* **Feature Extraction**:
    * DWT: wavelet='db4', levels=5. (BO, Section 2.B)
    * DE: Welch params for PSD: `nperseg=500` (for 2s window), `window='hann'`. (see Section 3)
    * STFT (EEG for ViT): `nperseg=32`, `noverlap=16`, `nfft=256`, `window='hann'`. Resize to (224,224). (see Section 3)
    * STFT (Speech for ViT): `frame_length=0.025*SR_speech`, `frame_step=0.010*SR_speech`, `nfft=512`, `num_mel_bins=80`. Resize to (224,224). (see Section 2)
    * Pearson Correlation Adjacency: threshold=0.3. (BO, Section 2.C)
* **MHA-GCN (BO, Section 3.A)**:
    * `node_feature_dim = 15 * 180 = 2700`
    * `gcn1_out_channels = 128`
    * `gcn2_out_channels = 512`
    * `mha_num_heads = 4`
    * `mha_dropout = 0.1` (common default)
* **DepL-GCN (BO, Section 3.B & DepL-GCN Paper)**:
    * `u_rate = 0.6`
    * Sample confidence threshold (epochs to start applying weights): e.g., 40 epochs if total is 100-200. (BO: 40-60 epochs).
* **Vision Transformer (BO, Section 3.C)**:
    * `image_size = 224`
    * `patch_size = 16`
    * `num_classes = 5`
    * `dim = 768`
    * `depth = 12`
    * `heads = 12`
    * `mlp_dim = 3072`
    * `dropout_vit = 0.1` (common default for ViT)
    * `emb_dropout_vit = 0.1` (common default for ViT)
* **Training (BO, Section 4.5)**:
    * Optimizer: Adam, `lr=0.001`. Betas `(0.9, 0.999)` (Adam default). Epsilon `1e-8` (Adam default).
    * Loss: `CrossEntropyLoss` (or custom for DepL-GCN).
    * Batch size: `4`.
    * Epochs: `400` (This is high, might be for specific dataset/model. MHA-GCN ViT used 100, DepL-GCN used 100-200. Let's suggest 100 as a start, tunable up to 400).
    * Cross-validation: Leave-One-Out (LOOCV).
* **Data Augmentation (BO, Section 7.4)**:
    * Add noise: SNR 10-20 dB. (This would be applied to EEG window data before feature extraction).
* **Reproducibility Seeds**:
    * `numpy.random.seed(42)`
    * `torch.manual_seed(42)`
    * `torch.cuda.manual_seed_all(42)`
    * `torch.backends.cudnn.deterministic = True` (can slow down, but good for reproducibility)
    * `torch.backends.cudnn.benchmark = False`

This detailed blueprint should provide a strong foundation for the AI coding agent. Key areas requiring extremely careful implementation or potential further clarification from domain experts include the exact formation of the `[15 features, 180 time steps]` DWT feature representation from a single 2s window and the specifics of ICA component removal without dedicated EOG/ECG channels.