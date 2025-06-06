Finalizing Clarity-AI Jupyter Sandbox – Implementation Plan

clarity/data/audio.py – Audio Data Handling Module
 • Purpose: Provide a dedicated module for loading and preprocessing audio data from the MODMA dataset (e.g. speech recordings). This will serve as the optional audio pathway, kept separate from EEG processing ￼.
 • Required Functions/Classes:
 • load_audio(subject_id): Load raw audio file for a given subject (e.g., .wav or .mp3), returning a waveform tensor and sample rate. If no audio is available (current case), this can return a dummy waveform or raise a controlled warning.
 • preprocess_audio(waveform, sample_rate): Clean or normalize audio (e.g., resample to a standard rate, amplitude normalization). For now, it can pass through or apply a simple normalization since real audio specifics are unknown.
 • extract_mel_spectrogram(waveform, sample_rate, n_mels=128): Convert audio into a Mel-spectrogram or STFT spectrogram image ￼. Use torchaudio or librosa to compute a Mel-spectrogram (size e.g. 128 mel bands × time frames), then optionally convert to a 2D image (e.g. using librosa.display or saving to a NumPy array). This function will underpin feature extraction for audio.
 • Placeholder logic: If actual audio files are not present, the functions can generate placeholder outputs (e.g., a zero tensor or random spectrogram) with the correct shape, clearly marked as dummy data.
 • Data Flow & I/O: Given a subject ID, load_audio reads the file from disk (path constructed from a base audio directory and subject id). The raw waveform then passes to preprocess_audio (if any filtering needed). Finally, extract_mel_spectrogram produces features (e.g., a 2D array or PyTorch tensor) representing the audio for that subject. This module’s functions will be called inside the dataset or training loop when the audio modality is enabled.
 • Dependencies: Utilize torchaudio (for waveform loading and MelSpectrogram transformation) and librosa (as alternative for spectrogram, if needed). Matplotlib or Pillow may be used if we need to convert spectrograms to image format. These libraries are already in the environment (torchaudio is available ￼). Ensure to handle imports within this module.
 • Example Usage:

audio_wave, sr = load_audio(subject_id="15")        # Load raw audio for subject 15  
audio_wave = preprocess_audio(audio_wave, sr)       # Normalize or filter (if needed)  
mel_spec = extract_mel_spectrogram(audio_wave, sr)  # Compute Mel-spectrogram (e.g., shape [128, T])  

In practice, the dataset class will call these to retrieve each subject’s audio features. If subject_id="15" has no actual file (current scenario), the functions might log “No audio available, using placeholder” and return a dummy spectrogram.

clarity/models/audio_model.py – Audio Classification Model (Placeholder)
 • Purpose: Define a model to handle audio input (spectrogram or waveform) and output depression predictions. This provides an audio-only classification pipeline, using either a simple CNN or a pre-trained network on spectrograms. The initial implementation will use placeholder logic (e.g. random weights or a trivial network) since no real audio data is yet available.
 • Required Classes/Functions:
 • BaselineAudioCNN: A simple convolutional neural network for audio. For example, 2–3 2D conv layers (if input is spectrogram image) with pooling, followed by a fully connected layer outputting num_classes (should default to 5 for multi-class) ￼. This CNN will serve as a baseline audio model.
 • PretrainedAudioModel: Optionally, a class that wraps a pre-trained model for audio. Since MODMA audio could be speech, one idea is to use a Vision Transformer or EfficientNet pre-trained on AudioSet or ImageNet (by treating spectrogram as image) ￼. Using timm library, we can instantiate a model like timm.create_model('vit_base_patch16_224', pretrained=True) and replace its classifier head for 5 classes ￼. This class would handle forwarding a spectrogram through the pre-trained network.
 • Both classes should inherit from torch.nn.Module and implement forward(x) where x is the audio feature input (e.g., a tensor of shape (batch, channels, height, width) for spectrogram images, where channels=1 or 3). In placeholder mode, these models can be very simple (or even return a fixed output) but structured so they can be filled in later.
 • Data Flow & I/O: The audio model will receive processed audio features: if using spectrogram images, the data may be a 3-channel tensor (e.g., duplicated grayscale or RGB spectrogram) of size 224×224. BaselineAudioCNN will produce an output logit vector of length 5. PretrainedAudioModel will internally load the chosen architecture (e.g., ViT) on first call. Both output class probabilities or logits used for classification. We ensure the model interface matches the EEG models (so it can be trained/evaluated in the same loop).
 • Dependencies: PyTorch for model building; timm for pre-trained CNN/ViT models ￼ ￼. If using spectrogram images, torchvision transforms might be used to normalize/reshape the input. Ensure to add any new dependency (like a specific pre-trained model weight) to requirements if needed.
 • Example Usage:

model = BaselineAudioCNN(num_classes=5)  
dummy_spec = torch.rand(1, 1, 128, 128)  # e.g., a 128x128 mel-spectrogram  
logits = model(dummy_spec)               # forward pass (outputs a [1,5] tensor)  
probs = torch.softmax(logits, dim=1)  

For a pre-trained approach:

model = PretrainedAudioModel(num_classes=5, pretrained=True)  
spec_img = torch.rand(4, 3, 224, 224)    # batch of 4 spectrogram “images”  
outputs = model(spec_img)               # outputs shape [4,5]  

Placeholder note: Initially, BaselineAudioCNN could be extremely simple (or even output zeros) just to allow pipeline integration, marked clearly as to be replaced with real model once data is available.

clarity/models/multimodal_fusion.py – Multimodal Fusion Model (EEG+Audio)
 • Purpose: Introduce a mechanism to combine EEG and audio modalities for joint depression classification. This can be done via a fusion model that takes both EEG and audio features and produces a single prediction. It enables EEG-only, audio-only, or combined EEG+Audio experimentation by toggling components.
 • Required Classes/Functions:
 • MultimodalFusionModel: a wrapper model that contains two sub-models (one for EEG, one for audio) and a fusion strategy. For example, it could hold an eeg_model (like MHA_GCN or CNN) and an audio_model (from above). In forward(eeg_input, audio_input), it gets the separate predictions or feature embeddings from each, then fuses them. Fusion strategies to support: early fusion (concatenate high-level features before final classifier) or late fusion (average or weight the predicted probabilities from each modality) ￼. Initially, a simple late fusion can be implemented: e.g., take the softmax outputs from each model and average them element-wise to get a final probability vector ￼. Early fusion could be prepared by adding a linear layer that takes concatenated features (if both models output feature vectors from a penultimate layer).
 • The class should allow enabling/disabling each modality. For EEG-only or audio-only runs, we can instantiate just that respective model (or still use MultimodalFusionModel but with one branch disabled).
 • Utility function fuse_predictions(probs_eeg, probs_audio, method="average"): Can implement late fusion outside the model class if needed. But if using MultimodalFusionModel, it will handle fusion internally.
 • Data Flow & I/O: For combined mode, during training the LOOCV loop will create both an EEG dataset and an audio dataset. One approach is to have a combined Dataset that yields (eeg_data, audio_data, label) for each sample. The MultimodalFusionModel.forward will accept two inputs (or one combined tuple) and produce one output per sample. Internally, it will call eeg_model(eeg_data) and audio_model(audio_data). If using late fusion, it will return the averaged logits or probabilities. If early fusion, it will concatenate features and pass them through a fusion classifier layer (e.g., nn.Linear). The output is a logit or probability vector of length num_classes for each sample.
 • Dependencies: PyTorch (for model definitions). No new libraries beyond what EEG and audio models require. If using early fusion, ensure the shapes of EEG and audio feature vectors are compatible (may need to flatten or project them to a common size). Use numpy or torch for implementing any fusion math (averaging, etc.).
 • Example Usage:

eeg_model = MHA_GCN(num_classes=5)             # pre-existing EEG model  
audio_model = BaselineAudioCNN(num_classes=5)  # audio model from above  
fusion_model = MultimodalFusionModel(eeg_model, audio_model, fusion_method="average")  
eeg_x, audio_x = eeg_batch[0], audio_batch[0]   # one sample’s EEG and audio input  
output = fusion_model(eeg_x, audio_x)           # output is a [5]-sized prediction vector  

If only EEG or only audio is desired, either skip using fusion_model (and just use the single-modality model), or design MultimodalFusionModel to handle a None input for the missing modality (e.g., if audio_data is None, it just returns eeg_model output). In practice, simpler is to conditionally use the appropriate model in the training loop.

clarity/data/metadata.py (Label Ingestion & Mapping)
 • Purpose: Provide a clear mechanism to ingest real labels (depression severity scores or categories) from metadata files in the future, while using placeholder logic in the interim. This ensures the code is ready to handle multi-class labels once actual data (e.g., PHQ-9 or BDI scores) is available ￼.
 • Required Functions:
 • load_labels_from_file(path): Read a CSV or JSON containing subject IDs and their depression scores/levels. For example, a CSV with columns Subject, PHQ9_Score. This function would return a dictionary or pandas DataFrame mapping each subject to a label. It should handle five-class categorization: either the file might directly provide a class 0–4, or a raw score that needs conversion.
 • score_to_severity_class(score): A helper to convert a numerical questionnaire score into one of five severity classes ￼. For example, using PHQ-9 ranges: 0–4 = class 0 (Normal), 5–9 = class 1 (Mild), 10–14 = class 2 (Moderate), 15–19 = class 3 (Moderate-Severe), ≥20 = class 4 (Severe) ￼. This logic should be documented in code so it’s clear and based on clinical conventions. If the dataset defines categories differently, this function can be adjusted.
 • The module can also define a constant DEPRESSION_LEVELS = {0: "Normal", 1: "Mild", 2: "Moderate", 3: "Moderate-Severe", 4: "Severe"} for reference (e.g., for plotting confusion matrix axes).
 • Placeholder Implementation: Until an actual metadata file is provided, we can simulate labels. For instance, create a fake dictionary of subject IDs to classes using random or heuristic assignment (ensure roughly the correct distribution). Clearly mark this as dummy. For example, labels_dict = {sid: np.random.randint(0,5) for sid in subject_ids} or use a fixed pattern (first 10 subjects class0, next 10 class1, etc.) to have all five classes present ￼. Print a warning that real labels should replace this.
 • Data Flow & I/O: In the notebook or dataset initialization, instead of the old binary label assignment, use load_labels_from_file if a file path is configured (e.g., config.LABELS_FILE). If no file is present, fall back to the placeholder. The resulting labels_dict maps each subject (as string or int ID) to an integer 0–4. This dict will be used throughout the LOOCV to assign self.labels_dict in the dataset class. When real data becomes available, the only change needed is pointing to the actual file; the logic to process it is already in place.
 • Dependencies: pandas or Python CSV/JSON libraries for reading label files. NumPy (for placeholder generation or any array ops). This module remains lightweight – mostly reading a file and applying a mapping.
 • Example Usage:

# Suppose a CSV "labels.csv" with columns: SubjectID, PHQ9  

labels = load_labels_from_file("data/labels.csv")  
labels_dict = {subj: score_to_severity_class(score) for subj, score in labels.items()}  

# Use labels_dict in dataset or training loop  

If no file:

labels_dict = {str(i): np.random.randint(0, 28) for i in range(1, NUM_SUBJECTS+1)}  
labels_dict = {subj: score_to_severity_class(fake_score) for subj, fake_score in labels_dict.items()}  
print("WARNING: Using placeholder labels – replace with real data when available!")  

This yields labels 0–4 for each of the 53 subjects, distributed arbitrarily ￼. The mapping function ensures consistency with defined severity ranges.

clarity/training/config.py – Configuration Updates
 • Purpose: Update the central configuration to support new features (multi-class labels, audio modality settings, model selection) while maintaining the structured hyperparameters. This allows easy toggling of options in one place, aligning with the repository’s design of central config ￼.
 • Key Updates:
 • NUM_CLASSES = 5: Change from 2 to 5 to accommodate multi-class classification ￼. All models should reference this for output layer dimensions.
 • DEPRESSION_LEVELS: Add a list or dict defining the five class names or score ranges (as noted above), so that plotting and interpretation use consistent labels.
 • LABELS_FILE = "./data/labels.csv" (or similar): Path to a future labels metadata file. Currently can be None or a placeholder path; used by load_labels_from_file.
 • Audio settings: Add flags/parameters for audio: e.g., INCLUDE_AUDIO = False (global toggle), AUDIO_DIR = "./data/MODMA_audio/" (base path for audio files), AUDIO_SAMPLE_RATE = 16000 (expected sample rate), N_MELS = 128 (spectrogram parameters). These will be used by clarity/data/audio.py.
 • MODEL_OPTIONS = ["cnn", "mha_gcn", "eegnet", "vit", "audio_cnn", "multimodal"]: Expand the list of allowable model keywords for selection. This list is used in notebooks or scripts to validate chosen model. For example, adding "audio_cnn" (audio-only model) and "multimodal" (EEG+audio fusion) as new options.
 • Possibly DEFAULT_MODEL = "cnn" for convenience, and FUSION_METHOD = "late" (could be “early” or “late”) if we want to configure how fusion is done.
 • Ensure existing constants (like channel lists, sampling rate, etc.) remain unchanged. Add comments where needed (e.g., clarify the 29-channel selection rationale as per audit ￼).
 • Data Flow: The config is imported throughout the code (e.g., dataset and model definitions). Changing NUM_CLASSES to 5 will propagate to model initialization (if those use config or pass the value). The INCLUDE_AUDIO flag will be checked in the training loop to decide whether to load audio data and instantiate audio models. MODEL_OPTIONS and such help the notebook logic for model selection.
 • Dependencies: Just the Python standard library; this is a plain config file. (If using dataclasses or similar for config, not necessary here — simple constants suffice.)
 • Example Usage:
In the analysis notebook or training script:

from clarity.training import config  
if config.INCLUDE_AUDIO:  
    print("Audio modality enabled – using audio data from", config.AUDIO_DIR)  
model_choice = "multimodal"  # could come from user input  
assert model_choice in config.MODEL_OPTIONS  
NUM_CLASSES = config.NUM_CLASSES  # e.g., 5  

The config centralizes these options, making it easy to switch to multi-class or multimodal experiments by editing one file (or via a single flag in the notebook).

clarity/training/loop.py – LOOCV Dataset and Training Loop Updates
 • Purpose: Modify the dataset class and training/evaluation functions to handle multi-class labels and optional audio data, while preserving the LOOCV workflow and caching optimizations. The goal is to integrate new modalities seamlessly without breaking the one-subject-out cross-validation logic.
 • Key Updates:
 • Custom Dataset for Audio/Multimodal: Extend the existing CustomEEGDataset to either handle multiple inputs or introduce a new dataset class. One approach is to generalize CustomEEGDataset to accept a modality parameter. For example, model_type could be "eeg" (default EEG-only), "audio" (audio-only), or "multimodal" (both). If model_type == "audio", the dataset will use clarity/data/audio.py functions: for each subject in training, load and preprocess audio, get features (Mel-spectrogram), and append to self.data. If "multimodal", the dataset must load both EEG epochs and audio features. In that case, self.data could store tuples (eeg_features, audio_features) per sample, and __getitem__ returns a tuple along with the label.
 • Caching Integration: Extend the caching logic (if implemented as in Module 2) to include audio data ￼. For example, caching functions can use a key that includes modality or model type. If model_type="audio", cache the preprocessed audio features for each subject (e.g., the computed spectrograms) so that subsequent folds don’t reload from scratch. For "multimodal", cache could store a tuple of (eeg_epochs, audio_features) per subject. The caching module (clarity/data/caching.py) can be updated accordingly (e.g., filename subject_<ID>_audio.pkl for audio-only, and subject_<ID>_multimodal.pkl for combined). This ensures we avoid redundant computation in LOOCV, maintaining efficiency ￼.
 • Label Handling: Use labels_dict from clarity/data/metadata.py instead of the hardcoded binary logic. The dataset initialization will be passed a labels_dict mapping each subject to 0–4. Ensure that in __getitem__, the label returned corresponds to those classes. The training loop and loss function should naturally handle the multi-class labels (with CrossEntropyLoss using 5 classes). Metrics calculation (accuracy, etc.) should be updated to multi-class: e.g., use sklearn.metrics.classification_report or specify averaging=‘macro’ for precision/recall if using scikit.
 • Train/Evaluate Functions: The train_model loop likely remains similar, but evaluate_model should now aggregate predictions and true labels for multi-class confusion matrix output ￼. Modify evaluate_model to return lists of all predictions and labels in addition to metrics ￼. For example, return (acc, prec, rec, f1), (all_preds, all_labels). This does not break existing code but extends it for analysis.
 • LOOCV Main Loop Adjustments: In the main training loop (in the notebook or if encapsulated here), incorporate model selection and modality logic. Pseudo-code for one fold:

if MODEL_TO_RUN == 'audio_cnn':  
    train_dataset = CustomEEGDataset(train_ids, labels_dict, model_type='audio')  
    test_dataset  = CustomEEGDataset([test_id], labels_dict, model_type='audio')  
    model = BaselineAudioCNN(num_classes=NUM_CLASSES)  
elif MODEL_TO_RUN == 'multimodal':  
    train_dataset = CustomMultimodalDataset(train_ids, labels_dict)  # or CustomEEGDataset with internal switch  
    test_dataset  = CustomMultimodalDataset([test_id], labels_dict)  
    model = MultimodalFusionModel(eeg_model=BaselineCNN(), audio_model=BaselineAudioCNN(), fusion='average')  
else:  # EEG-only  
    train_dataset = CustomEEGDataset(train_ids, labels_dict, model_type='eeg')  
    test_dataset  = CustomEEGDataset([test_id], labels_dict, model_type='eeg')  
    model = ... (BaselineCNN or MHA_GCN as per selection)  

This ensures we initialize the correct dataset and model depending on the experiment. Both dataset and model are constructed in a modular way, so unused modalities don’t interfere when not needed. The rest of the training loop (optimizer, loss, epochs iteration) remains unchanged and should work as before.

 • Data Flow & I/O: During LOOCV, for each fold we load training data for 52 subjects and test data for 1 subject. With caching, each subject’s processed data (EEG epochs, audio features) will be loaded from cache if available, drastically reducing overhead on repeated folds ￼ ￼. The dataset yields either:
 • EEG-only: (eeg_epoch_tensor, label)
 • Audio-only: (spectrogram_tensor, label)
 • Multimodal: ((eeg_tensor, spectrogram_tensor), label)
The DataLoader collates these accordingly. The model forward pass needs to handle one or two inputs. After training, predictions from evaluate_model (for that single test subject’s samples) are collected.
 • Dependencies: No new libraries beyond what’s already used. Ensure torch_geometric integration (if planned for GCN) still works – if using PyG Data objects, the dataset may need separate handling, but that is orthogonal to audio integration (focus on making audio path work in parallel with existing code). Continue using tqdm for progress, sklearn.metrics for performance metrics.
 • Example Usage:
The LOOCV execution is typically triggered in the notebook. With these changes, one could do:

MODEL_TO_RUN = 'multimodal'  
config.INCLUDE_AUDIO = True  

# ...inside LOOCV loop

train_data = CustomEEGDataset(train_ids, labels_dict, model_type='multimodal')  
test_data = CustomEEGDataset([held_out_id], labels_dict, model_type='multimodal')  
model = MultimodalFusionModel(BaselineCNN(), BaselineAudioCNN(), fusion_method='average')  

# train model as usual  

This would train a combined model using both EEG and audio (with placeholders in current state). For EEG-only, one would set MODEL_TO_RUN = 'cnn' (and INCLUDE_AUDIO = False), and the code would only use EEG paths. The design ensures the same training loop supports all cases.

Results Logging and Analysis Enhancements
 • Purpose: Improve the end-of-experiment reporting for better insight and scientific rigor ￼ ￼. This includes logging per-subject outcomes, visualizing a confusion matrix (especially important for multi-class), and conducting basic statistical tests to compare models’ performance ￼. These additions make results more interpretable and trustworthy.
 • Components:
 • Per-Subject Prediction Log: During each fold’s evaluation, record the predicted labels for each test sample (each window or aggregate per subject) and the true label. Since in LOOCV the test subject’s label is uniform for all its windows, we can simplify to logging one prediction per subject (e.g., majority vote or average probability across that subject’s windows). Collect these in a list or dict. After LOOCV, output a CSV or printed table of each subject ID, true class, and predicted class. This helps identify which subjects were misclassified. (This logging can be done in the notebook after the loop completes, using data stored from evaluate_model outputs.)
 • Confusion Matrix: Using the aggregated all_fold_labels and all_fold_preds from all folds, compute a confusion matrix ￼. Leverage sklearn.metrics.confusion_matrix(y_true, y_pred) to get a 5x5 matrix (for multi-class; 2x2 if binary). Plot it with seaborn heatmap for clarity, labeling axes with class names from DEPRESSION_LEVELS ￼. This visualization immediately shows how often each class is mistaken for another (e.g., Moderate vs Moderate-Severe may be frequently confused). Include this plot in the notebook’s results section.
 • Statistical Significance Test: If comparing multiple models (say, baseline vs advanced), perform a paired t-test on their fold accuracies ￼. For example, run LOOCV with model A and model B, collect accuracy for each fold as two lists, then use scipy.stats.ttest_rel(listA, listB). Print the t-statistic and p-value ￼. If p < 0.05, note that the difference is statistically significant (meaning one model is likely truly better) ￼. This adds scientific rigor by not just reporting metrics, but validating differences aren’t due to chance.
 • Interpretability Visuals (optional): To emphasize scientific insight, provide optional code to visualize model internals. For instance, if using GCN, average the learned adjacency matrices or attention weights across depressed vs control subjects and display them (e.g., as images or topographical maps) ￼. Similarly, for EEG, one could plot the average differential entropy features per class. These help in understanding why the model makes certain predictions, aligning with the explainability focus in medical AI ￼. These should be clearly separated (perhaps under an “Interpretation” section) and only run if the user is interested, as they are exploratory.
 • Data Flow: The confusion matrix and t-test are computed after the LOOCV loop completes. The notebook will likely run through LOOCV for one model at a time; for comparison, the plan could be to loop through a list of models. Example:

models_to_compare = ['cnn', 'mha_gcn']  
results = {}  
for m in models_to_compare:  
    MODEL_TO_RUN = m  
    # run LOOCV loop, collect fold accuracies into results[m]['acc'] list  

# After, do ttest_rel(results['cnn']['acc'], results['mha_gcn']['acc'])  

This structure ensures we reuse the same pipeline for both models and then analyze collectively. Logging each fold’s metrics into a list is straightforward since we already print them; just append to a list as well. Confusion matrix uses all predictions across folds (for multi-class, consider normalizing or showing percentages too).

 • Dependencies: sklearn.metrics for confusion_matrix (and possibly classification_report for a quick text summary), seaborn for heatmap, matplotlib for plotting; scipy.stats for the t-test. These are already in the environment. Ensure to import them in the analysis section. No new dependencies needed.
 • Example Usage: After running the LOOCV:

# Assuming all_fold_labels and all_fold_preds are collected  

cm = confusion_matrix(all_fold_labels, all_fold_preds)  
plt.figure(figsize=(6,5))  
sns.heatmap(cm, annot=True, cmap="Blues",
            xticklabels=DEPRESSION_LEVELS.values(),
            yticklabels=DEPRESSION_LEVELS.values())  
plt.title("LOOCV Confusion Matrix"); plt.xlabel("Predicted"); plt.ylabel("True")  
plt.show()  

And for model comparison:

from scipy.stats import ttest_rel  
t_stat, p_val = ttest_rel(results['cnn']['acc'], results['mha_gcn']['acc'])  
print(f"Paired t-test CNN vs GCN: t={t_stat:.2f}, p={p_val:.3f}")  

which would output something like Paired t-test CNN vs GCN: t=-2.13, p=0.037 and then we can conclude significance if p<0.05. All such analyses should be clearly explained with Markdown text in the notebook (for human readability), and code commented for clarity to both humans and AI agents reviewing it.

Ensuring Rigor and Readability: Throughout the implementation, we will maintain high code quality and documentation. Each new function or class will include docstrings explaining its purpose and usage. Type hints will be used for function signatures (e.g., def load_audio(subject_id: str) -> np.ndarray:) to aid static analysis and AI agents in understanding the code structure. We’ll add comments, especially for any placeholder logic, clearly marking them as TODO or PLACEHOLDER for future data insertion. By following the repository’s modular style ￼ and configuration-driven design, the final sandbox will be extensible (easy to add or swap components), scientifically rigorous (faithful to recommended practices and evaluation methods), and understandable by both experienced researchers and automated tools. This positions the sandbox to fully leverage the MODMA EEG (and audio) dataset for comprehensive experimentation, even before all real-world data (like audio files or multi-class labels) are in hand.

Sources: Key design choices are informed by the repository audit and design docs – e.g., MODMA’s multi-class depression severity labels ￼, the suggestion to add an audio modality pipeline ￼, and recommendations for caching ￼ and richer evaluation metrics ￼ ￼ – ensuring our implementation plan aligns with cutting-edge research practices.
