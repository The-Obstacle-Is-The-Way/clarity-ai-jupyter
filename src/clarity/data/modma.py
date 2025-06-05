import os

import mne

from src.clarity.training.config import CHANNELS_29, DATA_DIR, OVERLAP, SEED, WINDOW_SIZE


def load_subject_data(subject_id):
    """Loads EEG data for a single subject from the MODMA dataset."""
    file_path = os.path.join(
        DATA_DIR, f"EEG_128channel_resting/sub{subject_id:02d}/rest.set"
    )
    if not os.path.exists(file_path):
        print(
            f"Warning: Data file not found for subject {subject_id} at {file_path}. "
            "Returning None."
        )
        return None
    raw = mne.io.read_raw_eeglab(file_path, preload=True, verbose=False)
    return raw


def preprocess_raw_data(raw):
    """Applies channel selection, filtering, and ICA to the raw MNE object."""
    channel_mapping = {"Fpz": "FPz", "Iz": "I"}
    mapped_channels = []
    for ch in CHANNELS_29:
        if ch in raw.ch_names:
            mapped_channels.append(ch)
        elif ch in channel_mapping and channel_mapping[ch] in raw.ch_names:
            mapped_channels.append(channel_mapping[ch])
    raw.pick_channels(mapped_channels, ordered=True)

    raw.filter(l_freq=1.0, h_freq=40.0, fir_design="firwin", verbose=False)

    n_components_ica = len(raw.ch_names) - 1
    ica = mne.preprocessing.ICA(
        n_components=n_components_ica,
        method="fastica",
        random_state=SEED,
        max_iter="auto",
    )
    ica.fit(raw)

    eog_indices, _ = ica.find_bads_eog(
        raw, ch_name=["Fp1", "Fp2", "Fpz"], threshold=2.5, verbose=False
    )
    if eog_indices:
        ica.exclude = eog_indices
        ica.apply(raw)
    else:
        print(
            "Warning: No EOG components automatically found. "
            "ICA will not remove any components."
        )

    return raw


def segment_data(raw) -> mne.Epochs:
    """Segments preprocessed data into 2s windows with 50% overlap."""
    epochs = mne.make_fixed_length_epochs(
        raw,
        duration=WINDOW_SIZE,
        overlap=WINDOW_SIZE * OVERLAP,
        preload=True,
        verbose=False,
    )
    return epochs
