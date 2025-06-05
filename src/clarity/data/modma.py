import os

import mne
import numpy as np

# Import only what's needed at module level to avoid circular imports

def load_subject_data(subject_id):
    """Loads EEG data for a single subject from the MODMA dataset."""
    # Import here to avoid circular imports
    from ...clarity.training.config import DATA_DIR

    # Convert subject_id to int if it's a string
    subject_id = int(subject_id) if isinstance(subject_id, str) else subject_id

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


def _select_channels(raw):
    """Select appropriate channels from raw data.
    Returns:
        MNE Raw object with selected channels.
    """
    # Import here to avoid circular imports
    from ...clarity.training.config import CHANNELS_29

    channel_mapping = {"Fpz": "FPz", "Iz": "I"}
    mapped_channels = []
    for ch in raw.ch_names:
        mapped_ch = channel_mapping.get(ch, ch)
        if mapped_ch in CHANNELS_29:
            mapped_channels.append(ch)

    if not mapped_channels:
        print("Warning: No common channels found between data and standard montage.")
        return raw

    return raw.pick_channels(mapped_channels)


def _apply_filters(raw):
    """Apply frequency filters to raw data.
    Returns:
        MNE Raw object with filters applied.
    """
    # Apply basic filters
    raw = raw.copy().filter(l_freq=1, h_freq=None)
    raw.events_from_annotations(event_id="auto", chunk_duration=2.0)
    raw.notch_filter(freqs=50, n_jobs=1, verbose=False)  # Remove line noise
    return raw


def _detect_eog_with_channels(raw, ica, eog_channels):
    """Detect eye movement artifacts using EOG channels.
    Returns:
        List of ICA components indices related to EOG.
    """
    try:
        eog_indices = ica.find_bads_eog(
            raw, ch_name=eog_channels, threshold=3.0, verbose=False
        )[0]
        if not eog_indices:
            print("Warning: No EOG components found using available channels.")
        return eog_indices
    except Exception as e:
        print("Warning: Could not detect EOG artifacts: {}".format(str(e)))
        return []


def _detect_eog_automatically(raw, ica):
    """Automatically detect eye movement artifacts without EOG channels.
    Returns:
        List of ICA components indices likely related to EOG.
    """
    try:
        # Try ECG-based detection (works for some eye movements too)
        eog_indices = ica.find_bads_ecg(
            raw, method='correlation', threshold='auto', verbose=False
        )[0]

        if not eog_indices:
            # Topography-based approach as fallback
            components = ica.get_components()[:8]
            eog_indices = []
            for idx, component in enumerate(components):
                front_mean = np.abs(component[:2].mean())
                back_mean = np.abs(component[2:].mean())
                if front_mean > back_mean:
                    eog_indices.append(idx)

        if not eog_indices:
            print("Warning: Could not automatically detect EOG components.")

        return eog_indices
    except Exception as e:
        print("Warning: Automatic artifact detection failed: {}".format(str(e)))
        print("Continuing without EOG artifact removal.")
        return []


def preprocess_raw_data(raw):
    """Applies channel selection, filtering, and ICA to the raw MNE object.
    This function orchestrates the preprocessing pipeline by calling more specialized
    functions for each step in the process.
    Args:
        raw: Raw MNE object containing EEG data

    Returns:
        Preprocessed MNE Raw object
    """
    # Import here to avoid circular imports
    # No imports needed here - using helper functions
    # Step 1: Select channels
    raw = _select_channels(raw)
    # Step 2: Apply filters
    raw = _apply_filters(raw)
    # Step 3: Apply ICA for artifact removal (primarily EOG)
    ica = mne.preprocessing.ICA(n_components=20, random_state=97)  # SEED, verbose=False
    ica.fit(raw, verbose=False)
    # Step 4: Detect and remove artifacts
    # First check if we have EOG channels in the data
    # Check for EOG channels in data
    eog_channels = []
    for ch in raw.ch_names:
        if 'EOG' in ch or 'eog' in ch:
            eog_channels.append(ch)
    if eog_channels:
        # If we have EOG channels, use them to find related ICA components
        eog_indices = _detect_eog_with_channels(raw, ica, eog_channels)
    else:
        # If no EOG channels are available, try to detect artifacts automatically
        eog_indices = _detect_eog_automatically(raw, ica)
    # Apply ICA correction if components were found
    if eog_indices:
        ica.exclude = eog_indices
        ica.apply(raw)
    # Log what happened for diagnostic purposes
    has_excluded = hasattr(ica, 'exclude') and ica.exclude
    excluded_str = str(ica.exclude) if has_excluded else 'None'
    print("ICA excluded components: {}".format(excluded_str))
    return raw


def segment_data(raw) -> list:
    """Segments preprocessed data into 2s windows with 50% overlap.

    Returns:
        list: A list of numpy arrays representing epochs
    """
    # Import here to avoid circular imports
    from ...clarity.training.config import OVERLAP, WINDOW_SIZE

    # Now segment into epochs
    epochs = raw.events_from_annotations(event_id="auto", chunk_duration=2.0)
    epochs = mne.make_fixed_length_epochs(
        raw,
        duration=WINDOW_SIZE,
        overlap=WINDOW_SIZE * OVERLAP,
        preload=True,
        verbose=False,
    )

    # Convert MNE Epochs to a list of numpy arrays as expected by the test
    # Tests expect each epoch to have shape (1, n_channels, n_times)
    # Adding the trial dimension
    # Add a trial dimension (batch size of 1) to each epoch
    data = epochs.get_data()
    epochs_list = []
    for epoch in data:
        # Reshape to (1, n_channels, n_times)
        epochs_list.append(epoch[np.newaxis, :, :])

    return epochs_list
