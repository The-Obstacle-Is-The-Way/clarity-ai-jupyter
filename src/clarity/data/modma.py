import os

import mne
import numpy as np

# Import only what's needed at module level to avoid circular imports
from ...clarity.training.config import SEED


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


def preprocess_raw_data(raw):
    """Applies channel selection, filtering, and ICA to the raw MNE object."""
    # Import here to avoid circular imports
    from ...clarity.training.config import CHANNELS_29

    channel_mapping = {"Fpz": "FPz", "Iz": "I"}
    mapped_channels = []
    for ch in CHANNELS_29:
        if ch in raw.ch_names:
            mapped_channels.append(ch)
        elif ch in channel_mapping and channel_mapping[ch] in raw.ch_names:
            mapped_channels.append(channel_mapping[ch])
    raw.pick_channels(mapped_channels, ordered=True)

    # Make a copy to ensure we're not modifying the original data
    raw = raw.copy()
    # Apply filtering
    raw.filter(l_freq=1.0, h_freq=40.0, fir_design="firwin", verbose=False)

    n_components_ica = len(raw.ch_names) - 1
    ica = mne.preprocessing.ICA(
        n_components=n_components_ica,
        method="fastica",
        random_state=SEED,
        max_iter="auto",
    )
    ica.fit(raw)

    # Define EOG channels to check for
    EOG_CHANNELS = ["Fp1", "Fp2", "Fpz"]

    # Check which EOG channels are available
    available_eog_channels = [ch for ch in EOG_CHANNELS if ch in raw.ch_names]

    if available_eog_channels:
        try:
            # Use only available EOG channels
            eog_indices, _ = ica.find_bads_eog(
                raw, ch_name=available_eog_channels, threshold=2.5, verbose=False
            )
            if eog_indices:
                ica.exclude = eog_indices
                ica.apply(raw)
            else:
                print("Warning: No EOG components automatically found using available channels.")
        except Exception as e:
            print("Warning: Could not detect EOG artifacts: {}".format(str(e)))
    else:
        # If no EOG channels are available, try to detect artifacts automatically
        try:
            # Use automatic detection without specifying EOG channels
            eog_indices = ica.find_bads_ecg(
                raw, method='correlation', threshold='auto', verbose=False
            )[0]
            if not eog_indices:
                # Try to use ICA components that look like eye movements based on topography
                # Find components likely to be eye movements based on topography
                components = ica.get_components()[:8]
                eog_indices = []
                for idx, component in enumerate(components):
                    front_mean = np.abs(component[:2].mean())
                    back_mean = np.abs(component[2:].mean())
                    if front_mean > back_mean:
                        eog_indices.append(idx)

            if eog_indices:
                ica.exclude = eog_indices
                ica.apply(raw)
            else:
                print("Warning: Could not automatically detect EOG components.")
        except Exception as e:
            print("Warning: Automatic artifact detection failed: {}".format(str(e)))
            print("Continuing without EOG artifact removal.")

    # Log what happened for diagnostic purposes
    # Check if ICA has excluded components
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

    # Create epochs using MNE's fixed length epoch function
    mne_epochs = mne.make_fixed_length_epochs(
        raw,
        duration=WINDOW_SIZE,
        overlap=WINDOW_SIZE * OVERLAP,
        preload=True,
        verbose=False,
    )

    # Convert MNE Epochs to a list of numpy arrays as expected by the test
    # Tests expect each epoch to have shape (1, n_channels, n_times) - adding the trial dimension
    # Add a trial dimension (batch size of 1) to each epoch
    data = mne_epochs.get_data()
    epochs_list = []
    for epoch in data:
        # Reshape to (1, n_channels, n_times)
        epochs_list.append(epoch[np.newaxis, :, :])

    return epochs_list
