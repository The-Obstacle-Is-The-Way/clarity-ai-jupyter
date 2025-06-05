import numpy as np
import pywt
import scipy.signal
from skimage.transform import resize

# Import constants only when needed within functions to avoid circular imports


def calculate_de_features(epoch_data):
    """Calculate differential entropy features for each frequency band."""
    # Import constants locally to avoid circular imports
    from ..clarity.training.config import FREQ_BANDS, SAMPLING_RATE

    n_channels, n_times = epoch_data.shape
    de_features = np.zeros((n_channels, len(FREQ_BANDS)))

    for band_idx, (_band_name, (low_freq, high_freq)) in enumerate(FREQ_BANDS.items()):
        freqs, psd = scipy.signal.welch(
            epoch_data, fs=SAMPLING_RATE, nperseg=n_times, axis=-1
        )
        band_mask = (freqs >= low_freq) & (freqs <= high_freq)
        psd_in_band = psd[:, band_mask]

        freq_res = freqs[1] - freqs[0] if len(freqs) > 1 else 1
        power_in_band = np.sum(psd_in_band, axis=-1) * freq_res

        de_features[:, band_idx] = 0.5 * np.log(
            2 * np.pi * np.e * power_in_band + 1e-10
        )

    return de_features


def extract_dwt_features(epoch_data_single_channel):
    """Extracts 15 scalar DWT features from a single 2s window for one channel."""
    coeffs = pywt.wavedec(epoch_data_single_channel, "db4", level=5)
    dwt_features = []
    for i in range(1, 6):
        detail_coeffs = coeffs[i]
        dwt_features.append(np.mean(detail_coeffs))
        dwt_features.append(np.std(detail_coeffs))
        dwt_features.append(np.sum(np.square(detail_coeffs)))
    return np.array(dwt_features)


def extract_stft_spectrogram_eeg(epoch_data_all_channels, target_size=(224, 224)):
    """Creates a 3-channel 224x224 spectrogram image from a 2s EEG window."""
    # Import constants locally to avoid circular imports
    from ..clarity.training.config import SAMPLING_RATE

    n_channels, _n_times = epoch_data_all_channels.shape
    nperseg = 32
    noverlap = nperseg // 2
    nfft = 256

    all_channel_spectrograms = []
    for i in range(n_channels):
        _f, _t, Zxx = scipy.signal.stft(
            epoch_data_all_channels[i, :],
            fs=SAMPLING_RATE,
            window="hann",
            nperseg=nperseg,
            noverlap=noverlap,
            nfft=nfft,
        )
        spectrogram = np.abs(Zxx)
        all_channel_spectrograms.append(spectrogram)

    # Cast to float64 to resolve type checking error
    all_spectrograms = np.array(all_channel_spectrograms, dtype=np.float64)
    avg_spectrogram = np.mean(all_spectrograms, axis=0)
    log_spectrogram = np.log(avg_spectrogram + 1e-10)

    resized_spectrogram = resize(
        log_spectrogram, target_size, anti_aliasing=True, mode="reflect"
    )

    min_val, max_val = np.min(resized_spectrogram), np.max(resized_spectrogram)
    if max_val > min_val:
        norm_spectrogram = (resized_spectrogram - min_val) / (max_val - min_val)
    else:
        norm_spectrogram = np.zeros(target_size)

    three_channel_spectrogram = np.stack([norm_spectrogram] * 3, axis=0)
    return three_channel_spectrogram


def compute_adjacency_matrix(epoch_data_all_channels, threshold=0.3):
    """Computes the Pearson correlation-based adjacency matrix with no self-loops.

    Args:
        epoch_data_all_channels: EEG data of shape (num_channels, num_timepoints)
        threshold: Correlation threshold below which connections are set to 0

    Returns:
        adj_matrix: Adjacency matrix with zeros on the diagonal (no self-loops)
    """
    # Compute Pearson correlation between channels
    adj_matrix = np.corrcoef(epoch_data_all_channels)

    # Handle NaN values
    adj_matrix[np.isnan(adj_matrix)] = 0

    # Threshold weak connections
    adj_matrix[np.abs(adj_matrix) < threshold] = 0

    # Set diagonal to zero (no self-loops)
    np.fill_diagonal(adj_matrix, 0)

    return adj_matrix
