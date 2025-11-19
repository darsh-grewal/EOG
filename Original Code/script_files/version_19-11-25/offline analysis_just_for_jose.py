import pandas as pd
import numpy as np
from signal_processing_2 import bandpass_filter, notch_filter, process_signal, wavelet_transform
from signal_processing_wavelet import process_eog_signals_with_blinks
from config import FS, center_pos, WIDTH, HEIGHT, FILTER_ORDER, HIGHCUT, LOWCUT
import scipy.signal as sig
import matplotlib.pyplot as plt

file_path = "C:/Users/gruit/OneDrive - The University of Melbourne/EOG/EOG/EOG/Original Code/results/20251113_113813/main_task_raw_signals.csv"
raw_multichannels = pd.read_csv(file_path)

MIN_SIGNAL_LENGTH = 50

def normalize_signal(signal):
    return (signal - np.mean(signal)) / np.std(signal)


calibration_sequence = [
        ("center", center_pos),
        ("left", [int(0.05 * WIDTH), HEIGHT // 2]),
        ("center", center_pos),
        ("right", [int(0.95 * WIDTH), HEIGHT // 2]),
        ("center", center_pos),
        ("up", [WIDTH // 2, int(0.05 * HEIGHT)]),
        ("center", center_pos),
        ("down", [WIDTH // 2, int(0.95 * HEIGHT)]),
        ("center", center_pos),
        ("left", [int(0.05 * WIDTH), HEIGHT // 2]),
        ("center", center_pos),
        ("right", [int(0.95 * WIDTH), HEIGHT // 2]),
        ("center", center_pos),
        ("up", [WIDTH // 2, int(0.05 * HEIGHT)]),
        ("center", center_pos),
        ("down", [WIDTH // 2, int(0.95 * HEIGHT)]),
        ("center", center_pos),
        ("left", [int(0.05 * WIDTH), HEIGHT // 2]),
        ("center", center_pos),
        ("right", [int(0.95 * WIDTH), HEIGHT // 2]),
        ("center", center_pos),
        ("up", [WIDTH // 2, int(0.05 * HEIGHT)]),
        ("center", center_pos),
        ("down", [WIDTH // 2, int(0.95 * HEIGHT)]),
        ("center", center_pos),
        ("left", [int(0.05 * WIDTH), HEIGHT // 2]),
        ("center", center_pos),
        ("right", [int(0.95 * WIDTH), HEIGHT // 2]),
        ("center", center_pos),
        ("up", [WIDTH // 2, int(0.05 * HEIGHT)]),
        ("center", center_pos),
        ("down", [WIDTH // 2, int(0.95 * HEIGHT)]),
    ]

t = raw_multichannels['time']
ch1 = raw_multichannels['ch1']
ch2 = raw_multichannels['ch2']
ch3 = raw_multichannels['ch3']
ch8 = raw_multichannels['ch8']

def butter_bandpass_sos(lowcut, highcut, fs, order=2):
    """
    Design a bandpass filter using second-order sections (SOS) format.

    Args:
        lowcut: Low cutoff frequency of the filter
        highcut: High cutoff frequency of the filter
        fs: Sampling frequency
        order: Filter order (must be even for bandpass)

    Returns:
        sos: Second-order sections representation of the filter
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    return sig.butter(order, [low, high], btype='band', output='sos')

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    """
    Apply a bandpass filter to the data using second-order sections (SOS) format.

    Args:
        data: Input signal to be filtered
        lowcut: Low cutoff frequency of the filter
        highcut: High cutoff frequency of the filter
        fs: Sampling frequency
        order: Filter order

    Returns:
        Filtered data
    """
    # Check if data is long enough for filtering
    if len(data) < MIN_SIGNAL_LENGTH:
        print(f"Warning: Signal too short for bandpass filtering ({len(data)} < {MIN_SIGNAL_LENGTH} samples)")
        return data  # Return unfiltered data

    try:
        # Get the SOS representation of the filter
        sos = butter_bandpass_sos(lowcut, highcut, fs, order)

        # Apply the filter using sosfilt (zero-phase filtering)
        # sosfiltfilt applies the filter twice (forward and backward) for zero-phase filtering
        if hasattr(sig, 'sosfiltfilt'):
            # Use sosfiltfilt if available (preferred for zero-phase filtering)
            return sig.sosfilt(sos, data)
        else:
            # Fallback to sosfilt if sosfiltfilt is not available
            # Apply forward and backward for zero-phase filtering
            filtered = sig.sosfilt(sos, data)
            filtered = sig.sosfilt(sos, filtered[::-1])[::-1]
            return filtered
    except Exception as e:
        print(f"Error in bandpass filtering: {e}")
        return data  # Return unfiltered data if filtering fails


def process_signal_no_bandpass(data, fs, channel_type):
    """Apply notch, bandpass filters, and wavelet transform to signal data"""
    # Check if data is empty or too short
    if len(data) == 0:
        return data

    if len(data) < MIN_SIGNAL_LENGTH:
        print(f"Warning: Signal too short for full processing ({len(data)} < {MIN_SIGNAL_LENGTH} samples)")
        return data

    # First apply notch and median filter to remove power line interference
    try:
        data = notch_filter(data, fs)
        data = sig.medfilt(data, kernel_size=3)
    except Exception as e:
        print(f"Error in notch filter: {e}")

    # Then apply bandpass filter
    try:
        data = bandpass_filter(data, LOWCUT, HIGHCUT, fs, FILTER_ORDER)
    except Exception as e:
        print(f"Error in bandpass filter: {e}")

    # Apply wavelet transform
    try:
        data = wavelet_transform(data, channel_type)
    except Exception as e:
        print(f"Error in wavelet transform: {e}")

    return data

plt.figure(figsize=(12, 8))
plt.subplot(4, 1, 1)
plt.plot(t, ch1, label='Ch1')
plt.title('Channel 1 Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()

plt.subplot(4, 1, 2)
plt.plot(t, ch2, label='Ch2', color='orange')
plt.title('Channel 2 Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()

plt.subplot(4, 1, 3)
plt.plot(t, ch3, label='Ch3', color='green')
plt.title('Channel 3 Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()

plt.subplot(4, 1, 4)
plt.plot(t, ch8, label='Ch8', color='red')
plt.title('Channel 8 Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()
plt.tight_layout()
plt.show()

# Process EOG signals
h_filt, v_filt, v_denoised, blink_events = process_eog_signals_with_blinks(ch1, ch2, ch3, ch8)
h_raw = ch1 - ch3
v_raw = ch8 - ch2
ch1_processed = process_signal(ch1, FS, "ch1")
ch2_processed = process_signal(ch2, FS, "ch2")
ch3_processed = process_signal(ch3, FS, "ch3")
ch8_processed = process_signal(ch8, FS, "ch8")

ch1_no_band = process_signal_no_bandpass(ch1, FS, "ch1")
ch2_no_band = process_signal_no_bandpass(ch2, FS, "ch2")
ch3_no_band = process_signal_no_bandpass(ch3, FS, "ch3")
ch8_no_band = process_signal_no_bandpass(ch8, FS, "ch8")

h_processed = ch1_processed - ch3_processed
v_processed = ch8_processed - ch2_processed

h_no_band = ch1_no_band - ch3_no_band
v_no_band = ch8_no_band - ch2_no_band

# Ensure same length for plotting
min_len = min(len(t), len(h_filt))
t = t[1000:min_len]
h_raw = h_raw[1000:min_len]
v_raw = v_raw[1000:min_len]
h_filt = h_filt[1000:min_len]
v_filt = v_filt[1000:min_len]
v_denoised = v_denoised[1000:min_len]
h_processed = h_processed[1000:min_len]
v_processed = v_processed[1000:min_len]
h_no_band = h_no_band[1000:min_len]
v_no_band = v_no_band[1000:min_len]

h_filt = normalize_signal(h_filt)
v_filt = normalize_signal(v_filt)
v_denoised = normalize_signal(v_denoised)

plt.figure(figsize=(12, 6))
plt.subplot(3, 1, 1)
plt.plot(t, h_filt, label='Horizontal EOG', color='blue')
#plt.plot(t, h_processed, label='Filtered H', color='red', alpha=0.5)
#plt.plot(t, h_no_band, label='H no bandpass', color='green', alpha=0.3)
plt.title('Filtered Horizontal EOG Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(t, v_filt, label='Vertical EOG', color='green')
#plt.plot(t, v_processed, label='Filtered V', color='red', alpha=0.5)
#plt.plot(t, v_no_band, label='V no bandpass', color='green', alpha=0.3)
plt.title('Filtered Vertical EOG Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(t, v_denoised, label='Denoised Vertical EOG', color='red')
plt.title('Denoised Vertical EOG Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()
plt.tight_layout()
plt.show()