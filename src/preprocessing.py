# src/preprocessing.py

import numpy as np
import wfdb
from scipy.signal import butter, filtfilt
import os

FS = 360  # Sampling frequency — constant for all MIT-BIH records

AAMI_MAP = {
    'N':'N','L':'N','R':'N','e':'N','j':'N',
    'A':'S','a':'S','J':'S','S':'S',
    'V':'V','E':'V',
    'F':'F',
    '/':'Q','f':'Q','Q':'Q'
}

def bandpass_filter(signal, lowcut=0.5, highcut=40.0, fs=FS, order=4):
    """
    Apply zero-phase Butterworth bandpass filter.
    Removes baseline wander (< 0.5 Hz) and powerline noise (> 40 Hz).
    """
    nyquist = fs / 2
    b, a = butter(order, [lowcut/nyquist, highcut/nyquist], btype='band')
    return filtfilt(b, a, signal)


def segment_beats(signal, annotation, window_before=90, window_after=110):
    """
    Extract individual beat windows centered on annotated R-peaks.
    Applies per-beat z-score normalization.
    Returns beat matrix (n_beats, window_length) and AAMI label list.
    """
    beats, labels = [], []
    for sample, symbol in zip(annotation.sample, annotation.symbol):
        aami = AAMI_MAP.get(symbol)
        if aami is None:
            continue
        start, end = sample - window_before, sample + window_after
        if start < 0 or end > len(signal):
            continue
        beat = signal[start:end].astype(float)
        std = beat.std()
        if std < 1e-6:
            continue
        beats.append((beat - beat.mean()) / std)
        labels.append(aami)
    return np.array(beats), labels


def load_record(record_path):
    """
    Load a MIT-BIH record and return filtered signal + annotation.
    Always uses Channel 0 (primary lead).
    """
    record     = wfdb.rdrecord(record_path)
    annotation = wfdb.rdann(record_path, 'atr')
    signal_raw = record.p_signal[:, 0]
    signal_filtered = bandpass_filter(signal_raw, fs=record.fs)
    return signal_filtered, annotation, record.sig_name[0]


def process_record_list(record_list, data_path):
    """
    Run the full pipeline on a list of record names.
    Returns X (beat matrix) and y (label array).
    """
    all_beats, all_labels = [], []
    for name in record_list:
        try:
            sig, ann, _ = load_record(data_path + name)
            beats, labels = segment_beats(sig, ann)
            if len(beats) > 0:
                all_beats.append(beats)
                all_labels.extend(labels)
        except Exception as e:
            print(f"Warning: could not process record {name}: {e}")
    return np.vstack(all_beats), np.array(all_labels)

def get_valid_records(data_path, required_exts=('.hea', '.dat', '.atr')):
    """
    Scans data_path and returns the list of record names that have
    ALL required files (.hea, .dat, .atr).
    Also reports incomplete/discarded records.
    """
    all_files = os.listdir(data_path)

    # Base names of all .hea files found
    candidates = sorted(
        {os.path.splitext(f)[0] for f in all_files if f.endswith('.hea')},
        key=lambda x: int(x) if x.isdigit() else x
    )

    valid_records = []
    incomplete = {}

    for rec in candidates:
        missing = [
            ext for ext in required_exts
            if not os.path.exists(os.path.join(data_path, rec + ext))
        ]
        if missing:
            incomplete[rec] = missing
        else:
            valid_records.append(rec)

    return valid_records, incomplete

def get_beat(signal, r_sample, before=90, after=110):
    """Extract a normalized beat window centered on R-peak."""
    start, end = r_sample - before, r_sample + after
    if start < 0 or end > len(signal):
        return None
    beat = signal[start:end].astype(float)
    std = beat.std()
    if std < 1e-6:
        return None
    return (beat - beat.mean()) / std
