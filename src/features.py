# src/features.py

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from scipy.signal import welch

FS = 360

def extract_features_single(beat, fs=FS):
    """
    Extract 20 temporal and spectral features from a single beat window.
    Each feature has a documented clinical or signal-processing rationale.
    """
    f = {}

    # ── Temporal: statistical shape descriptors ──────────────────────────
    f['mean']         = np.mean(beat)
    f['std']          = np.std(beat)
    f['skewness']     = skew(beat)        # asymmetry of waveform
    f['kurtosis']     = kurtosis(beat)    # peakedness — high in sharp QRS
    f['max_amp']      = np.max(beat)
    f['min_amp']      = np.min(beat)
    f['peak_to_peak'] = f['max_amp'] - f['min_amp']
    f['rms']          = np.sqrt(np.mean(beat**2))

    # ── Temporal: QRS morphology ─────────────────────────────────────────
    r_idx              = np.argmax(beat)
    f['r_amplitude']   = beat[r_idx]
    f['r_position']    = r_idx / len(beat)   # normalized position of R-peak
    f['pre_r_mean']    = np.mean(beat[:r_idx])
    f['post_r_mean']   = np.mean(beat[r_idx:])
    f['qrs_asymmetry'] = f['pre_r_mean'] - f['post_r_mean']
    # PVCs often show asymmetric QRS due to aberrant conduction

    # ── Temporal: energy distribution ────────────────────────────────────
    f['area_pos']   = np.sum(beat[beat > 0])
    f['area_neg']   = np.abs(np.sum(beat[beat < 0]))
    f['area_total'] = np.sum(np.abs(beat))
    f['area_ratio'] = f['area_pos'] / (f['area_neg'] + 1e-8)
    # PVCs tend to have more balanced or inverted area ratio (T-wave discordance)

    # ── Spectral: frequency domain descriptors ────────────────────────────
    freqs, psd       = welch(beat, fs=fs, nperseg=min(64, len(beat)))
    total_power      = np.sum(psd) + 1e-8
    f['spec_centroid']  = np.sum(freqs * psd) / total_power
    f['low_freq_power'] = np.sum(psd[freqs < 10])    # P/T wave energy
    f['high_freq_power']= np.sum(psd[freqs >= 10])   # QRS energy
    f['spectral_ratio'] = f['low_freq_power'] / (f['high_freq_power'] + 1e-8)
    # Wide QRS (PVC) → more low-frequency energy → higher spectral_ratio

    return f


def build_feature_dataframe(beats, labels):
    """
    Apply feature extraction to all beats and return a labeled DataFrame.
    This is the input format expected by the classical ML models in Notebook 04.
    """
    rows = []
    for beat, label in zip(beats, labels):
        row = extract_features_single(beat)
        row['label'] = label
        rows.append(row)
    return pd.DataFrame(rows)


def get_rr_features(beat_samples, beat_labels, idx, fs=FS):
    """
    Extract R-R interval context features for beat at position idx.
    Returns pre-RR, post-RR, and their ratio — captures compensatory pause of PVCs.
    Requires the full list of beat sample positions to compute neighbors.
    """
    if idx == 0 or idx >= len(beat_samples) - 1:
        return {'rr_pre': np.nan, 'rr_post': np.nan, 'rr_ratio': np.nan}
    rr_pre  = (beat_samples[idx]   - beat_samples[idx-1]) / fs * 1000
    rr_post = (beat_samples[idx+1] - beat_samples[idx])   / fs * 1000
    return {
        'rr_pre'  : rr_pre,
        'rr_post' : rr_post,
        'rr_ratio': rr_pre / (rr_post + 1e-8)
    }