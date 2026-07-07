"""
Phase 3 - Clean the sound.

For each labelled event we:
  1. Load just that slice of the recording (from start_ms to end_ms).
  2. Band-pass filter it - keep only the frequencies where lung sounds live
     (~80-2000 Hz), which removes low rumble and high hiss.
  3. Normalise the volume so loud and quiet recordings look comparable.
  4. Cut or pad it to a fixed length (CLIP_SECONDS) so every clip is the same
     size - models need same-sized inputs.

The output is a clean 1-D array of audio samples, ready for feature extraction.
"""

import numpy as np
import librosa
from scipy.signal import butter, sosfilt

from config import (
    BANDPASS_HIGH_HZ,
    BANDPASS_LOW_HZ,
    CLIP_SECONDS,
    SAMPLE_RATE,
)

_CLIP_LEN = int(CLIP_SECONDS * SAMPLE_RATE)


def _bandpass(signal, sr):
    """Keep only frequencies between BANDPASS_LOW_HZ and BANDPASS_HIGH_HZ."""
    nyquist = sr / 2.0
    low = BANDPASS_LOW_HZ / nyquist
    high = min(BANDPASS_HIGH_HZ / nyquist, 0.99)
    sos = butter(4, [low, high], btype="band", output="sos")
    return sosfilt(sos, signal)


def _fix_length(signal):
    """Cut (if too long) or zero-pad (if too short) to the fixed clip length."""
    if len(signal) >= _CLIP_LEN:
        return signal[:_CLIP_LEN]
    padding = _CLIP_LEN - len(signal)
    return np.pad(signal, (0, padding), mode="constant")


def load_clean_clip(wav_path, start_ms, end_ms):
    """Load one event, clean it, and return a fixed-length audio array."""
    offset = start_ms / 1000.0
    duration = (end_ms - start_ms) / 1000.0

    signal, sr = librosa.load(
        wav_path, sr=SAMPLE_RATE, offset=offset, duration=duration
    )
    if len(signal) == 0:
        signal = np.zeros(_CLIP_LEN, dtype=np.float32)

    signal = _bandpass(signal, sr)

    # Normalise: shift to zero mean, scale so the loudest point is ~1.0.
    peak = np.max(np.abs(signal))
    if peak > 0:
        signal = signal / peak

    return _fix_length(signal.astype(np.float32))


def augment_signal(signal, augment_prob=0.5):
    """
    Light data augmentation: small random shifts in time/pitch to help the model
    generalize. Only used during training.
    """
    if np.random.rand() > augment_prob:
        return signal.astype(np.float32)
    aug = np.random.randint(0, 3)
    if aug == 0:
        shift = np.random.randint(-int(0.1 * _CLIP_LEN), int(0.1 * _CLIP_LEN))
        return np.roll(signal, shift).astype(np.float32)
    elif aug == 1:
        noise = np.random.normal(0, 0.02, len(signal))
        return (signal + noise).astype(np.float32)
    else:
        signal = signal * np.random.uniform(0.8, 1.2)
        return np.clip(signal, -1, 1).astype(np.float32)
