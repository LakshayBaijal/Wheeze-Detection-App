"""
Turn a clean audio clip into numbers the computer can learn from.

Two kinds of "features":

  * MFCCs  -> a short list of numbers summarising the sound. Fast, and works
             well with traditional machine-learning models (Random Forest etc.).

  * Log-mel spectrogram -> a picture of the sound (time on one axis, pitch on
             the other, brightness = loudness). This is what we feed the CNN,
             which is good at learning from pictures.
"""

import numpy as np
import librosa

from config import N_MELS, N_MFCC, SAMPLE_RATE


def mfcc_vector(signal):
    """
    Return one fixed-length vector of numbers describing the clip.

    We take MFCCs over time, then summarise each coefficient by its mean and
    standard deviation. That gives a compact, same-sized description of any clip.
    """
    mfcc = librosa.feature.mfcc(y=signal, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
    means = mfcc.mean(axis=1)
    stds = mfcc.std(axis=1)
    return np.concatenate([means, stds]).astype(np.float32)


def mel_spectrogram_image(signal):
    """
    Return a 2-D "image" (log-mel spectrogram) for the CNN.

    Values are scaled to roughly 0..1 so the network trains smoothly.
    """
    mel = librosa.feature.melspectrogram(
        y=signal, sr=SAMPLE_RATE, n_mels=N_MELS
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)  # to decibels
    # Scale from [-80, 0] dB up to [0, 1].
    scaled = (log_mel + 80.0) / 80.0
    return np.clip(scaled, 0.0, 1.0).astype(np.float32)
