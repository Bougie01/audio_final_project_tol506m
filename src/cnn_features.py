"""Preprocessing for the from-scratch CNN, shared by training and the demo.

The CNN uses its own spectrogram settings (22.05 kHz, 128 mel bands, hop 512),
separate from the 16 kHz features in audio.py used by the MFCC baseline. Every
clip, training or live, goes through prepare() -> logmel() below.
"""
import numpy as np
import librosa

SR = 22050
DURATION = 5.0
N_SAMPLES = int(SR * DURATION)
N_FFT, HOP, N_MELS = 1024, 512, 128     # -> log-mel of shape (128, 216)


def prepare(y, sr):
    """Any waveform -> mono, 22.05 kHz, peak-normalised, exactly 5 s.
    Longer clips are centre-cropped (keeps the middle of a live recording)."""
    y = np.asarray(y)
    if np.issubdtype(y.dtype, np.integer):          # e.g. int16 from a microphone
        y = y / np.iinfo(y.dtype).max
    y = y.astype(np.float32)
    if y.ndim > 1:                                  # stereo -> mono
        y = y.mean(axis=1 if y.shape[1] < y.shape[0] else 0)
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    peak = np.abs(y).max()
    if peak > 1e-6:
        y = y / peak
    if len(y) >= N_SAMPLES:
        start = (len(y) - N_SAMPLES) // 2
        return y[start:start + N_SAMPLES]
    return np.pad(y, (0, N_SAMPLES - len(y)))


def logmel(y):
    """(N_MELS, frames) log-mel spectrogram in dB."""
    m = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS)
    return librosa.power_to_db(m, ref=1.0, top_db=80.0).astype(np.float32)
