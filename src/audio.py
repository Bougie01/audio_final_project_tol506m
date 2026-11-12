"""Shared audio preprocessing. Import this in training AND in the demo,
so live recordings are processed exactly like the training data."""
import numpy as np
import librosa

SR = 16000            # all audio is resampled to 16 kHz mono
DURATION = 5.0        # ESC-50 clips are 5 s
N_SAMPLES = int(SR * DURATION)
N_FFT, HOP, N_MELS, N_MFCC = 1024, 320, 64, 40


def preprocess_waveform(y, sr):
    """Any waveform -> mono, 16 kHz, exactly 5 s, peak-normalised float32."""
    y = np.asarray(y, dtype=np.float32)
    if y.ndim > 1:                       # stereo -> mono
        y = y.mean(axis=1 if y.shape[1] < y.shape[0] else 0)
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    if len(y) < N_SAMPLES:               # pad with silence
        y = np.pad(y, (0, N_SAMPLES - len(y)))
    else:                                # crop
        y = y[:N_SAMPLES]
    return y / (np.abs(y).max() + 1e-9)


def load_file(path):
    y, sr = librosa.load(path, sr=None, mono=False)
    if y.ndim > 1:
        y = y.mean(axis=0)
    return preprocess_waveform(y, sr)


def log_mel(y):
    """(n_mels, frames) log-mel spectrogram in dB."""
    m = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS)
    return librosa.power_to_db(m).astype(np.float32)


def mfcc_stats(y):
    """Mean and std of MFCCs over time -> vector of length 2*N_MFCC."""
    m = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP)
    return np.concatenate([m.mean(axis=1), m.std(axis=1)]).astype(np.float32)
