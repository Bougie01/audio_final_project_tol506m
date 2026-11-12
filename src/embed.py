"""CLAP (LAION, HTSAT audio encoder) helpers, shared by training and the demo."""
import numpy as np, torch, librosa

CLAP_ID = "laion/clap-htsat-unfused"
CLAP_SR = 48000                       # CLAP expects 48 kHz
CLAP_SAMPLES = CLAP_SR * 5            # our clips are 5 s


def preprocess_for_clap(y, sr):
    """Any waveform -> mono, 48 kHz, exactly 5 s, peak-normalised float32.
    Use this for training clips AND for live recordings in the demo."""
    y = np.asarray(y, dtype=np.float32)
    if y.ndim > 1:
        y = y.mean(axis=1 if y.shape[1] < y.shape[0] else 0)
    if sr != CLAP_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=CLAP_SR)
    y = np.pad(y, (0, max(0, CLAP_SAMPLES - len(y))))[:CLAP_SAMPLES]
    return y / (np.abs(y).max() + 1e-9)


def get_device():
    if torch.cuda.is_available(): return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"


def load_clap(device=None):
    from transformers import ClapModel, ClapFeatureExtractor, AutoTokenizer
    device = device or get_device()
    model = ClapModel.from_pretrained(CLAP_ID).to(device).eval()
    return model, ClapFeatureExtractor.from_pretrained(CLAP_ID), AutoTokenizer.from_pretrained(CLAP_ID)


def _tensor(out):
    # transformers versions differ: tensor or ModelOutput
    return out if torch.is_tensor(out) else out.pooler_output


def _l2(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)


@torch.no_grad()
def embed_audio(model, extractor, waveforms, device=None, batch_size=16):
    """List of preprocessed 48 kHz waveforms -> (N, 512) L2-normalised embeddings."""
    device = device or next(model.parameters()).device
    outs = []
    for i in range(0, len(waveforms), batch_size):
        inp = extractor(list(waveforms[i:i + batch_size]), sampling_rate=CLAP_SR, return_tensors="pt")
        inp = {k: v.to(device) for k, v in inp.items()}
        outs.append(_tensor(model.get_audio_features(**inp)).float().cpu().numpy())
    return _l2(np.concatenate(outs))


@torch.no_grad()
def embed_text(model, tokenizer, texts, device=None):
    device = device or next(model.parameters()).device
    inp = tokenizer(texts, padding=True, return_tensors="pt").to(device)
    return _l2(_tensor(model.get_text_features(**inp)).float().cpu().numpy())
