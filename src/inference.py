"""Load the trained models once and classify any audio clip with all three.

  - the small CNN trained from scratch (SpecAugment + mixup, best validation seed),
  - CLAP + linear probe (C=10, refit on folds 1-3 at start-up, about 1 s),
  - CLAP zero-shot ("This is a sound of X.").
Each model gets exactly its training preprocessing: cnn_features.prepare/logmel
for the CNN and embed.preprocess_for_clap for CLAP. Used by app.py and
eval_recordings.py, so the app, the scripts and the report all run the same code.

Usage (from the repo root):
  python src/inference.py --selftest   # classify a few held-out test-fold clips, print timings
"""
import argparse, time
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from sklearn.linear_model import LogisticRegression

import cnn_features
from embed import embed_audio, get_device, load_clap, preprocess_for_clap
from train_cnn import SmallCNN

ESC = Path("data/ESC-50-master")
DEFAULT_CNN = Path("models/cnn_specaug_mixup_s1.pt")   # best validation seed
CLAP_CACHE = Path("cache/esc50_clap.npz")
RECORDINGS = Path("data/my_recordings")                 # real-world TEST set: never trained on
MODEL_NAMES = ("CNN", "CLAP probe", "CLAP zero-shot")


def save_recording(sr, wave, category, folder=RECORDINGS):
    """Save raw recorded audio as <category>_<NN>.wav (next free number). Returns the path."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    taken = {p.stem for p in folder.glob(f"{category}_*")}
    n = 1
    while f"{category}_{n:02d}" in taken:
        n += 1
    path = folder / f"{category}_{n:02d}.wav"
    wave = np.asarray(wave)
    if not np.issubdtype(wave.dtype, np.integer):          # float audio -> 16-bit PCM
        wave = (np.clip(wave, -1, 1) * 32767).astype(np.int16)
    sf.write(path, wave, sr, subtype="PCM_16")
    return path


class Models:
    def __init__(self, cnn_path=DEFAULT_CNN):
        ck = torch.load(cnn_path, map_location="cpu")
        self.cnn = SmallCNN(); self.cnn.load_state_dict(ck["state_dict"]); self.cnn.eval()
        self.mean, self.std, self.classes = ck["mean"], ck["std"], ck["classes"]

        d = np.load(CLAP_CACHE, allow_pickle=True)
        assert [str(c) for c in d["class_names"]] == self.classes, "class order mismatch"
        train = np.isin(d["fold"], [1, 2, 3])
        self.probe = LogisticRegression(C=10, max_iter=3000).fit(d["emb"][train], d["y"][train])
        self.text_emb = d["text_emb"]
        self.fold, self.filenames = d["fold"], d["filename"]

        self.device = get_device()
        self.clap, self.extractor, _ = load_clap(self.device)
        self.scale = self.clap.logit_scale_a.detach().exp().item() if hasattr(self.clap, "logit_scale_a") else 100.0

    def _labels(self, probs):
        return {c.replace("_", " "): float(p) for c, p in zip(self.classes, probs)}

    def embed(self, wave, sr):
        """CLAP embedding (1, 512) of a raw waveform, with CLAP's own preprocessing."""
        return embed_audio(self.clap, self.extractor, [preprocess_for_clap(wave, sr)], self.device)

    @torch.no_grad()
    def predict(self, wave, sr):
        """wave: raw samples (any dtype, mono or stereo); sr: its sample rate.
        Returns (cnn, clap_probe, clap_zeroshot) {class: probability} dicts and timings in ms."""
        wave = np.asarray(wave)
        if np.issubdtype(wave.dtype, np.integer):            # microphone input is int16
            wave = wave / np.iinfo(wave.dtype).max
        t = time.perf_counter()
        x = (torch.from_numpy(cnn_features.logmel(cnn_features.prepare(wave, sr))) - self.mean) / self.std
        cnn_p = self.cnn(x[None, None]).softmax(-1)[0].numpy()
        cnn_ms = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        e = self.embed(wave, sr)
        probe_p = self.probe.predict_proba(e)[0]
        s = (e @ self.text_emb.T)[0] * self.scale
        zs_p = np.exp(s - s.max()); zs_p /= zs_p.sum()
        clap_ms = (time.perf_counter() - t) * 1000
        return self._labels(cnn_p), self._labels(probe_p), self._labels(zs_p), cnn_ms, clap_ms

    def test_examples(self, n, seed):
        rng = np.random.default_rng(seed)
        return [ESC / "audio" / f for f in rng.choice(self.filenames[self.fold == 5], n, replace=False)]


def selftest(m):
    top = lambda d: max(d, key=d.get)
    for path in m.test_examples(6, seed=0):
        wave, sr = sf.read(path)
        cnn, probe, zs, cnn_ms, clap_ms = m.predict(wave, sr)
        true = m.classes[int(path.stem.split("-")[-1])].replace("_", " ")
        print(f"true: {true:<18} CNN: {top(cnn):<18} CLAP probe: {top(probe):<18} "
              f"zero-shot: {top(zs):<18} ({cnn_ms:.0f} ms / {clap_ms:.0f} ms)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--cnn", type=Path, default=DEFAULT_CNN)
    args = ap.parse_args()
    if args.selftest:
        selftest(Models(args.cnn))
    else:
        print("This module has no UI. Start the app with: python -m streamlit run app.py")
