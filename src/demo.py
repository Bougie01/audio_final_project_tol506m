"""Live demo: "What sound is this?"

Record from the microphone (or upload a file) and compare the top-5 predictions of
  - the small CNN trained from scratch (SpecAugment + mixup),
  - CLAP + linear probe (C=10, refit on folds 1-3 at start-up, about 1 s), and
  - CLAP zero-shot ("This is a sound of X.").
Each model gets exactly its training preprocessing: cnn_features.prepare/logmel
for the CNN and embed.preprocess_for_clap for CLAP.

Usage (from the repo root):
  python src/demo.py               # opens the Gradio app (http://127.0.0.1:7860)
  python src/demo.py --selftest    # classify a few held-out test-fold clips, print timings
Needs cache/esc50_clap.npz (extract_embeddings.py) and a CNN checkpoint in models/.
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


class Models:
    def __init__(self, cnn_path=DEFAULT_CNN):
        ck = torch.load(cnn_path, map_location="cpu")
        self.cnn = SmallCNN(); self.cnn.load_state_dict(ck["state_dict"]); self.cnn.eval()
        self.mean, self.std, self.classes = ck["mean"], ck["std"], ck["classes"]

        d = np.load("cache/esc50_clap.npz", allow_pickle=True)
        assert [str(c) for c in d["class_names"]] == self.classes, "class order mismatch"
        train = np.isin(d["fold"], [1, 2, 3])
        self.probe = LogisticRegression(C=10, max_iter=3000).fit(d["emb"][train], d["y"][train])
        self.text_emb = d["text_emb"]
        self.fold, self.filenames = d["fold"], d["filename"]

        self.device = get_device()
        self.clap, self.extractor, _ = load_clap(self.device)
        self.scale = float(self.clap.logit_scale_a.exp()) if hasattr(self.clap, "logit_scale_a") else 100.0

    def _labels(self, probs):
        return {c.replace("_", " "): float(p) for c, p in zip(self.classes, probs)}

    @torch.no_grad()
    def predict(self, wave, sr):
        """wave: raw samples (any dtype, mono or stereo); sr: its sample rate."""
        wave = np.asarray(wave)
        if np.issubdtype(wave.dtype, np.integer):            # microphone input is int16
            wave = wave / np.iinfo(wave.dtype).max
        t = time.perf_counter()
        x = (torch.from_numpy(cnn_features.logmel(cnn_features.prepare(wave, sr))) - self.mean) / self.std
        cnn_p = self.cnn(x[None, None]).softmax(-1)[0].numpy()
        cnn_ms = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        e = embed_audio(self.clap, self.extractor, [preprocess_for_clap(wave, sr)], self.device)
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


def launch(m):
    import gradio as gr

    def fn(rec):
        if rec is None:
            return None, None, None, "Record or upload a sound first."
        sr, wave = rec
        cnn, probe, zs, cnn_ms, clap_ms = m.predict(wave, sr)
        return cnn, probe, zs, (f"Input {len(wave) / sr:.1f} s at {sr} Hz, used as a 5 s clip. "
                                f"CNN {cnn_ms:.0f} ms, CLAP {clap_ms:.0f} ms.")

    gr.Interface(
        fn=fn,
        inputs=gr.Audio(sources=["microphone", "upload"], type="numpy", label="Sound (5 s is used)"),
        outputs=[gr.Label(num_top_classes=5, label="Small CNN (from scratch, 1.2M params)"),
                 gr.Label(num_top_classes=5, label="CLAP + linear probe"),
                 gr.Label(num_top_classes=5, label="CLAP zero-shot"),
                 gr.Textbox(label="Details")],
        examples=[[str(p)] for p in m.test_examples(6, seed=1)],
        title="What sound is this?",
        description="Environmental sound classification on the 50 ESC-50 classes. "
                    "Examples are held-out test-fold clips.",
        flagging_mode="never",
    ).launch()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--cnn", type=Path, default=DEFAULT_CNN)
    args = ap.parse_args()
    models = Models(args.cnn)
    selftest(models) if args.selftest else launch(models)
