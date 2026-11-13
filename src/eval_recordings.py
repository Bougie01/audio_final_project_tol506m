"""Real-world test: score all three models on our own recordings.

Recordings live in data/my_recordings/ (a test set: never train on these; the
app's "Add training clips" tab saves to data/user_samples/ instead). Record them
in the app (Classify tab -> "Save to my test set") or copy files in, named by
ESC-50 class, optionally followed by a number:
    door_wood_knock_01.wav   keyboard_typing_2.m4a   clapping.wav
Accepted formats: wav, mp3, flac, ogg, m4a. Files whose name does not start with
a class name are skipped.

Writes runs/recordings_predictions.csv (per file: true class, each model's top-1,
confidence and top-3) and prints accuracy next to the official test-fold accuracy.
The same function powers the app's "My recordings" tab.

Usage (from the repo root): python src/eval_recordings.py [--dir data/my_recordings]
"""
import argparse
from pathlib import Path
import librosa
import numpy as np
import pandas as pd
from inference import MODEL_NAMES, RECORDINGS

EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
TEST_COLS = {"CNN": "cnn_mixup_s1", "CLAP probe": "clap_probe", "CLAP zero-shot": "clap_zeroshot"}


def label_from_name(stem, classes):
    """Longest class name that the file name starts with (so 'door_wood_knock_01' -> door_wood_knock)."""
    stem = stem.lower().replace("-", "_").replace(" ", "_")
    hits = [c for c in classes if stem == c or stem.startswith(c + "_") or
            (stem.startswith(c) and stem[len(c):].isdigit())]
    return max(hits, key=len) if hits else None


def list_recordings(classes, folder=RECORDINGS):
    """[(path, true_class or None)] for every audio file in the folder."""
    folder = Path(folder)
    files = sorted(p for p in folder.glob("*") if p.suffix.lower() in EXTS) if folder.exists() else []
    return [(p, label_from_name(p.stem, classes)) for p in files]


def score_recordings(models, folder=RECORDINGS, progress=None):
    """Classify every labelled recording with all three models.
    Returns (per-file DataFrame, summary DataFrame, skipped file names)."""
    items = list_recordings(models.classes, folder)
    skipped = [p.name for p, t in items if t is None]
    items = [(p, t) for p, t in items if t is not None]
    rows = []
    for i, (path, true) in enumerate(items):
        wave, sr = librosa.load(path, sr=None, mono=True)    # same raw input the app receives
        outs = models.predict(wave, sr)[:3]
        row = {"file": path.name, "true": true, "seconds": round(len(wave) / sr, 1)}
        for name, probs in zip(MODEL_NAMES, outs):
            ranked = sorted(probs, key=probs.get, reverse=True)
            row[f"{name} top1"] = ranked[0].replace(" ", "_")
            row[f"{name} conf"] = round(probs[ranked[0]], 3)
            row[f"{name} top3"] = "; ".join(r.replace(" ", "_") for r in ranked[:3])
        rows.append(row)
        if progress:
            progress((i + 1) / len(items), path.name)
    df = pd.DataFrame(rows)
    test = pd.read_csv("runs/test_predictions.csv") if Path("runs/test_predictions.csv").exists() else None
    summary = []
    for n in MODEL_NAMES:
        summary.append({
            "Model": n,
            "Own recordings top-1": (df[f"{n} top1"] == df.true).mean() if len(df) else np.nan,
            "Own recordings top-3": (np.mean([t in s.split("; ") for t, s in zip(df.true, df[f"{n} top3"])])
                                     if len(df) else np.nan),
            "Official test (fold 5)": (test[TEST_COLS[n]] == test.true).mean() if test is not None else np.nan,
        })
    return df, pd.DataFrame(summary), skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=RECORDINGS)
    args = ap.parse_args()
    from inference import Models
    m = Models()
    df, summary, skipped = score_recordings(m, args.dir)
    for name in skipped:
        print(f"skipped {name}: name does not start with an ESC-50 class")
    if df.empty:
        print(f"No labelled recordings found in {args.dir}/ - add files like door_wood_knock_01.wav first.")
        return
    for _, r in df.iterrows():
        print(f"{r.file:<32} true {r.true:<18} " + "  ".join(f"{n}: {r[n + ' top1']:<16}" for n in MODEL_NAMES))
    df.to_csv("runs/recordings_predictions.csv", index=False)
    print(f"\n{len(df)} recordings, {df.true.nunique()} classes")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\nSaved runs/recordings_predictions.csv")


if __name__ == "__main__":
    main()
