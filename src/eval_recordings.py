"""Real-world test: score all three models on our own recordings.

Put recordings in data/my_recordings/ (a test set: never train on these; the
Sound Lab's data/user_samples/ is for training clips). Name each file by its
ESC-50 class, optionally followed by a number:
    door_wood_knock_01.wav   keyboard_typing_2.m4a   clapping.wav
Accepted formats: wav, mp3, flac, ogg, m4a. Files whose name does not start with
a class name are skipped, with a warning.

Writes runs/recordings_predictions.csv (per file: true class, each model's top-1
and top-3) and prints accuracy next to the official test-fold accuracy.

Usage (from the repo root): python src/eval_recordings.py [--dir data/my_recordings]
"""
import argparse
from pathlib import Path
import librosa
import numpy as np
import pandas as pd
from demo import Models

EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
MODELS = ("CNN", "CLAP probe", "CLAP zero-shot")
TEST_COLS = {"CNN": "cnn_mixup_s1", "CLAP probe": "clap_probe", "CLAP zero-shot": "clap_zeroshot"}


def label_from_name(stem, classes):
    """Longest class name that the file name starts with (so 'door_wood_knock_01' -> door_wood_knock)."""
    stem = stem.lower().replace("-", "_").replace(" ", "_")
    hits = [c for c in classes if stem == c or stem.startswith(c + "_") or
            (stem.startswith(c) and stem[len(c):].isdigit())]
    return max(hits, key=len) if hits else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=Path("data/my_recordings"))
    args = ap.parse_args()
    files = sorted(p for p in args.dir.glob("*") if p.suffix.lower() in EXTS) if args.dir.exists() else []
    if not files:
        print(f"No recordings found in {args.dir}/ - add files like door_wood_knock_01.wav first.")
        return

    m = Models()
    classes = m.classes
    rows = []
    for path in files:
        true = label_from_name(path.stem, classes)
        if true is None:
            print(f"skipped {path.name}: name does not start with an ESC-50 class")
            continue
        wave, sr = librosa.load(path, sr=None, mono=True)   # same raw input the demo receives
        outs = m.predict(wave, sr)[:3]
        row = {"file": path.name, "true": true, "seconds": round(len(wave) / sr, 1)}
        for name, probs in zip(MODELS, outs):
            ranked = sorted(probs, key=probs.get, reverse=True)
            row[f"{name} top1"] = ranked[0].replace(" ", "_")
            row[f"{name} conf"] = round(probs[ranked[0]], 3)
            row[f"{name} top3"] = "; ".join(r.replace(" ", "_") for r in ranked[:3])
        rows.append(row)
        print(f"{path.name:<32} true {true:<18} " +
              "  ".join(f"{n}: {row[n + ' top1']:<16}" for n in MODELS))
    if not rows:
        return

    df = pd.DataFrame(rows)
    df.to_csv("runs/recordings_predictions.csv", index=False)
    test = pd.read_csv("runs/test_predictions.csv") if Path("runs/test_predictions.csv").exists() else None
    print(f"\n{len(df)} recordings, {df.true.nunique()} classes")
    print(f"{'model':<16}{'own top-1':>11}{'own top-3':>11}{'official test':>15}")
    for n in MODELS:
        top1 = (df[f"{n} top1"] == df.true).mean()
        top3 = np.mean([t in s.split("; ") for t, s in zip(df.true, df[f"{n} top3"])])
        off = f"{(test[TEST_COLS[n]] == test.true).mean():.3f}" if test is not None else "-"
        print(f"{n:<16}{top1:>11.3f}{top3:>11.3f}{off:>15}")
    print("\nSaved runs/recordings_predictions.csv")


if __name__ == "__main__":
    main()
