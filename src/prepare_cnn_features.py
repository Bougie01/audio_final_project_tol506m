"""Compute log-mel spectrograms for the CNN once and cache them.
Usage (from the repo root): python src/prepare_cnn_features.py
Needs data/ESC-50-master (run prepare_data.py first). ~2-4 min."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np, pandas as pd, soundfile as sf
from tqdm import tqdm
from cnn_features import prepare, logmel

ROOT = Path("data/ESC-50-master")


def featurise(path):
    y, sr = sf.read(str(path))
    return logmel(prepare(y, sr)).astype(np.float16)


def main():
    meta = pd.read_csv(ROOT / "meta" / "esc50.csv")
    paths = [ROOT / "audio" / f for f in meta.filename]
    with ProcessPoolExecutor() as ex:
        mels = list(tqdm(ex.map(featurise, paths, chunksize=16), total=len(paths), desc="log-mel"))
    names = meta.drop_duplicates("target").sort_values("target").category.tolist()
    Path("cache").mkdir(exist_ok=True)
    np.savez("cache/esc50_cnn_logmel.npz", logmel=np.stack(mels), y=meta.target.values,
             fold=meta.fold.values, filename=meta.filename.values, class_names=np.array(names))
    print("Saved cache/esc50_cnn_logmel.npz", np.stack(mels).shape)


if __name__ == "__main__":
    main()
