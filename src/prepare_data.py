"""Download ESC-50, compute features once, cache to disk.
Usage: python src/prepare_data.py [--limit 200]   (--limit = quick test on a subset)"""
import argparse, io, urllib.request, zipfile
from pathlib import Path
import numpy as np, pandas as pd
from tqdm import tqdm
from audio import load_file, log_mel, mfcc_stats

URL = "https://github.com/karolpiczak/ESC-50/archive/master.zip"
DATA = Path("data"); CACHE = Path("cache")


def download():
    root = DATA / "ESC-50-master"
    if root.exists():
        return root
    DATA.mkdir(exist_ok=True)
    print("Downloading ESC-50 (~600 MB)...")
    buf = urllib.request.urlopen(URL).read()
    zipfile.ZipFile(io.BytesIO(buf)).extractall(DATA)
    return root


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    root = download()
    meta = pd.read_csv(root / "meta" / "esc50.csv")
    if args.limit:
        meta = meta.sample(args.limit, random_state=0).reset_index(drop=True)

    mels, mfccs = [], []
    for fn in tqdm(meta.filename, desc="features"):
        y = load_file(root / "audio" / fn)
        mels.append(log_mel(y)); mfccs.append(mfcc_stats(y))

    CACHE.mkdir(exist_ok=True)
    np.savez_compressed(
        CACHE / "esc50_features.npz",
        logmel=np.stack(mels), mfcc=np.stack(mfccs),
        y=meta.target.values, fold=meta.fold.values,
        filename=meta.filename.values, category=meta.category.values)
    print("Saved cache/esc50_features.npz", np.stack(mels).shape)


if __name__ == "__main__":
    main()
