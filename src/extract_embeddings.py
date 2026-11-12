"""Run frozen CLAP over ESC-50 once and cache the embeddings (the expensive step).
Usage: python src/extract_embeddings.py [--limit 100]
Needs data/ESC-50-master (run prepare_data.py first)."""
import argparse
from pathlib import Path
import numpy as np, pandas as pd, librosa
from tqdm import tqdm
from embed import load_clap, embed_audio, embed_text, preprocess_for_clap, get_device

ROOT = Path("data/ESC-50-master")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    meta = pd.read_csv(ROOT / "meta" / "esc50.csv")
    names = meta.drop_duplicates("target").sort_values("target").category.tolist()  # index = class id
    if args.limit:
        meta = meta.sample(args.limit, random_state=0).reset_index(drop=True)

    device = get_device(); print("device:", device)
    model, extractor, tokenizer = load_clap(device)

    embs = []
    for i in tqdm(range(0, len(meta), args.batch_size), desc="CLAP"):
        wavs = []
        for fn in meta.filename[i:i + args.batch_size]:
            y, sr = librosa.load(ROOT / "audio" / fn, sr=None, mono=True)
            wavs.append(preprocess_for_clap(y, sr))
        embs.append(embed_audio(model, extractor, wavs, device))

    prompts = [f"This is a sound of {n.replace('_', ' ')}." for n in names]
    Path("cache").mkdir(exist_ok=True)
    np.savez_compressed("cache/esc50_clap.npz", emb=np.concatenate(embs), y=meta.target.values,
                        fold=meta.fold.values, filename=meta.filename.values,
                        class_names=np.array(names), text_emb=embed_text(model, tokenizer, prompts, device))
    print("Saved cache/esc50_clap.npz")


if __name__ == "__main__":
    main()
