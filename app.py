"""Sound Lab: one local app for the whole project.

Tabs
  Classify            record (microphone) or upload a clip -> top-5 of the CNN, CLAP probe and CLAP
                      zero-shot side by side; optionally save the recording to our test set
  Add training clips  add labelled clips to data/user_samples/ (training data for the CLAP probe)
  Train & compare     retrain the CLAP probe with/without the added clips; compare on ESC-50 fold 4
                      and on our own recordings
  My recordings       the real-world test set in data/my_recordings/: listen, delete, score all models

Run from the repo root:  python -m streamlit run app.py
"""
from __future__ import annotations

import csv
import hashlib
import io
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)                                   # the src/ modules use repo-relative paths
sys.path.insert(0, str(ROOT / "src"))

# Show the page before the heavy imports: torch + transformers can take a minute or more to import
st.set_page_config(page_title="Sound Lab", page_icon="🎙️", layout="wide")
st.title("Sound Lab")
st.caption("Environmental sound classification on the 50 ESC-50 classes: a small CNN trained from "
           "scratch vs the pretrained CLAP model.")
with st.spinner("Starting up: importing PyTorch and transformers (first start only, can take 1-2 minutes)…"):
    from eval_recordings import list_recordings, score_recordings  # noqa: E402
    from inference import MODEL_NAMES, save_recording  # noqa: E402

DATASET = ROOT / "data" / "ESC-50-master"
CACHE = ROOT / "cache" / "esc50_clap.npz"
SAMPLES = ROOT / "data" / "user_samples"         # training clips
MANIFEST = SAMPLES / "samples.csv"
AUDIO_TYPES = ["wav", "mp3", "flac", "ogg", "m4a"]


# ---------------------------------------------------------------- cached resources
@st.cache_resource(show_spinner="Loading the CNN and CLAP models (first start only, about a minute)…")
def get_models():
    from inference import Models
    return Models()


@st.cache_data(show_spinner=False)
def load_base_data(path: str, modified: float):
    """Cache the small precomputed embedding archive across Streamlit reruns."""
    with np.load(path, allow_pickle=True) as archive:
        return tuple(archive[key].copy() for key in ("emb", "y", "fold", "class_names"))


@st.cache_data(show_spinner="Embedding your recordings with CLAP…")
def embed_recordings(_models, files: tuple[tuple[str, float], ...]) -> np.ndarray:
    """CLAP embeddings of the test recordings (cache key: paths + modification times)."""
    import librosa
    return np.concatenate([_models.embed(*librosa.load(p, sr=None, mono=True)) for p, _ in files])


# ---------------------------------------------------------------- helpers
def load_samples(names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    if not MANIFEST.exists():
        return np.empty((0, 512), dtype=np.float32), np.empty(0, dtype=np.int64)
    manifest = pd.read_csv(MANIFEST)
    manifest = manifest[manifest.category.isin(names)]
    if manifest.empty:
        return np.empty((0, 512), dtype=np.float32), np.empty(0, dtype=np.int64)
    return np.stack(manifest.embedding.apply(lambda x: np.fromstring(x, sep=" "))), manifest.target.to_numpy()


def score(model, X, y):
    pred = model.predict(X)
    return float(accuracy_score(y, pred)), float(f1_score(y, pred, average="macro", zero_division=0))


def decode(data: bytes) -> tuple[np.ndarray, int]:
    """Audio bytes -> (mono float waveform, sample rate). Microphone input is WAV."""
    import soundfile as sf
    try:
        wave, sr = sf.read(io.BytesIO(data), always_2d=False)
        if wave.ndim > 1:
            wave = wave.mean(axis=1)
        return wave.astype(np.float32), sr
    except Exception:                                    # mp3 / m4a etc.
        import librosa
        wave, sr = librosa.load(io.BytesIO(data), sr=None, mono=True)
        return wave, sr


def pretty(name: str) -> str:
    return name.replace("_", " ")


def show_top5(title: str, probs: dict[str, float]):
    top = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)[:5]
    st.markdown(f"**{title}**")
    st.metric("Top guess", top[0][0])
    st.caption(f"{top[0][1]:.0%} confident")
    st.dataframe(pd.DataFrame({"Sound": [k for k, _ in top], "Probability": [100 * v for _, v in top]}),
                 hide_index=True, width="stretch",
                 column_config={"Probability": st.column_config.ProgressColumn(
                     "Probability", min_value=0, max_value=100, format="%.0f%%")})


# ---------------------------------------------------------------- page
if not CACHE.exists() or not (DATASET / "meta" / "esc50.csv").exists():
    st.error("Project data or cached CLAP embeddings are missing. Follow the setup steps in README.md first.")
    st.stop()

X, y, fold, class_names = load_base_data(str(CACHE), CACHE.stat().st_mtime)
names = [str(name) for name in class_names]
train_mask, val_mask = np.isin(fold, [1, 2, 3]), fold == 4
saved_X, saved_y = load_samples(names)
models = get_models()

tab_classify, tab_add, tab_train, tab_rec = st.tabs(
    ["🎙️ Classify", "➕ Add training clips", "📊 Train & compare", "🧪 My recordings"])

# ================================================================ 1. Classify
with tab_classify:
    st.write("Record a sound or upload a clip, then click **Classify**. Each model ranks the 50 "
             "categories; up to 5 seconds (the middle) of the clip is used.")
    source = st.segmented_control("Input", ["Microphone", "Upload a file", "Example from the test set"],
                                  default="Microphone", key="source")
    data, origin = None, None
    if source == "Microphone":
        rec = st.audio_input("Record a sound (3-5 seconds is ideal)", key="mic")
        if rec is not None:
            data, origin = rec.getvalue(), "microphone"
    elif source == "Upload a file":
        up = st.file_uploader("Choose an audio file", type=AUDIO_TYPES, key="classify_upload")
        if up is not None:
            data, origin = up.getvalue(), "upload"
            st.audio(data)
    elif source == "Example from the test set":
        examples = models.test_examples(8, seed=1)
        pick = st.selectbox("Held-out test-fold clip", examples, format_func=lambda p: p.name)
        data, origin = pick.read_bytes(), "example"
        st.audio(data)

    if data is not None:
        key = hashlib.md5(data).hexdigest()
        if st.button("Classify", type="primary", key="classify_btn"):
            wave, sr = decode(data)
            with st.spinner("Running all three models…"):
                cnn, probe, zs, cnn_ms, clap_ms = models.predict(wave, sr)
            st.session_state["classify"] = dict(key=key, out=(cnn, probe, zs), secs=len(wave) / sr, sr=sr,
                                                cnn_ms=cnn_ms, clap_ms=clap_ms)
        res = st.session_state.get("classify")
        if res and res["key"] == key:
            cols = st.columns(3)
            labels = ["Small CNN (from scratch, 1.2M params)", "CLAP + linear probe", "CLAP zero-shot (no training)"]
            for col, label, probs in zip(cols, labels, res["out"]):
                with col:
                    show_top5(label, probs)
            st.caption(f"Input {res['secs']:.1f} s at {res['sr']} Hz, used as a 5 s clip. "
                       f"CNN {res['cnn_ms']:.0f} ms, CLAP {res['clap_ms']:.0f} ms.")

        if origin != "example":
            with st.expander("💾 Save this clip to my test set (data/my_recordings/)"):
                st.caption("The test set is only used for evaluation and never trained on. "
                           "Pick what the sound really is.")
                cat = st.selectbox("True class", names, index=None, format_func=pretty,
                                   placeholder="Choose a class…", key="save_class")
                if st.button("Save to test set", disabled=cat is None, key="save_btn"):
                    wave, sr = decode(data)
                    path = save_recording(sr, wave, cat)
                    total = len(list_recordings(names))
                    st.success(f"Saved `{path.as_posix()}` ({len(wave) / sr:.1f} s). "
                               f"{total} recordings in the test set.")
    else:
        st.info("Record, upload or pick an example to start.")

# ================================================================ 2. Add training clips
with tab_add:
    st.write("Clips added here become **training data** for the CLAP probe (see *Train & compare*). "
             "Do not add your test-set recordings here.")
    left, right = st.columns([1.05, 1])
    with left:
        st.subheader("Add a sound bite")
        upload = st.file_uploader("Choose an audio file", type=AUDIO_TYPES, key="train_upload")
        category = st.selectbox("What sound is it?", names, format_func=pretty, key="train_class")
        if upload:
            st.audio(upload.getvalue())
            if st.button("Add to dataset", type="primary", width="stretch"):
                try:
                    wave, sr = decode(upload.getvalue())
                    with st.spinner("Embedding this clip with CLAP…"):
                        emb = models.embed(wave, sr)[0]
                    SAMPLES.mkdir(parents=True, exist_ok=True)
                    safe_name = Path(upload.name).name
                    destination = SAMPLES / f"{int(time.time())}_{safe_name}"
                    destination.write_bytes(upload.getvalue())
                    fields = ["filename", "category", "target", "embedding"]
                    new_row = {"filename": destination.name, "category": category,
                               "target": names.index(category), "embedding": " ".join(map(str, emb.tolist()))}
                    exists = MANIFEST.exists()
                    with MANIFEST.open("a", newline="", encoding="utf-8") as f:
                        writer = csv.DictWriter(f, fieldnames=fields)
                        if not exists:
                            writer.writeheader()
                        writer.writerow(new_row)
                    st.success(f"Added {safe_name} as {pretty(category)}.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Could not add this audio file: {exc}")
        else:
            st.info("Choose a clip and its label. Clips are padded or trimmed to five seconds, "
                    "matching the project preprocessing.")
    with right:
        st.subheader("Dataset")
        a, b, c = st.columns(3)
        a.metric("ESC-50 clips", f"{len(y):,}")
        b.metric("Your added clips", len(saved_y))
        c.metric("Sound categories", len(names))
        if len(saved_y):
            manifest = pd.read_csv(MANIFEST)
            counts = manifest.category.value_counts().rename_axis("Category").reset_index(name="Clips")
            counts["Category"] = counts.Category.str.replace("_", " ")
            st.dataframe(counts, hide_index=True, width="stretch", height=250)
        else:
            st.write("No custom clips yet. Added clips are saved in `data/user_samples/`.")

# ================================================================ 3. Train & compare
with tab_train:
    st.write("Trains a fresh logistic-regression probe on CLAP embeddings of ESC-50 folds 1–3, with and "
             "without your added training clips. It is evaluated on the ESC-50 validation fold (fold 4) "
             "and, if you have any, on your own test recordings. The official test fold is not used.")
    recs = [(p, t) for p, t in list_recordings(names) if t is not None]
    if st.button("Run training", type="primary", disabled=len(saved_y) == 0, width="stretch"):
        with st.spinner("Training probes and evaluating…"):
            baseline = LogisticRegression(C=10, max_iter=3000).fit(X[train_mask], y[train_mask])
            augmented = LogisticRegression(C=10, max_iter=3000).fit(
                np.concatenate([X[train_mask], saved_X]), np.concatenate([y[train_mask], saved_y]))
            rows = []
            for label, model in (("ESC-50 only", baseline), ("ESC-50 + your clips", augmented)):
                row = {"Model": label}
                row["Val accuracy (fold 4)"], row["Val macro-F1"] = score(model, X[val_mask], y[val_mask])
                rows.append(row)
            if recs:
                E = embed_recordings(models, tuple((str(p), p.stat().st_mtime) for p, _ in recs))
                yr = np.array([names.index(t) for _, t in recs])
                for row, model in zip(rows, (baseline, augmented)):
                    row["Own recordings accuracy"] = float(accuracy_score(yr, model.predict(E)))
            st.session_state["results"] = pd.DataFrame(rows)

    if "results" in st.session_state:
        result = st.session_state["results"]
        pct = {c: "{:.1%}" for c in result.columns if "accuracy" in c}
        st.dataframe(result.style.format({**pct, "Val macro-F1": "{:.3f}"}), hide_index=True, width="stretch")
        d1, d2 = st.columns(2)
        d1.metric("Change on ESC-50 fold 4",
                  f"{result.iloc[1]['Val accuracy (fold 4)'] - result.iloc[0]['Val accuracy (fold 4)']:+.1%}")
        if "Own recordings accuracy" in result:
            d2.metric("Change on your recordings",
                      f"{result.iloc[1]['Own recordings accuracy'] - result.iloc[0]['Own recordings accuracy']:+.1%}",
                      help=f"{len(recs)} recordings in data/my_recordings/")
        else:
            d2.caption("Save some test recordings (Classify tab) to also measure the effect on your own microphone.")
        st.download_button("Download results CSV", result.to_csv(index=False), "sound_lab_results.csv", "text/csv")
    elif len(saved_y) == 0:
        st.caption("Add at least one labelled clip under *Add training clips* to enable training.")

# ================================================================ 4. My recordings
with tab_rec:
    st.write("Your real-world **test set** in `data/my_recordings/`. Add to it from the *Classify* tab. "
             "It is only used for evaluation and never trained on.")
    items = list_recordings(names)
    labelled = [(p, t) for p, t in items if t is not None]
    unlabelled = [p.name for p, t in items if t is None]
    a, b = st.columns(2)
    a.metric("Recordings", len(labelled))
    b.metric("Classes covered", len({t for _, t in labelled}))
    if unlabelled:
        st.warning("These files are ignored because their name does not start with an ESC-50 class: "
                   + ", ".join(unlabelled))
    if not labelled:
        st.info("No recordings yet. Record a sound in the *Classify* tab and use *Save to my test set*.")
    else:
        counts = pd.Series([pretty(t) for _, t in labelled]).value_counts()
        st.dataframe(counts.rename_axis("Class").reset_index(name="Recordings"), hide_index=True, height=200)

        play = st.selectbox("Listen to a recording", [p for p, _ in labelled], format_func=lambda p: p.name)
        st.audio(play.read_bytes())
        with st.popover("🗑️ Delete this recording"):
            st.write(f"Permanently delete `{play.name}`? This cannot be undone.")
            if st.button("Yes, delete it", type="primary", key="delete_btn"):
                play.unlink()
                st.session_state.pop("rec_scores", None)
                st.rerun()

        if st.button("Score all three models on my recordings", type="primary", width="stretch"):
            bar = st.progress(0.0, text="Scoring…")
            df, summary, _ = score_recordings(models, progress=lambda f, n: bar.progress(f, text=f"Scoring {n}"))
            bar.empty()
            df.to_csv("runs/recordings_predictions.csv", index=False)
            st.session_state["rec_scores"] = (df, summary)
        if "rec_scores" in st.session_state:
            df, summary = st.session_state["rec_scores"]
            st.subheader("Accuracy")
            st.dataframe(summary.style.format({c: "{:.1%}" for c in summary.columns if c != "Model"}),
                         hide_index=True, width="stretch")
            st.caption("Top-3: the true class is among the model's three best guesses. "
                       "Saved per-file results to runs/recordings_predictions.csv.")
            st.subheader("Per recording")
            view = df[["file", "true"] + [f"{n} top1" for n in MODEL_NAMES]].copy()
            for n in MODEL_NAMES:
                view[f"{n} top1"] = [("✅ " if p == t else "❌ ") + pretty(p)
                                     for p, t in zip(view[f"{n} top1"], view.true)]
            view["true"] = view.true.map(pretty)
            st.dataframe(view, hide_index=True, width="stretch")
