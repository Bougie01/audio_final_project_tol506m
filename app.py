"""Local browser app for adding labeled audio and retraining the CLAP probe."""
from __future__ import annotations

import csv
import io
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

DATASET = ROOT / "data" / "ESC-50-master"
CACHE = ROOT / "cache" / "esc50_clap.npz"
SAMPLES = ROOT / "data" / "user_samples"
MANIFEST = SAMPLES / "samples.csv"


@st.cache_resource(show_spinner="Loading the CLAP audio model (first use only)…")
def get_clap_runtime():
    """Load the heavyweight encoder once and reuse it for every uploaded clip."""
    from embed import get_device, load_clap

    device = get_device()
    model, extractor, _ = load_clap(device)
    return model, extractor, device


@st.cache_resource(show_spinner="Loading the CNN model…")
def get_cnn_runtime(checkpoint_path: str, modified: float):
    import torch
    from train_cnn import SmallCNN

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = SmallCNN(n_classes=len(checkpoint["classes"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint["mean"], checkpoint["std"], [str(c) for c in checkpoint["classes"]]


@st.cache_data(show_spinner=False)
def load_base_data(path: str, modified: float):
    """Cache the small precomputed embedding archive across Streamlit reruns."""
    with np.load(path, allow_pickle=True) as archive:
        return tuple(archive[key].copy() for key in ("emb", "y", "fold", "class_names"))


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


st.set_page_config(page_title="Sound Lab", page_icon="🎙️", layout="wide")
st.title("Sound Lab")
st.caption("Add labeled sound bites to ESC-50 categories, train a CLAP probe, and compare validation results.")

if not CACHE.exists() or not (DATASET / "meta" / "esc50.csv").exists():
    st.error("Project data or cached CLAP embeddings are missing. Follow the setup steps in README.md first.")
    st.stop()

X, y, fold, class_names = load_base_data(str(CACHE), CACHE.stat().st_mtime)
names = [str(name) for name in class_names]
train_mask, val_mask = np.isin(fold, [1, 2, 3]), fold == 4
saved_X, saved_y = load_samples(names)

left, right = st.columns([1.05, 1])
with left:
    st.subheader("Add a sound bite")
    upload = st.file_uploader("Choose an audio file", type=["wav", "mp3", "flac", "ogg", "m4a"])
    category = st.selectbox("What sound is it?", names, format_func=lambda n: n.replace("_", " "))
    if upload:
        st.audio(upload.getvalue())
        if st.button("Add to dataset", type="primary", width="stretch"):
            try:
                import librosa
                from embed import embed_audio, preprocess_for_clap

                audio, sr = librosa.load(io.BytesIO(upload.getvalue()), sr=None, mono=True)
                wav = preprocess_for_clap(audio, sr)
                with st.spinner("Loading CLAP and embedding this clip…"):
                    model, extractor, device = get_clap_runtime()
                    emb = embed_audio(model, extractor, [wav], device)[0]
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
                st.success(f"Added {safe_name} as {category.replace('_', ' ')}.")
                st.rerun()
            except Exception as exc:
                st.error(f"Could not add this audio file: {exc}")
    else:
        st.info("Choose a clip and its label. Clips are padded or trimmed to five seconds, matching the project preprocessing.")

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

st.divider()
st.subheader("Train and compare")
st.write("The app trains a fresh logistic regression probe on ESC-50 folds 1–3, optionally including your added clips. It evaluates on the untouched fold 4 validation set.")
if st.button("Run training", type="primary", disabled=len(saved_y) == 0, width="stretch"):
    with st.spinner("Training probes and evaluating on validation data…"):
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, f1_score

        baseline = LogisticRegression(C=10, max_iter=3000).fit(X[train_mask], y[train_mask])
        augmented = LogisticRegression(C=10, max_iter=3000).fit(
            np.concatenate([X[train_mask], saved_X]), np.concatenate([y[train_mask], saved_y]))
        base_acc, base_f1 = score(baseline, X[val_mask], y[val_mask])
        add_acc, add_f1 = score(augmented, X[val_mask], y[val_mask])
        st.session_state["results"] = {
            "results": pd.DataFrame([
                {"Model": "ESC-50 baseline", "Validation accuracy": base_acc, "Macro F1": base_f1},
                {"Model": "With your clips", "Validation accuracy": add_acc, "Macro F1": add_f1},
            ]),
            "baseline": baseline, "augmented": augmented,
        }

if "results" in st.session_state:
    result = st.session_state["results"]
    st.dataframe(result["results"].style.format({"Validation accuracy": "{:.1%}", "Macro F1": "{:.3f}"}), hide_index=True, width="stretch")
    base_acc = result["results"].iloc[0]["Validation accuracy"]
    add_acc = result["results"].iloc[1]["Validation accuracy"]
    delta = add_acc - base_acc
    st.metric("Validation accuracy change", f"{delta:+.1%}", help="Change from baseline on ESC-50 fold 4.")
    if delta < 0:
        st.caption("The added examples lowered validation accuracy in this run. More varied clips per category can help the probe learn a useful change.")
    else:
        st.caption("Validation results are for comparison on ESC-50 fold 4. The official test fold remains untouched.")
    st.download_button("Download results CSV", result["results"].to_csv(index=False), "sound_lab_results.csv", "text/csv")
elif len(saved_y) == 0:
    st.caption("Add at least one labeled clip to enable training.")

st.divider()
st.subheader("Test a clip")
st.write("Upload one clip to compare the CNN and CLAP probe. Each model ranks the ESC-50 sound categories by its predicted probability.")
test_file = st.file_uploader("Choose a clip to check", type=["wav", "mp3", "flac", "ogg", "m4a"], key="test_clip")
if test_file:
    st.audio(test_file.getvalue())
    if st.button("Analyze clip", key="analyze_clip", type="primary"):
        try:
            import librosa
            import torch
            from cnn_features import logmel as cnn_logmel, prepare as prepare_for_cnn
            from embed import embed_audio, preprocess_for_clap
            from sklearn.linear_model import LogisticRegression

            sound, sample_rate = librosa.load(io.BytesIO(test_file.getvalue()), sr=None, mono=True)
            cnn_path = ROOT / "models" / "cnn_specaug_mixup_s1.pt"
            if not cnn_path.exists():
                raise FileNotFoundError(f"CNN checkpoint is missing: {cnn_path}")
            with st.spinner("Running both models…"):
                cnn, mean, std, cnn_names = get_cnn_runtime(str(cnn_path), cnn_path.stat().st_mtime)
                if cnn_names != names:
                    raise ValueError("The CNN and CLAP cache use different class orders.")
                cnn_input = (torch.from_numpy(cnn_logmel(prepare_for_cnn(sound, sample_rate))) - mean) / std
                with torch.no_grad():
                    cnn_probs = cnn(cnn_input[None, None]).softmax(-1)[0].numpy()

                clap_model, extractor, device = get_clap_runtime()
                clap_wave = preprocess_for_clap(sound, sample_rate)
                clip_embedding = embed_audio(clap_model, extractor, [clap_wave], device)[0]
                probe = LogisticRegression(C=10, max_iter=3000)
                probe.fit(np.concatenate([X[train_mask], saved_X]), np.concatenate([y[train_mask], saved_y]))
                clap_probs = probe.predict_proba(clip_embedding[None, :])[0]
                clap_names = [names[int(i)] for i in probe.classes_]

            def top_predictions(labels, probabilities):
                order = np.argsort(probabilities)[::-1][:5]
                return pd.DataFrame({
                    "Sound category": [labels[i].replace("_", " ") for i in order],
                    "Probability": [float(probabilities[i]) for i in order],
                })

            st.session_state["clip_predictions"] = {
                "cnn": top_predictions(cnn_names, cnn_probs),
                "clap": top_predictions(clap_names, clap_probs),
            }
        except Exception as exc:
            st.error(f"Could not analyze this audio file: {exc}")

if "clip_predictions" in st.session_state:
    predictions = st.session_state["clip_predictions"]
    cnn_col, clap_col = st.columns(2)
    with cnn_col:
        st.markdown("**CNN**")
        st.metric("Top prediction", predictions["cnn"].iloc[0]["Sound category"],
                  f"{predictions['cnn'].iloc[0]['Probability']:.1%}", delta_color="off")
        st.dataframe(predictions["cnn"].style.format({"Probability": "{:.1%}"}), hide_index=True, width="stretch")
    with clap_col:
        st.markdown("**CLAP + linear probe**")
        st.metric("Top prediction", predictions["clap"].iloc[0]["Sound category"],
                  f"{predictions['clap'].iloc[0]['Probability']:.1%}", delta_color="off")
        st.dataframe(predictions["clap"].style.format({"Probability": "{:.1%}"}), hide_index=True, width="stretch")
    st.caption("Probabilities rank the 50 ESC-50 categories; the highest-probability category is each model’s guess. The CLAP probe includes your added labeled clips when available.")
