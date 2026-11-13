"""Final evaluation on the test fold (fold 5) -- run once, at the end.

Scores the models that were chosen on validation, without retraining:
  - majority class and MFCC + logistic regression (C=0.1, chosen on val)
  - CNN + SpecAugment + mixup: the saved best-validation checkpoints of seeds 0, 1, 2
  - CLAP zero-shot ("This is a sound of X.") and CLAP linear probe (C=10, chosen on val)
and writes the error analysis:
  runs/test_predictions.csv   per-clip predictions of every model
  runs/error_analysis.md      per-class accuracy, most-confused pairs, per-group accuracy
FINAL rows for the baselines and CNN are appended to runs/results.csv (the CLAP
FINAL rows were already logged by train_probe.py --final).

Usage (from the repo root): python src/evaluate.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from baselines import log
from train_cnn import SmallCNN

CNN_SEEDS = (0, 1, 2)
DEMO_SEED = 1          # the checkpoint used by the demo; used for the error analysis
GROUPS = ["Animals", "Natural soundscapes & water", "Human, non-speech",
          "Interior / domestic", "Exterior / urban"]   # ESC-50: class id // 10


def score(y, p):
    return accuracy_score(y, p), f1_score(y, p, average="macro")


@torch.no_grad()
def cnn_predict(ckpt_path, X):
    ck = torch.load(ckpt_path, map_location="cpu")
    model = SmallCNN(); model.load_state_dict(ck["state_dict"]); model.eval()
    X = (X - ck["mean"]) / ck["std"]
    return torch.cat([model(X[i:i + 100]) for i in range(0, len(X), 100)]).argmax(1).numpy()


def main():
    # --- baselines (16 kHz MFCC features)
    b = np.load("cache/esc50_features.npz", allow_pickle=True)
    Xm, y, fold = b["mfcc"], b["y"], b["fold"]
    tr, va, te = np.isin(fold, [1, 2, 3]), fold == 4, fold == 5
    preds = {}
    dummy = DummyClassifier(strategy="most_frequent").fit(Xm[tr], y[tr])
    preds["majority"] = dummy.predict(Xm[te])
    mfcc = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000)).fit(Xm[tr], y[tr])
    preds["mfcc_logreg"] = mfcc.predict(Xm[te])

    # --- CNN (saved checkpoints, chosen on validation)
    c = np.load("cache/esc50_cnn_logmel.npz", allow_pickle=True)
    assert (c["y"] == y).all() and (c["fold"] == fold).all(), "cache row order mismatch"
    Xc = torch.from_numpy(c["logmel"][te].astype(np.float32))[:, None]
    cnn_scores = []
    for s in CNN_SEEDS:
        p = cnn_predict(Path(f"models/cnn_specaug_mixup_s{s}.pt"), Xc)
        preds[f"cnn_mixup_s{s}"] = p
        cnn_scores.append(score(y[te], p))

    # --- CLAP
    d = np.load("cache/esc50_clap.npz", allow_pickle=True)
    assert (d["y"] == y).all(), "cache row order mismatch"
    E, T = d["emb"], d["text_emb"]
    preds["clap_zeroshot"] = (E[te] @ T.T).argmax(1)
    preds["clap_probe"] = LogisticRegression(C=10, max_iter=3000).fit(E[tr], y[tr]).predict(E[te])

    names = [str(n) for n in d["class_names"]]

    # --- log FINAL rows (baselines + CNN; CLAP is already logged)
    for m, params in (("majority", "- FINAL"), ("mfcc_logreg", "C=0.1 FINAL")):
        a, f = score(y[te], preds[m])
        log(dict(model=m, params=params, test_acc=a, test_macro_f1=f))
    for s, (a, f) in zip(CNN_SEEDS, cnn_scores):
        log(dict(model="cnn_specaug_mixup", params=f"seed={s} FINAL (saved best-val checkpoint)",
                 test_acc=a, test_macro_f1=f))

    # --- summary
    print(f"{'model':<28}{'test acc':>10}{'macro-F1':>10}")
    for m in ("majority", "mfcc_logreg"):
        print(f"{m:<28}{score(y[te], preds[m])[0]:>10.3f}{score(y[te], preds[m])[1]:>10.3f}")
    ca, cf = np.array(cnn_scores).T
    print(f"{'cnn_mixup (3 seeds)':<28}{ca.mean():>6.3f}±{ca.std(ddof=1):.3f}{cf.mean():>6.3f}±{cf.std(ddof=1):.3f}")
    for m in ("clap_zeroshot", "clap_probe"):
        print(f"{m:<28}{score(y[te], preds[m])[0]:>10.3f}{score(y[te], preds[m])[1]:>10.3f}")

    # --- per-clip predictions
    out = pd.DataFrame({"filename": b["filename"][te], "true": [names[i] for i in y[te]]})
    for m, p in preds.items():
        out[m] = [names[i] for i in p]
    out.to_csv("runs/test_predictions.csv", index=False)

    # --- error analysis (demo CNN vs CLAP probe)
    yt = y[te]
    models = {"CNN (mixup, seed 1)": preds[f"cnn_mixup_s{DEMO_SEED}"], "CLAP probe": preds["clap_probe"],
              "CLAP zero-shot": preds["clap_zeroshot"]}
    per_class = pd.DataFrame({"class": names})
    for label, p in models.items():
        per_class[label] = [(p[yt == k] == k).mean() for k in range(len(names))]
    per_class["group"] = [GROUPS[k // 10] for k in range(len(names))]
    per_class.to_csv("runs/per_class_test_accuracy.csv", index=False)

    lines = ["# Error analysis (test fold 5, 8 clips per class)", ""]
    lines += ["## Accuracy per ESC-50 group", "", "| Group | " + " | ".join(models) + " |",
              "|---|" + "---|" * len(models)]
    for g in GROUPS:
        row = per_class[per_class.group == g]
        lines.append(f"| {g} | " + " | ".join(f"{row[m].mean():.2f}" for m in models) + " |")
    for label, p in models.items():
        wrong = pd.Series([f"{names[t]} -> {names[q]}" for t, q in zip(yt, p) if t != q]).value_counts()
        worst = per_class.sort_values(label).head(8)
        lines += ["", f"## {label}: {int((p != yt).sum())} errors out of {len(yt)}", "",
                  "Hardest classes: " + ", ".join(f"{r['class']} ({r[label]:.2f})" for _, r in worst.iterrows()),
                  "", "Most frequent confusions (true -> predicted, count):", ""]
        lines += [f"- {k}: {v}" for k, v in wrong.head(10).items()]
    Path("runs/error_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\nWrote runs/test_predictions.csv, runs/per_class_test_accuracy.csv, runs/error_analysis.md")


if __name__ == "__main__":
    main()
