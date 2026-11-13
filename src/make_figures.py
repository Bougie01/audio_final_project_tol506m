"""Report figures -> runs/figures/ (PNG, 200 dpi, sized for a two-column report page).

  results_overview.png    test accuracy of every model vs the baseline
  learning_curves.png     CNN train/val accuracy per epoch, no augmentation vs SpecAugment+mixup
  hardest_classes.png     per-class test accuracy, CNN vs CLAP probe, for the CNN's 12 hardest classes
  confusion_cnn.png       50x50 test confusion matrix of the demo CNN

Usage (from the repo root, after evaluate.py): python src/make_figures.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

OUT = Path("runs/figures")
# validated reference palette (light mode): categorical slots 1-2, text and grid tokens
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"
SEQ = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "legend.frameon": False, "savefig.dpi": 200, "savefig.bbox": "tight",
})


def results_overview():
    pred = pd.read_csv("runs/test_predictions.csv")
    acc = lambda col: (pred[col] == pred["true"]).mean()
    cnn = np.array([acc(f"cnn_mixup_s{s}") for s in (0, 1, 2)])
    rows = [("Majority class", acc("majority"), 0),
            ("MFCC + logistic regression", acc("mfcc_logreg"), 0),
            ("CNN from scratch (mixup, 3 seeds)", cnn.mean(), cnn.std()),
            ("CLAP zero-shot (no training)", acc("clap_zeroshot"), 0),
            ("CLAP + linear probe", acc("clap_probe"), 0)]
    fig, ax = plt.subplots(figsize=(6.2, 2.4))
    ylab = [r[0] for r in rows][::-1]
    vals = [r[1] for r in rows][::-1]
    errs = [r[2] for r in rows][::-1]
    bars = ax.barh(ylab, vals, height=0.55, color=BLUE)
    for b, v, e in zip(bars, vals, errs):
        if e:   # seed spread only exists for the CNN
            ax.errorbar(v, b.get_y() + b.get_height() / 2, xerr=e, color=INK2, lw=1, capsize=3)
        txt = f"{v:.3f}" + (f" ± {e:.3f}" if e else "")
        ax.text(v + max(e, 0) + 0.01, b.get_y() + b.get_height() / 2, txt, va="center", color=INK, fontsize=8.5)
    ax.set_xlim(0, 1.12); ax.set_xticks(np.arange(0, 1.01, 0.2))
    ax.set_xlabel("Test accuracy (ESC-50 fold 5, 400 clips, 50 classes)")
    ax.grid(axis="y", visible=False)
    ax.set_title("Test accuracy by model")
    fig.savefig(OUT / "results_overview.png"); plt.close(fig)


def learning_curves():
    fig, ax = plt.subplots(figsize=(6.2, 2.8))
    for variant, color, label in (("none", ORANGE, "No augmentation"), ("specaug_mixup", BLUE, "SpecAugment + mixup")):
        cs = [pd.read_csv(f"runs/curves/cnn_{variant}_s{s}.csv") for s in (0, 1, 2)]
        ep = cs[0].epoch.values
        va = np.stack([c.val_acc.values for c in cs]); trn = np.stack([c.train_acc.values for c in cs])
        ax.fill_between(ep, va.min(0), va.max(0), color=color, alpha=0.15, lw=0)
        ax.plot(ep, va.mean(0), color=color, lw=2, label=f"{label}: validation (final {va.mean(0)[-1]:.2f})")
        ax.plot(ep, trn.mean(0), color=color, lw=1.2, ls="--", label=f"{label}: training")
    ax.set_xlim(1, 80); ax.set_ylim(0, 1.02)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
    ax.set_title("CNN learning curves (mean of 3 seeds; band = min-max)")
    ax.legend(loc="lower right", fontsize=7.5, ncol=1)
    fig.savefig(OUT / "learning_curves.png"); plt.close(fig)


def hardest_classes():
    pc = pd.read_csv("runs/per_class_test_accuracy.csv")
    cnn_col, clap_col = "CNN (mixup, seed 1)", "CLAP probe"
    worst = pc.sort_values([cnn_col, clap_col]).head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    yy = np.arange(len(worst))
    tie = (worst[cnn_col] == worst[clap_col]).values * 0.18     # nudge equal values apart
    ax.hlines(yy, worst[cnn_col], worst[clap_col], color=GRID, lw=2, zorder=1)
    ax.scatter(worst[cnn_col], yy - tie, s=40, color=ORANGE, label="CNN from scratch", zorder=3,
               edgecolor="white", linewidth=1)
    ax.scatter(worst[clap_col], yy + tie, s=40, color=BLUE, label="CLAP + linear probe", zorder=3,
               edgecolor="white", linewidth=1)
    ax.set_yticks(yy, [c.replace("_", " ") for c in worst["class"]])
    ax.set_xlim(-0.03, 1.03); ax.set_xlabel("Test accuracy per class (8 clips each)")
    ax.grid(axis="y", visible=False)
    ax.set_title("The CNN's 12 hardest classes", pad=22)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, fontsize=8, borderaxespad=0.2)
    fig.savefig(OUT / "hardest_classes.png"); plt.close(fig)


def confusion_cnn():
    pred = pd.read_csv("runs/test_predictions.csv")
    names = pd.read_csv("runs/per_class_test_accuracy.csv")["class"].tolist()
    idx = {n: i for i, n in enumerate(names)}
    M = np.zeros((50, 50), int)
    for t, p in zip(pred["true"], pred["cnn_mixup_s1"]):
        M[idx[t], idx[p]] += 1
    fig, ax = plt.subplots(figsize=(7.5, 7.0))
    im = ax.imshow(M, cmap=LinearSegmentedColormap.from_list("seq", SEQ), vmin=0, vmax=8)
    labels = [n.replace("_", " ") for n in names]
    ax.set_xticks(range(50), labels, rotation=90, fontsize=6)
    ax.set_yticks(range(50), labels, fontsize=6)
    for k in range(1, 5):                       # separate the five ESC-50 groups
        ax.axhline(k * 10 - 0.5, color=MUTED, lw=0.6); ax.axvline(k * 10 - 0.5, color=MUTED, lw=0.6)
    ax.grid(False)
    ax.set_xlabel("Predicted class"); ax.set_ylabel("True class")
    ax.set_title("CNN confusion matrix, test fold (8 clips per class)")
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02); cb.set_label("Clips", color=INK2)
    fig.savefig(OUT / "confusion_cnn.png"); plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    results_overview(); learning_curves(); hardest_classes(); confusion_cnn()
    print("Saved figures to", OUT)
