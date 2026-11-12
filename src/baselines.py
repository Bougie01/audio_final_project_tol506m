"""Baselines on ESC-50 using the official folds.
Split: folds 1-3 = train, fold 4 = validation, fold 5 = test.
Tune on validation. The test set is only evaluated with --final.
Every run is appended to runs/results.csv."""
import argparse, csv, time
from pathlib import Path
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

LOG = Path("runs/results.csv")
FIELDS = ["time", "model", "params", "val_acc", "val_macro_f1", "test_acc", "test_macro_f1"]


def log(row):
    LOG.parent.mkdir(exist_ok=True)
    new = not LOG.exists()
    with open(LOG, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new: w.writeheader()
        w.writerow({"time": time.strftime("%F %T"), **row})


def metrics(model, X, y):
    p = model.predict(X)
    return accuracy_score(y, p), f1_score(y, p, average="macro")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", action="store_true", help="also evaluate the test fold (do this once, at the end)")
    args = ap.parse_args()

    d = np.load("cache/esc50_features.npz", allow_pickle=True)
    X, y, fold = d["mfcc"], d["y"], d["fold"]
    tr, va, te = np.isin(fold, [1, 2, 3]), fold == 4, fold == 5
    print(f"train {tr.sum()}  val {va.sum()}  test {te.sum()}")

    # 1) Majority class
    dummy = DummyClassifier(strategy="most_frequent").fit(X[tr], y[tr])
    a, f = metrics(dummy, X[va], y[va])
    print(f"majority:        val acc {a:.3f}  macro-F1 {f:.3f}")
    row = dict(model="majority", params="-", val_acc=a, val_macro_f1=f)
    if args.final:
        row["test_acc"], row["test_macro_f1"] = metrics(dummy, X[te], y[te])
    log(row)

    # 2) MFCC mean+std -> logistic regression (scaler fit on train only)
    best = None
    for C in [0.01, 0.1, 1, 10]:
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000))
        clf.fit(X[tr], y[tr])
        a, f = metrics(clf, X[va], y[va])
        print(f"MFCC+LogReg C={C:<5} val acc {a:.3f}  macro-F1 {f:.3f}")
        row = dict(model="mfcc_logreg", params=f"C={C}", val_acc=a, val_macro_f1=f)
        log(row)
        if best is None or a > best[0]:
            best = (a, C, clf)

    if args.final:
        _, C, clf = best
        a, f = metrics(clf, X[te], y[te])
        print(f"\nFINAL (best C={C} chosen on val)  test acc {a:.3f}  macro-F1 {f:.3f}")
        log(dict(model="mfcc_logreg", params=f"C={C} FINAL", val_acc=best[0],
                 val_macro_f1="", test_acc=a, test_macro_f1=f))


if __name__ == "__main__":
    main()
