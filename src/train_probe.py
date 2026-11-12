"""CLAP on ESC-50: zero-shot baseline, linear probe, small MLP head (3 seeds).
Split: folds 1-3 train, 4 val, 5 test. Test is only used with --final."""
import argparse
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score
from baselines import log


def score(p, y):
    return accuracy_score(y, p), f1_score(y, p, average="macro")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", action="store_true")
    args = ap.parse_args()

    d = np.load("cache/esc50_clap.npz", allow_pickle=True)
    X, y, fold, T = d["emb"], d["y"], d["fold"], d["text_emb"]
    tr, va, te = np.isin(fold, [1, 2, 3]), fold == 4, fold == 5
    print(f"train {tr.sum()}  val {va.sum()}  test {te.sum()}")

    def row(model, params, pv, pt=None):
        a, f = score(pv, y[va]); r = dict(model=model, params=params, val_acc=a, val_macro_f1=f)
        if pt is not None: r["test_acc"], r["test_macro_f1"] = score(pt, y[te])
        log(r); return a, f

    # 1) Zero-shot: no training, nearest text prompt
    zs = lambda m: (X[m] @ T.T).argmax(1)
    a, f = row("clap_zeroshot", "prompt='This is a sound of X.'", zs(va), zs(te) if args.final else None)
    print(f"zero-shot            val acc {a:.3f}  macro-F1 {f:.3f}")

    # 2) Linear probe, C tuned on validation
    best = None
    for C in [0.1, 1, 10, 100]:
        clf = LogisticRegression(C=C, max_iter=3000).fit(X[tr], y[tr])
        a, f = row("clap_linear_probe", f"C={C}", clf.predict(X[va]))
        print(f"linear probe C={C:<5} val acc {a:.3f}  macro-F1 {f:.3f}")
        if best is None or a > best[0]: best = (a, C, clf)
    if args.final:
        _, C, clf = best
        a, f = score(clf.predict(X[te]), y[te])
        print(f"FINAL linear probe (C={C} chosen on val): test acc {a:.3f}  macro-F1 {f:.3f}")
        log(dict(model="clap_linear_probe", params=f"C={C} FINAL", test_acc=a, test_macro_f1=f))

    # 3) MLP head, 3 seeds -> mean +- std
    accs, taccs = [], []
    for s in range(3):
        m = MLPClassifier((512,), alpha=1e-3, max_iter=300, random_state=s).fit(X[tr], y[tr])
        a, _ = row("clap_mlp_probe", f"seed={s}", m.predict(X[va]), m.predict(X[te]) if args.final else None)
        accs.append(a)
        if args.final: taccs.append(accuracy_score(y[te], m.predict(X[te])))
    print(f"MLP head (3 seeds)   val acc {np.mean(accs):.3f} +- {np.std(accs):.3f}")
    if args.final: print(f"FINAL MLP head        test acc {np.mean(taccs):.3f} +- {np.std(taccs):.3f}")


if __name__ == "__main__":
    main()
