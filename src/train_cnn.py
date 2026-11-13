"""Train a small 2D CNN from scratch on cached log-mel spectrograms.
Split: folds 1-3 train, 4 val, 5 test. Test is only used with --final.

Usage (from the repo root):
  python src/train_cnn.py --augment none          --seed 0
  python src/train_cnn.py --augment specaug       --seed 0
  python src/train_cnn.py --augment specaug_mixup --seed 0
  python src/train_cnn.py --augment specaug_mixup --seed 1 --final   # once, at the end

The checkpoint with the best validation accuracy is kept (early stopping) and
saved to models/ with the normalisation statistics the demo needs. Per-epoch
learning curves go to runs/curves/, and a summary row to runs/results.csv.
~6 s/epoch on an RTX 3060 laptop GPU (80 epochs = ~7.5 min).
"""
import argparse, csv, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from baselines import log


class SmallCNN(nn.Module):
    """4 blocks of (conv-BN-ReLU) x2 + max-pool, then pool over frequency and time, then linear."""

    def __init__(self, n_classes=50, widths=(32, 64, 128, 256), dropout=0.3):
        super().__init__()
        blocks, c_in = [], 1
        for c in widths:
            blocks += [nn.Conv2d(c_in, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), nn.ReLU(),
                       nn.Conv2d(c, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), nn.ReLU(),
                       nn.MaxPool2d(2)]
            c_in = c
        self.features = nn.Sequential(*blocks)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(c_in, n_classes)

    def forward(self, x):                   # x: [B, 1, mels, frames]
        h = self.features(x).mean(dim=2)    # average over frequency -> [B, C, T]
        h = h.mean(dim=2) + h.amax(dim=2)   # mean + max over time   -> [B, C]
        return self.fc(self.dropout(h))


def spec_augment(x, n_freq=2, f_max=15, n_time=2, t_max=30, max_shift=40):
    """Random circular time shift plus SpecAugment frequency and time masks."""
    B, _, M, T = x.shape
    shifts = torch.randint(-max_shift, max_shift + 1, (B,))
    x = torch.stack([torch.roll(x[i], int(shifts[i]), dims=-1) for i in range(B)])
    for size, n, w_max, dim in ((M, n_freq, f_max, 2), (T, n_time, t_max, 3)):
        idx = torch.arange(size, device=x.device)[None]
        for _ in range(n):
            w = torch.randint(0, w_max + 1, (B,), device=x.device)
            start = (torch.rand(B, device=x.device) * (size - w)).long()
            mask = (idx >= start[:, None]) & (idx < (start + w)[:, None])
            shape = (B, 1, size, 1) if dim == 2 else (B, 1, 1, size)
            x = x.masked_fill(mask.view(shape), 0.0)
    return x


@torch.no_grad()
def predict(model, X, device, bs=128):
    model.eval()
    return torch.cat([model(X[i:i + bs].to(device)).softmax(-1).cpu() for i in range(0, len(X), bs)])


def score(p, y):
    return accuracy_score(y, p), f1_score(y, p, average="macro")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--augment", choices=["none", "specaug", "specaug_mixup"], default="specaug_mixup")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--final", action="store_true", help="also evaluate the test fold (once, at the end)")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    name = f"cnn_{args.augment}_s{args.seed}"

    d = np.load("cache/esc50_cnn_logmel.npz", allow_pickle=True)
    X = torch.from_numpy(d["logmel"].astype(np.float32))[:, None]
    y = torch.from_numpy(d["y"].astype(np.int64))
    fold = d["fold"]
    tr, va, te = (torch.from_numpy(m) for m in (np.isin(fold, [1, 2, 3]), fold == 4, fold == 5))

    mean, std = X[tr].mean().item(), X[tr].std().item()   # training folds only
    X = (X - mean) / std
    Xtr, ytr = X[tr], y[tr]

    model = SmallCNN().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps = args.epochs * ((len(Xtr) + args.bs - 1) // args.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps)

    Path("runs/curves").mkdir(parents=True, exist_ok=True)
    Path("models").mkdir(exist_ok=True)
    ckpt = Path("models") / f"{name}.pt"
    curve, best_acc, best_epoch = [], -1.0, -1
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm, total = torch.randperm(len(Xtr)), 0.0
        for i in range(0, len(Xtr), args.bs):
            idx = perm[i:i + args.bs]
            xb, yb = Xtr[idx].to(device), ytr[idx].to(device)
            if args.augment != "none":
                xb = spec_augment(xb)
            if args.augment == "specaug_mixup":
                lam = float(np.random.beta(0.4, 0.4))
                j = torch.randperm(len(xb), device=device)
                logits = model(lam * xb + (1 - lam) * xb[j])
                loss = lam * F.cross_entropy(logits, yb) + (1 - lam) * F.cross_entropy(logits, yb[j])
            else:
                loss = F.cross_entropy(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            total += loss.item() * len(idx)

        tr_acc = (predict(model, Xtr, device).argmax(1) == ytr).float().mean().item()
        p_va = predict(model, X[va], device)
        val_loss = F.nll_loss(p_va.clamp_min(1e-9).log(), y[va]).item()
        a, f = score(p_va.argmax(1).numpy(), y[va].numpy())
        curve.append(dict(epoch=epoch, train_loss=total / len(Xtr), train_acc=tr_acc,
                          val_loss=val_loss, val_acc=a, val_macro_f1=f))
        if a > best_acc:
            best_acc, best_epoch = a, epoch
            torch.save({"state_dict": model.state_dict(), "mean": mean, "std": std,
                        "classes": [str(c) for c in d["class_names"]], "args": vars(args)}, ckpt)
        print(f"ep {epoch:3d}  loss {total / len(Xtr):.3f}  train acc {tr_acc:.3f}  "
              f"val loss {val_loss:.3f}  val acc {a:.3f}")
    train_s = time.time() - t0

    with open(f"runs/curves/{name}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(curve[0])); w.writeheader(); w.writerows(curve)

    model.load_state_dict(torch.load(ckpt)["state_dict"])     # best-validation checkpoint
    a, f = score(predict(model, X[va], device).argmax(1).numpy(), y[va].numpy())

    cpu = SmallCNN(); cpu.load_state_dict(model.state_dict()); cpu.eval()
    with torch.no_grad():                                      # CPU latency for one 5 s clip
        for _ in range(3): cpu(X[:1])
        t1 = time.perf_counter()
        for _ in range(20): cpu(X[:1])
        cpu_ms = (time.perf_counter() - t1) / 20 * 1000

    params = (f"seed={args.seed} best_epoch={best_epoch} params={n_params} "
              f"train_s={train_s:.0f} cpu_ms={cpu_ms:.1f}")
    row = dict(model=f"cnn_{args.augment}", params=params, val_acc=a, val_macro_f1=f)
    msg = f"best epoch {best_epoch}: val acc {a:.3f} macro-F1 {f:.3f}"
    if args.final:
        ta, tf = score(predict(model, X[te], device).argmax(1).numpy(), y[te].numpy())
        row.update(test_acc=ta, test_macro_f1=tf)
        msg += f" | FINAL test acc {ta:.3f} macro-F1 {tf:.3f}"
    log(row)
    print(msg)
    print(f"params {n_params:,}  train time {train_s:.0f}s  CPU inference {cpu_ms:.1f} ms/clip")


if __name__ == "__main__":
    main()
