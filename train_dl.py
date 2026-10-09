"""Analyser B (deep learning): a 1D-CNN or BiLSTM that reads one rep's joint-angle sequence and says whether
the rep is valid. Evaluated leave-one-person-out and compared with a depth-threshold rule baseline.

  python build_sequences.py cfrep
  python train_dl.py                          # both architectures, LOPO table
  python train_dl.py --arch cnn --epochs 120 --seeds 5
  python train_dl.py --arch cnn --save models/analyser_b_cnn.pt      # final model trained on everyone

CPU is enough (a few thousand parameters); it runs unchanged on Colab or Kaggle, where it uses the GPU if present.
"""
import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class CNN1D(nn.Module):
    def __init__(self, c_in, hidden=32, p=0.3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(c_in, hidden, 5, padding=2), nn.BatchNorm1d(hidden), nn.ReLU(),
            nn.Conv1d(hidden, hidden * 2, 5, padding=2), nn.BatchNorm1d(hidden * 2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(hidden * 2, hidden * 2, 3, padding=1), nn.BatchNorm1d(hidden * 2), nn.ReLU())
        self.head = nn.Sequential(nn.Dropout(p), nn.Linear(hidden * 4 + 1, 32), nn.ReLU(), nn.Dropout(p), nn.Linear(32, 2))

    def forward(self, x, dur):
        h = self.conv(x)
        return self.head(torch.cat([h.mean(-1), h.amax(-1), dur[:, None]], 1))


class BiLSTM(nn.Module):
    def __init__(self, c_in, hidden=32, p=0.3):
        super().__init__()
        self.rnn = nn.LSTM(c_in, hidden, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Dropout(p), nn.Linear(hidden * 4 + 1, 32), nn.ReLU(), nn.Dropout(p), nn.Linear(32, 2))

    def forward(self, x, dur):
        h, _ = self.rnn(x.transpose(1, 2))
        return self.head(torch.cat([h.mean(1), h.amax(1), dur[:, None]], 1))


ARCHS = {"cnn": CNN1D, "lstm": BiLSTM}


def augment(x):
    """Random time stretch/shift, per-channel gain and a little noise, so a few hundred reps go further."""
    b, c, t = x.shape
    grid = torch.linspace(-1, 1, t, device=x.device)[None] * torch.empty(b, 1, device=x.device).uniform_(0.85, 1.15)
    grid = (grid + torch.empty(b, 1, device=x.device).uniform_(-0.1, 0.1)).clamp(-1, 1)
    grid = torch.stack([grid, torch.zeros_like(grid)], -1)[:, None]            # (b, 1, t, 2)
    x = F.grid_sample(x[:, :, None, :], grid, align_corners=True, padding_mode="border")[:, :, 0]
    return x * torch.empty(b, c, 1, device=x.device).uniform_(0.97, 1.03) + 0.01 * torch.randn_like(x)


def train_one(arch, xtr, dtr, ytr, epochs, seed, lr=3e-3, batch=32):
    torch.manual_seed(seed)
    model = ARCHS[arch](xtr.shape[1]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * ((len(xtr) + batch - 1) // batch))
    counts = torch.bincount(ytr, minlength=2).float().clamp(min=1)
    weight = (counts.sum() / (2 * counts)).to(DEVICE)                          # rarer class counts for more
    xtr, dtr, ytr = xtr.to(DEVICE), dtr.to(DEVICE), ytr.to(DEVICE)
    for _ in range(epochs):
        model.train()
        for idx in torch.randperm(len(xtr), device=DEVICE).split(batch):
            if len(idx) < 2:
                continue
            loss = F.cross_entropy(model(augment(xtr[idx]), dtr[idx]), ytr[idx], weight=weight)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    return model


@torch.no_grad()
def predict(model, x, d):
    model.eval()
    return F.softmax(model(x.to(DEVICE), d.to(DEVICE)), -1)[:, 1].cpu().numpy()      # P(invalid)


def best_threshold(score, y, invalid_if_low):
    """Cut-off on one number that maximises F1 for 'invalid' on the training reps (the rule baseline)."""
    best, thr = -1, None
    for t in np.unique(score):
        pred = (score <= t) if invalid_if_low else (score >= t)
        f = f1(y, pred)
        if f > best:
            best, thr = f, t
    return thr


def f1(y, pred):
    tp, fp, fn = int(((y == 1) & pred).sum()), int(((y == 0) & pred).sum()), int(((y == 1) & ~pred).sum())
    return 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0


def auc(y, score):
    pos, neg = score[y == 1], score[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float(((pos[:, None] > neg[None]).mean() + 0.5 * (pos[:, None] == neg[None]).mean()))


def metrics(y, pred, score):
    """Scores for the 'invalid rep' class."""
    tp, fp, fn = int(((y == 1) & pred).sum()), int(((y == 0) & pred).sum()), int(((y == 1) & ~pred).sum())
    tn = int(((y == 0) & ~pred).sum())
    prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return dict(precision=prec, recall=rec, f1=f1(y, pred), balanced_acc=0.5 * (rec + tn / max(tn + fp, 1)),
                auc=auc(y, score), tp=tp, fp=fp, fn=fn, tn=tn)


def report(name, y, pred, score, views):
    m = metrics(y, pred, score)
    per_view = "  ".join(f"{v} {f1(y[views == v], pred[views == v]):.2f}" for v in ("front", "diagonal", "side"))
    print(f"{name:<24}{m['precision']:>6.2f}{m['recall']:>6.2f}{m['f1']:>6.2f}{m['balanced_acc']:>7.2f}"
          f"{m['auc']:>7.2f}   | F1 by view: {per_view}")


def rule_baseline(d, key, invalid_if_low, folds=0):
    """Held-out-person predictions of a one-threshold rule on d[key] (threshold learned on the other people)."""
    s, yn, person = d[key], d["y"], d["person"]
    pred = np.zeros(len(yn), bool)
    for held in person_folds(person, folds):
        te = np.isin(person, held)
        thr = best_threshold(s[~te], yn[~te], invalid_if_low)
        pred[te] = (s[te] <= thr) if invalid_if_low else (s[te] >= thr)
    return pred, (-s if invalid_if_low else s)


def person_folds(person, folds=0, seed=0):
    """Groups of people that are held out together: one person per fold (leave-one-person-out, folds=0),
    or `folds` roughly equal groups (faster when there are many people)."""
    people = sorted(set(person))
    if not folds or folds >= len(people):
        return [[p] for p in people]
    order = np.random.RandomState(seed).permutation(len(people))
    return [[people[i] for i in order[k::folds]] for k in range(folds)]


def split_people(person, n_test, seed=0):
    """Boolean mask of reps from `n_test` randomly chosen people - the locked final test set."""
    people = sorted(set(person))
    test = np.random.RandomState(seed).choice(people, n_test, replace=False)
    return np.isin(person, test)


def lopo_probs(arch, x, dur, y, person, epochs=80, seeds=3, folds=0, batch=32):
    """P(invalid) for every rep from models that never saw that rep's person (average of `seeds` models).
    folds=0: leave-one-person-out; folds=k: k groups of people held out in turn."""
    prob = np.zeros(len(y))
    for held in person_folds(person, folds):
        te = np.isin(person, held)
        tr = ~te
        prob[te] = np.mean([predict(train_one(arch, x[tr], dur[tr], y[tr], epochs, seed, batch=batch), x[te], dur[te])
                            for seed in range(seeds)], 0)
    return prob


def load(path):
    """-> (npz dict, x tensor (N, C, T), dur tensor, y tensor) from a build_sequences.py file."""
    d = np.load(path)
    return d, torch.tensor(d["x"]), torch.tensor(d["dur"] / 3.0), torch.tensor(d["y"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="datasets/sequences_cfrep.npz")
    ap.add_argument("--arch", choices=["cnn", "lstm", "both"], default="both")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--seeds", type=int, default=3, help="models averaged per fold (small data is noisy)")
    ap.add_argument("--folds", type=int, default=0, help="0 = leave-one-person-out, k = k groups of people")
    ap.add_argument("--save", default="", help="train on everyone and save the model here (needs --arch cnn or lstm)")
    args = ap.parse_args()
    torch.set_num_threads(2)

    d, x, dur, y = load(args.data)
    person, view, yn = d["person"], d["view"], d["y"]
    print(f"{len(yn)} reps, {int(yn.sum())} invalid, {len(set(person))} people, device {DEVICE}")
    print(f"{'method (leave-one-person-out)':<24}{'P':>6}{'R':>6}{'F1':>6}{'balAcc':>7}{'AUC':>7}   (class = invalid rep)")

    for name, key, low in (("rule: hip below knee", "max_hbk", True), ("rule: knee angle", "min_knee", False)):
        pred, score = rule_baseline(d, key, low, args.folds)
        report(name, yn, pred, score, view)

    archs = ["cnn", "lstm"] if args.arch == "both" else [args.arch]
    for arch in archs:
        prob = lopo_probs(arch, x, dur, y, person, args.epochs, args.seeds, args.folds)
        report(f"DL: {arch.upper()} ({args.seeds} seeds)", yn, prob >= 0.5, prob, view)

    if args.save:
        if len(archs) != 1:
            raise SystemExit("--save needs --arch cnn or --arch lstm")
        model = train_one(archs[0], x, dur, y, args.epochs, 0)
        os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
        torch.save(dict(arch=archs[0], state=model.state_dict(), channels=d["channels"].tolist(),
                        seq_len=x.shape[-1]), args.save)
        print("saved", args.save)


if __name__ == "__main__":
    main()
