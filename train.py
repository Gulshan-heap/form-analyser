"""Analyser B: train one small random forest per fault on the labelled reps.csv files, and compare it
with the rule-based analyser using leave-one-person-out (LOPO) cross-validation.

  python train.py reps_*.csv                         # LOPO table: rules vs learned
  python train.py reps_*.csv --holdout dana,eli      # train on the rest, final test on dana+eli
  python train.py reps_*.csv --save models/analyser_b.joblib

Label column (fill by hand): "good", or one or more fault codes joined by "|", e.g.
  good
  shallow_depth
  excess_lean|knees_forward
Reps with an empty label are skipped. Each fault is only scored on reps from camera views where
the rule analyser can judge it, so rules and learned model are compared on the same reps.
"""
import argparse
import csv
import sys
from collections import defaultdict

import numpy as np

FAULTS = ["knees_inward", "excess_lean", "knees_forward", "shallow_depth"]
SIDE_FAULTS = {"excess_lean", "knees_forward", "shallow_depth"}      # judged from side/diagonal
VIEWS = {f: ({"side", "diagonal"} if f in SIDE_FAULTS else {"front", "diagonal"}) for f in FAULTS}
GOOD = {"good", "ok", "none", "correct", "clean"}
NUMERIC = ["duration", "descent_time", "min_knee", "hip_at_bottom", "lean_at_bottom", "max_lean",
           "knee_fwd_bottom", "valgus_idx"]
FEATURES = NUMERIC + ["view_front", "view_side"]


def parse_label(raw, where):
    tokens = [t.strip().lower() for t in raw.replace(",", "|").split("|") if t.strip()]
    if not tokens:
        return None
    if len(tokens) == 1 and tokens[0] in GOOD:
        return set()
    bad = [t for t in tokens if t not in FAULTS]
    if bad:
        sys.exit(f"{where}: unknown label {bad}. Use 'good' or any of {FAULTS} joined by '|'.")
    return set(tokens)


def num(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return np.nan


def load(paths):
    rows = []
    for path in paths:
        with open(path, newline="") as f:
            for n, r in enumerate(csv.DictReader(f), start=2):
                labels = parse_label(r.get("label", ""), f"{path}:{n}")
                if labels is None:
                    continue
                x = [num(r.get(c)) for c in NUMERIC] + [float(r["view"] == "front"), float(r["view"] == "side")]
                rows.append(dict(person=r["person"], view=r["view"], x=x, labels=labels,
                                 rules=set(filter(None, r.get("rule_faults", "").split("|")))))
    if not rows:
        sys.exit("No labelled reps found. Fill the 'label' column of reps.csv first.")
    return rows


def make_model(seed=0):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    return make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True, add_indicator=True),
        RandomForestClassifier(n_estimators=200, min_samples_leaf=2, class_weight="balanced_subsample",
                               random_state=seed, n_jobs=1))


def prf(y, p):
    tp, fp, fn = int(np.sum((y == 1) & (p == 1))), int(np.sum((y == 0) & (p == 1))), int(np.sum((y == 1) & (p == 0)))
    pr = tp / (tp + fp) if tp + fp else float("nan")
    rc = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * pr * rc / (pr + rc) if tp and pr + rc else (0.0 if tp + fp + fn else float("nan"))
    return pr, rc, f1


def fit_predict(train, test, seed):
    """Fit on `train` (X, y) and predict `test` X. If train has one class, predict that class."""
    X, y = train
    if len(set(y)) < 2:
        return np.full(len(test), int(y[0]) if len(y) else 0)
    return make_model(seed).fit(X, y).predict(test)


def subset(rows, fault):
    rows = [r for r in rows if r["view"] in VIEWS[fault]]
    X = np.array([r["x"] for r in rows]) if rows else np.empty((0, len(FEATURES)))
    y = np.array([int(fault in r["labels"]) for r in rows])
    rules = np.array([int(fault in r["rules"]) for r in rows])
    people = np.array([r["person"] for r in rows])
    return X, y, rules, people


def lopo(rows, fault, seed):
    """Pooled leave-one-person-out predictions. Returns (y, learned, rules) or None if not evaluable."""
    X, y, rules, people = subset(rows, fault)
    if len(set(people)) < 2 or y.sum() == 0 or y.sum() == len(y):
        return None
    pred = np.zeros(len(y), dtype=int)
    for person in sorted(set(people)):
        te = people == person
        pred[te] = fit_predict((X[~te], y[~te]), X[te], seed)
    return y, pred, rules


def table(title, results):
    print(f"\n{title}")
    print(f"{'fault':<15}{'n':>5}{'pos':>5} | {'rules P':>8}{'R':>6}{'F1':>6} | {'learned P':>10}{'R':>6}{'F1':>6}")
    f1s = {"rules": [], "learned": []}
    for fault in FAULTS:
        res = results.get(fault)
        if res is None:
            print(f"{fault:<15}{'-':>5}{'-':>5} | not evaluable (need >=2 people and both classes)")
            continue
        y, pred, rules = res
        rp, rr, rf = prf(y, rules)
        lp, lr, lf = prf(y, pred)
        f1s["rules"].append(rf), f1s["learned"].append(lf)
        print(f"{fault:<15}{len(y):>5}{int(y.sum()):>5} | {rp:>8.2f}{rr:>6.2f}{rf:>6.2f} | {lp:>10.2f}{lr:>6.2f}{lf:>6.2f}")
    if f1s["rules"]:
        print(f"{'macro F1':<25} | {'':>14}{np.nanmean(f1s['rules']):>6.2f} | {'':>16}{np.nanmean(f1s['learned']):>6.2f}")


def describe(rows):
    by = defaultdict(lambda: defaultdict(int))
    for r in rows:
        for f in (r["labels"] or {"good"}):
            by[r["person"]][f] += 1
    print(f"{len(rows)} labelled reps from {len(by)} people")
    for person, counts in sorted(by.items()):
        print(f"  {person:<12} " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="+", help="labelled reps.csv file(s)")
    ap.add_argument("--holdout", default="", help="comma-separated people kept out of training for a final test")
    ap.add_argument("--save", default="", help="save the final models to this .joblib")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = load(args.csv)
    describe(rows)
    held = {p.strip() for p in args.holdout.split(",") if p.strip()}
    unknown = held - {r["person"] for r in rows}
    if unknown:
        sys.exit(f"--holdout people not found in the data: {sorted(unknown)}")
    train = [r for r in rows if r["person"] not in held]
    test = [r for r in rows if r["person"] in held]

    table("Leave-one-person-out" + (f" (excluding held-out {sorted(held)})" if held else "")
          + "  - every rep is predicted by a model that never saw that person",
          {f: lopo(train, f, args.seed) for f in FAULTS})

    models = {}
    for fault in FAULTS:
        X, y, _, _ = subset(train, fault)
        if len(y) and 0 < y.sum() < len(y):
            models[fault] = make_model(args.seed).fit(X, y)

    if test:
        res = {}
        for fault in FAULTS:
            X, y, rules, _ = subset(test, fault)
            if fault in models and len(y) and 0 < y.sum() < len(y):
                res[fault] = (y, models[fault].predict(X), rules)
        table(f"Held-out test people {sorted(held)}", res)

    if models:
        print("\nTop features per fault (impurity importance)")
        names = FEATURES  # indicator columns are appended after these; only the first block is shown
        for fault, m in models.items():
            imp = m[-1].feature_importances_[:len(names)]
            top = np.argsort(imp)[::-1][:3]
            print(f"  {fault:<15}" + ", ".join(f"{names[i]} {imp[i]:.2f}" for i in top))

    if args.save:
        import os
        import joblib
        os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
        joblib.dump(dict(models=models, features=FEATURES, views={f: sorted(v) for f, v in VIEWS.items()}),
                    args.save)
        print(f"\nSaved {len(models)} models -> {args.save}")


if __name__ == "__main__":
    main()
