"""Phase 3b: train a classifier that predicts whether MediaPipe's landmarks on a frame are reliable.
Label: reliable = landmark error (vs 300VW ground truth, divided by eye distance) <= --thr.
Cross-validation splits by VIDEO, so no clip is in both train and test. NumPy only."""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from features import FEATURES
from reliability_model import GBM, Logistic, save_model

HERE = Path(__file__).resolve().parent


def balanced_weights(y):
    n1 = y.sum(); n0 = len(y) - n1
    return np.where(y == 1, len(y) / (2 * n1), len(y) / (2 * n0))


def auc(y, p):
    r = pd.Series(p).rank().to_numpy()
    n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def report(name, y, pred, p=None):
    def pr(c):
        tp = ((pred == c) & (y == c)).sum()
        return tp / max((pred == c).sum(), 1), tp / max((y == c).sum(), 1)
    (p0, r0), (p1, r1) = pr(0), pr(1)
    bal = (r0 + r1) / 2
    s = (f"{name:20s} balanced_acc {bal:.3f} | unreliable: precision {p0:.3f} recall {r0:.3f} | "
         f"reliable: precision {p1:.3f} recall {r1:.3f}")
    if p is not None:
        s += f" | AUC {auc(y, p):.3f}"
    print(s)
    return bal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", default=str(HERE / "data" / "feats"))
    ap.add_argument("--thr", type=float, default=0.10, help="error above this (x eye distance) = unreliable")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--prec", type=float, default=0.98, help="precision target for the OK cut-off")
    a = ap.parse_args()

    files = sorted(Path(a.feats).glob("*.csv"))
    df = pd.concat([pd.read_csv(f, dtype={"video": str}) for f in files], ignore_index=True)
    print(f"{len(files)} videos, {len(df)} rows, face found in {100 * df['face'].mean():.1f}% of rows")
    df = df[(df["face"] == 1) & df["err"].notna()].reset_index(drop=True)
    print("landmark error (x eye distance) quantiles:")
    print(df["err"].quantile([.5, .75, .9, .95, .99]).round(3).to_string())
    for t in (0.08, 0.10, 0.15):
        print(f"  share unreliable if threshold = {t}: {100 * (df['err'] > t).mean():.1f}%")

    y = (df["err"] <= a.thr).astype(int).to_numpy()          # 1 = reliable
    X = df[FEATURES].to_numpy(dtype=float)
    vids = df["video"].to_numpy()
    print(f"\nthreshold used: {a.thr} -> reliable {100 * y.mean():.1f}%, unreliable {100 * (1 - y.mean()):.1f}%")
    if y.min() == y.max():
        raise SystemExit("only one class present; change --thr")

    uv = np.array(sorted(set(vids)))
    np.random.default_rng(0).shuffle(uv)
    fold_of = {v: i % min(a.folds, len(uv)) for i, v in enumerate(uv)}
    fold = np.array([fold_of[v] for v in vids])

    makers = {"logistic_regression": Logistic, "gradient_boosting": GBM}
    print("\nCross-validated (each video is tested by a model that never saw it):")
    base = report("rule baseline", y, (df["rule_status"] == "ok").astype(int).to_numpy())
    oof, bal = {}, {}
    for name, mk in makers.items():
        t = time.perf_counter()
        p = np.zeros(len(y))
        for k in range(fold.max() + 1):
            tr, te = fold != k, fold == k
            p[te] = mk().fit(X[tr], y[tr], balanced_weights(y[tr])).predict_proba(X[te])
        oof[name] = p
        bal[name] = report(name, y, (p >= 0.5).astype(int), p)
        print(f"   ({time.perf_counter() - t:.0f}s)")
    best = max(oof, key=lambda k: auc(y, oof[k]))
    print(f"\nbest model by AUC: {best}")
    print("VERDICT:", "model beats the rule baseline" if bal[best] > base else "model does NOT beat the rule baseline",
          f"({bal[best]:.3f} vs {base:.3f} balanced accuracy)")

    # P(reliable) above which >= 90% of frames called reliable really are reliable
    order = np.argsort(-oof[best])
    ps, ys = oof[best][order], y[order]
    prec = np.cumsum(ys) / np.arange(1, len(ys) + 1)
    good = np.where((prec >= a.prec) & (np.arange(1, len(ys) + 1) >= 100))[0]
    ok_thr = float(ps[good[-1]]) if len(good) else 0.5
    print(f"P(reliable) cut-off for ~{a.prec:.0%} precision: {ok_thr:.2f} "
          f"(keeps {100 * (good[-1] + 1) / len(ys) if len(good) else float('nan'):.0f}% of frames)")

    final = makers[best]().fit(X, y, balanced_weights(y))
    out = HERE / "models" / "reliability.json"
    save_model(out, final, {"features": FEATURES, "label_threshold": a.thr, "ok_threshold": ok_thr,
                            "name": best, "n_videos": int(len(uv)),
                            "cv_balanced_acc": bal[best], "rule_balanced_acc": base})
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
