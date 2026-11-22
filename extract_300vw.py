"""Phase 3a: run the engine over 300VW videos and compare MediaPipe landmarks with the dataset's
ground-truth 68 points. Output: one CSV per video in data/feats/ (safe to stop and re-run)."""
import argparse, time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from engine import FaceEngine
from features import ANCHORS_MP, Temporal, reliability_features

HERE = Path(__file__).resolve().parent
# iBUG-68 index  ->  MediaPipe index  (same anatomical point). Subject's right = image left.
GT_ANCHORS = [36, 39, 42, 45, 48, 54, 30, 8, 27]
#             MP: 33, 133, 362, 263, 61, 291, 1, 152, 168   (ANCHORS_MP, same order)


def read_pts(path):
    lines = [l.strip() for l in Path(path).read_text().splitlines()]
    a, b = lines.index("{"), lines.index("}")
    arr = np.array([[float(v) for v in l.split()[:2]] for l in lines[a + 1:b]])
    if arr.shape != (68, 2):
        raise ValueError(f"{path}: expected 68 points, got {arr.shape}")
    return arr


def process_video(vid, stride):
    annot = {int(p.stem): p for p in (vid.parent / "annot").glob("*.pts")}
    cap = cv2.VideoCapture(str(vid))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {vid}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    eng, tmp, rows, i = FaceEngine(), Temporal(), [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        row, pts = eng.process(frame, i * 1000.0 / fps)   # every frame goes in, so tracking behaves as in live use
        jit = tmp.update(pts)
        if i % stride == 0 and (i + 1) in annot:           # annotation files are numbered from 1
            rec = {"video": vid.parent.name, "frame": i, "face": row["face"],
                   "rule_status": row["status"], "rule_reasons": row["reasons"], "err": float("nan")}
            if pts is not None:
                gt = read_pts(annot[i + 1])
                gt_iod = float(np.linalg.norm(gt[36] - gt[45]))
                rec["err"] = float(np.linalg.norm(pts[ANCHORS_MP] - gt[GT_ANCHORS], axis=1).mean() / gt_iod)
                rec.update(reliability_features(row, pts, frame.shape[:2], jit))
            rows.append(rec)
        i += 1
    cap.release()
    eng.lm.close()
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(HERE / "data" / "300VW"))
    ap.add_argument("--out", default=str(HERE / "data" / "feats"))
    ap.add_argument("--stride", type=int, default=2, help="record every Nth frame")
    ap.add_argument("--limit", type=int, default=0, help="only the first N videos (quick test)")
    a = ap.parse_args()

    vids = sorted(Path(a.root).rglob("vid.avi"))
    if a.limit:
        vids = vids[:a.limit]
    if not vids:
        raise SystemExit(f"no vid.avi found under {a.root}")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    print(f"{len(vids)} videos")
    t_all = time.perf_counter()
    for k, v in enumerate(vids, 1):
        dest = out / f"{v.parent.name}.csv"
        if dest.exists():
            print(f"[{k}/{len(vids)}] {v.parent.name}: already done, skipping")
            continue
        t = time.perf_counter()
        df = process_video(v, a.stride)
        df.to_csv(dest, index=False)
        det = df["face"].mean() if len(df) else float("nan")
        med = df["err"].median()
        print(f"[{k}/{len(vids)}] {v.parent.name}: {len(df)} rows, face found {100 * det:.0f}%, "
              f"median err {med:.3f}, {time.perf_counter() - t:.0f}s")
    print(f"done in {(time.perf_counter() - t_all) / 60:.1f} min")


if __name__ == "__main__":
    main()
