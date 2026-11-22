"""Phase 4a: one command for a whole session (live camera or video file).
Writes sessions/<name>/frames.csv, events.csv, summary.json, meta.json (and session.mp4 with --record)."""
import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from engine import CFG, DRAW, FaceEngine
from events import detect_events, fmt_ms, summarize
from features import FEATURES, Temporal, reliability_features
from reliability_model import load_model

HERE = Path(__file__).resolve().parent
COLS = ["frame", "t_ms", "proc_ms", "face", "status", "reasons", "reliability_score", "pitch", "yaw", "roll",
        "iod_px", "brightness", "sharpness", "mouth_open", "mouth_width", "mouth_corner_raise",
        "eye_open_r", "eye_open_l", "brow_gap_r", "brow_gap_l"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="cam", help="'cam' or a path to a video file")
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--out", default=None, help="session name (folder inside sessions/)")
    ap.add_argument("--show", action="store_true", help="live window (q to quit)")
    ap.add_argument("--record", action="store_true", help="also save the raw video to the session folder (consent required)")
    ap.add_argument("--seconds", type=float, default=0, help="stop after N seconds (live)")
    ap.add_argument("--model", default=str(HERE / "models" / "reliability.json"))
    a = ap.parse_args()

    model, mmeta = load_model(a.model)
    ok_thr = float(mmeta["ok_threshold"])
    live = a.source == "cam"
    cap = cv2.VideoCapture(a.cam, cv2.CAP_DSHOW) if live else cv2.VideoCapture(a.source)
    if not cap.isOpened():
        raise SystemExit("Could not open " + ("webcam" if live else a.source))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    name = a.out or datetime.now().strftime("session_%Y%m%d_%H%M%S")
    out = HERE / "sessions" / name
    out.mkdir(parents=True, exist_ok=True)

    eng, tmp, rows, writer, n = FaceEngine(), Temporal(), [], None, 0
    t0 = time.perf_counter()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            ts = (time.perf_counter() - t0) * 1000 if live else n * 1000.0 / src_fps
            if a.seconds and live and ts > a.seconds * 1000:
                break
            if a.record:
                if writer is None:
                    h, w = frame.shape[:2]
                    writer = cv2.VideoWriter(str(out / "session.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                             src_fps if src_fps >= 5 else 30.0, (w, h))
                writer.write(frame)
            row, pts = eng.process(frame, ts)
            jit = tmp.update(pts)
            row["frame"] = n
            if pts is not None and row["status"] != "not_observable":
                f = reliability_features(row, pts, frame.shape[:2], jit)
                score = float(model.predict_proba(np.array([[f[k] for k in FEATURES]], float))[0])
                row["reliability_score"] = round(score, 3)
                low = score < ok_thr
                row["status"] = "low_confidence" if low else "ok"
                row["reasons"] = "|".join([r for r in (row["reasons"], "model_score_low" if low else "") if r])
            rows.append({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items() if k in COLS})
            n += 1
            if a.show:
                if pts is not None:
                    for i in DRAW:
                        cv2.circle(frame, (int(pts[i][0]), int(pts[i][1])), 3, (0, 255, 0), -1)
                col = {"ok": (0, 200, 0), "low_confidence": (0, 200, 255)}.get(row["status"], (0, 0, 255))
                cv2.putText(frame, f"{row['status']} score {row.get('reliability_score', '-')}", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
                cv2.imshow("observe (q to stop)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        print("stopped by user")
    finally:
        cap.release(); cv2.destroyAllWindows()
        if writer is not None:
            writer.release()
    if not rows:
        raise SystemExit("no frames processed")

    df = pd.DataFrame(rows).reindex(columns=COLS)
    df.to_csv(out / "frames.csv", index=False)
    ev = detect_events(df)
    ev.to_csv(out / "events.csv", index=False)
    meta = {"source": a.source, "start": datetime.now().isoformat(timespec="seconds"), "source_fps": src_fps,
            "recorded_video": bool(a.record), "reliability_model": mmeta.get("name"),
            "ok_threshold": ok_thr, "label_threshold": mmeta.get("label_threshold"), "rule_thresholds": CFG}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    sm = summarize(df, ev, meta)
    (out / "summary.json").write_text(json.dumps(sm, indent=2))

    secs = time.perf_counter() - t0
    print(f"\n{n} frames, {sm['duration_s']}s of video, processed in {secs:.1f}s -> {out}")
    print("status share %:", sm["status_share_pct"])
    print(f"{len(ev)} events")
    for _, e in ev.head(15).iterrows():
        print(f"  {fmt_ms(e.start_ms)}-{fmt_ms(e.end_ms)}  {e['type']} {e.side}  change {e.change:+.3f}  conf {e.confidence}")
    if len(ev) > 15:
        print(f"  ... {len(ev) - 15} more in events.csv")


if __name__ == "__main__":
    main()
