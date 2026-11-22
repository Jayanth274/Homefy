"""Phase 2 engine: webcam / video file -> landmarks -> measurements -> tracking status -> CSV.
Neutral measurements only. No emotion or clinical labels. No raw frames are saved."""
import argparse, csv, json, math, time
from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
MODEL = HERE / "models" / "face_landmarker.task"

# MediaPipe FaceMesh indices. Left/right = subject's side per MediaPipe numbering (verify in Phase 5).
EYE_OUT_R, EYE_OUT_L = 33, 263          # outer eye corners -> inter-ocular distance (scale unit)
LIP_UP, LIP_LOW, MOUTH_R, MOUTH_L = 13, 14, 61, 291
LID_UP_R, LID_LOW_R, LID_UP_L, LID_LOW_L = 159, 145, 386, 374
BROW_R, BROW_L = 105, 334
DRAW = [EYE_OUT_R, EYE_OUT_L, LIP_UP, LIP_LOW, MOUTH_R, MOUTH_L,
        LID_UP_R, LID_LOW_R, LID_UP_L, LID_LOW_L, BROW_R, BROW_L, 10, 152]

# Placeholder rule thresholds. Phase 3 compares them with a trained classifier.
CFG = dict(max_abs_yaw=45, max_abs_pitch=35, max_abs_roll=30,
           min_iod_px=40, min_brightness=50, min_sharpness=20, edge_margin_px=2)

FIELDS = ["frame", "t_ms", "proc_ms", "face", "status", "reasons",
          "pitch", "yaw", "roll", "iod_px", "brightness", "sharpness",
          "mouth_open", "mouth_width", "mouth_corner_raise",
          "eye_open_r", "eye_open_l", "brow_gap_r", "brow_gap_l"]


def head_angles(M):
    """Pitch, yaw, roll in degrees from the 4x4 facial transformation matrix."""
    R = np.array(M, dtype=float)[:3, :3]
    R = R / np.linalg.norm(R, axis=0)
    pitch = math.degrees(math.atan2(R[2, 1], R[2, 2]))
    yaw = math.degrees(math.atan2(-R[2, 0], math.hypot(R[2, 1], R[2, 2])))
    roll = math.degrees(math.atan2(R[1, 0], R[0, 0]))
    return pitch, yaw, roll


def measure(pts, frame_gray, M=None, cfg=CFG):
    """pts: (478,2) landmark pixels. Returns dict of measurements + status + reasons."""
    h, w = frame_gray.shape[:2]
    d = lambda a, b: float(np.linalg.norm(pts[a] - pts[b]))
    iod = d(EYE_OUT_R, EYE_OUT_L)
    row = {"face": 1, "iod_px": round(iod, 1)}
    if iod > 1e-6:
        row.update(
            mouth_open=d(LIP_UP, LIP_LOW) / iod,
            mouth_width=d(MOUTH_R, MOUTH_L) / iod,
            # positive = mouth corners higher (smaller y) than the lip centre
            mouth_corner_raise=((pts[LIP_UP][1] + pts[LIP_LOW][1]) / 2
                                - (pts[MOUTH_R][1] + pts[MOUTH_L][1]) / 2) / iod,
            eye_open_r=d(LID_UP_R, LID_LOW_R) / iod,
            eye_open_l=d(LID_UP_L, LID_LOW_L) / iod,
            brow_gap_r=d(BROW_R, LID_UP_R) / iod,
            brow_gap_l=d(BROW_L, LID_UP_L) / iod,
        )
    pitch = yaw = roll = None
    if M is not None:
        pitch, yaw, roll = head_angles(M)
        row.update(pitch=pitch, yaw=yaw, roll=roll)

    # image quality inside the face box
    x0, y0 = np.maximum(pts.min(0).astype(int), 0)
    x1, y1 = np.minimum(pts.max(0).astype(int), [w, h])
    reasons, cut = [], False
    if x1 - x0 > 8 and y1 - y0 > 8:
        crop = cv2.resize(frame_gray[y0:y1, x0:x1], (160, 160))
        row["brightness"] = float(crop.mean())
        row["sharpness"] = float(cv2.Laplacian(crop, cv2.CV_64F).var())
        if row["brightness"] < cfg["min_brightness"]: reasons.append("dark")
        if row["sharpness"] < cfg["min_sharpness"]: reasons.append("blurry")
    m = cfg["edge_margin_px"]
    if pts[:, 0].min() < -m or pts[:, 1].min() < -m or pts[:, 0].max() > w + m or pts[:, 1].max() > h + m:
        cut = True
        reasons.append("face_cut_off")
    if iod < cfg["min_iod_px"]: reasons.append("face_too_small")
    if yaw is not None and abs(yaw) > cfg["max_abs_yaw"]: reasons.append("head_turned")
    if pitch is not None and abs(pitch) > cfg["max_abs_pitch"]: reasons.append("head_up_down")
    if roll is not None and abs(roll) > cfg["max_abs_roll"]: reasons.append("head_tilted")
    row["status"] = "not_observable" if cut else ("low_confidence" if reasons else "ok")
    row["reasons"] = "|".join(reasons)
    return row


class FaceEngine:
    def __init__(self, model_path=MODEL):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
        self.mp = mp
        opts = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO, num_faces=1,
            output_face_blendshapes=False, output_facial_transformation_matrixes=True)
        self.lm = vision.FaceLandmarker.create_from_options(opts)
        self.last_ts = -1

    def process(self, frame_bgr, ts_ms):
        ts = max(int(ts_ms), self.last_ts + 1)       # must strictly increase
        self.last_ts = ts
        h, w = frame_bgr.shape[:2]
        t = time.perf_counter()
        img = self.mp.Image(image_format=self.mp.ImageFormat.SRGB,
                            data=cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        res = self.lm.detect_for_video(img, ts)
        row, pts = {"t_ms": ts}, None
        if res.face_landmarks:
            pts = np.array([[p.x * w, p.y * h] for p in res.face_landmarks[0]])
            M = res.facial_transformation_matrixes[0] if res.facial_transformation_matrixes else None
            row.update(measure(pts, cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY), M))
        else:
            row.update(face=0, status="not_observable", reasons="no_face")
        row["proc_ms"] = round((time.perf_counter() - t) * 1000, 1)
        return row, pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="cam", help="'cam' or a path to a video file")
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--out", default=None, help="session name (folder inside sessions/)")
    ap.add_argument("--show", action="store_true", help="show a live window (q to quit)")
    a = ap.parse_args()

    live = a.source == "cam"
    cap = cv2.VideoCapture(a.cam, cv2.CAP_DSHOW) if live else cv2.VideoCapture(a.source)
    if not cap.isOpened():
        raise SystemExit("Could not open " + ("webcam" if live else a.source))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    name = a.out or datetime.now().strftime("session_%Y%m%d_%H%M%S")
    out = HERE / "sessions" / name
    out.mkdir(parents=True, exist_ok=True)
    (out / "meta.json").write_text(json.dumps(
        {"source": a.source, "start": datetime.now().isoformat(timespec="seconds"),
         "source_fps": src_fps, "thresholds": CFG}, indent=2))

    eng = FaceEngine()
    status, reasons, n = Counter(), Counter(), 0
    t0 = time.perf_counter()
    with open(out / "frames.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=FIELDS, restval="", extrasaction="ignore")
        wr.writeheader()
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            ts = (time.perf_counter() - t0) * 1000 if live else n * 1000.0 / src_fps
            row, pts = eng.process(frame, ts)
            row["frame"] = n
            wr.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})
            status[row["status"]] += 1
            for r in filter(None, row["reasons"].split("|")): reasons[r] += 1
            n += 1
            if a.show:
                if pts is not None:
                    for i in DRAW:
                        cv2.circle(frame, (int(pts[i][0]), int(pts[i][1])), 3, (0, 255, 0), -1)
                col = {"ok": (0, 200, 0), "low_confidence": (0, 200, 255)}.get(row["status"], (0, 0, 255))
                cv2.putText(frame, f"{row['status']} {row['reasons']}", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
                if "yaw" in row:
                    cv2.putText(frame, f"pitch {row['pitch']:.0f} yaw {row['yaw']:.0f} roll {row['roll']:.0f}",
                                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.imshow("engine (q to quit)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    cap.release(); cv2.destroyAllWindows()
    secs = time.perf_counter() - t0
    print(f"\n{n} frames in {secs:.1f}s ({n / max(secs, 1e-9):.1f} fps) -> {out / 'frames.csv'}")
    for k, v in status.items():
        print(f"  {k:15s} {v:6d}  {100 * v / max(n, 1):5.1f}%")
    print("  reasons:", dict(reasons))


if __name__ == "__main__":
    main()
