"""Features the reliability classifier sees. Shared by training (extract_300vw.py) and the live app."""
import numpy as np

# MediaPipe anchor landmarks (eye corners, mouth corners, nose tip, chin, nose bridge)
ANCHORS_MP = [33, 133, 362, 263, 61, 291, 1, 152, 168]

# Conditions only (pose, size, position, light, sharpness). Deliberately NOT the mouth/eye/brow
# measurements or jitter: the reliability gate must not penalise the movement being observed.
FEATURES = ["pitch", "yaw", "roll", "iod_px", "face_area", "edge_gap", "brightness", "sharpness"]


class Temporal:
    """Keeps the last two landmark sets. jitter = size of the 2nd difference of anchor points
    (high-frequency wobble, so smooth head motion is mostly cancelled), divided by eye distance."""
    def __init__(self):
        self.p1 = self.p2 = None

    def update(self, pts):
        if pts is None:
            self.p1 = self.p2 = None
            return float("nan")
        a = pts[ANCHORS_MP]
        jit = float("nan")
        if self.p1 is not None and self.p2 is not None:
            iod = np.linalg.norm(pts[33] - pts[263])
            jit = float(np.linalg.norm(a - 2 * self.p1 + self.p2, axis=1).mean() / max(iod, 1e-6))
        self.p2, self.p1 = self.p1, a
        return jit


def reliability_features(row, pts, frame_shape, jitter):
    """row: dict from engine.measure(); pts: (478,2) pixels; frame_shape: (h, w)."""
    h, w = frame_shape
    x0, y0 = pts.min(0)
    x1, y1 = pts.max(0)
    iod = max(float(row.get("iod_px", 0.0)), 1e-6)
    f = {k: row.get(k, float("nan")) for k in FEATURES}
    f["face_area"] = float((x1 - x0) * (y1 - y0) / (w * h))
    f["edge_gap"] = float(min(x0, y0, w - x1, h - y1) / iod)   # negative = landmarks outside the frame
    f["jitter"] = jitter
    return f
