import sys, json, cv2
from pathlib import Path
d = Path("sessions") / sys.argv[1]
s = json.loads((d / "summary.json").read_text(encoding="utf-8"))
vid = d / "session.mp4"
cap = cv2.VideoCapture(str(vid))
vfps = cap.get(cv2.CAP_PROP_FPS)
real = s["frames"] / s["duration_s"]
print(f"video fps {vfps:.2f}, real fps {real:.2f}")
if abs(real - vfps) / vfps < 0.1:
    print("ok, nothing to fix"); sys.exit()
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
tmp = d / "session_fixed.mp4"
wr = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), real, (w, h))
k = 0
while True:
    ok, f = cap.read()
    if not ok: break
    wr.write(f); k += 1
wr.release(); cap.release()
if k != s["frames"]: print(f"WARNING: wrote {k} frames, expected {s['frames']}")
vid.unlink(); tmp.rename(vid)
print(f"rewrote {k} frames at {real:.2f} fps")
