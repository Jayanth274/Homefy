import math, time
import cv2, numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

opts = vision.FaceLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path="models/face_landmarker.task"),
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=True,
)
landmarker = vision.FaceLandmarker.create_from_options(opts)

KEY = {13: "lip-up", 14: "lip-low", 61: "mouthL", 291: "mouthR", 10: "top", 152: "chin",
       159: "eyeR-up", 145: "eyeR-low", 386: "eyeL-up", 374: "eyeL-low",
       105: "browR", 334: "browL"}

def angles(M):
    R = np.array(M)[:3, :3]
    R = R / np.linalg.norm(R, axis=0)
    pitch = math.degrees(math.atan2(R[2, 1], R[2, 2]))
    yaw = math.degrees(math.atan2(-R[2, 0], math.hypot(R[2, 1], R[2, 2])))
    roll = math.degrees(math.atan2(R[1, 0], R[0, 0]))
    return pitch, yaw, roll

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    raise SystemExit("Webcam did not open")

t0 = time.perf_counter(); tf = t0; last_ts = -1; n = 0; fps = 0.0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    ts = max(int((time.perf_counter() - t0) * 1000), last_ts + 1)
    last_ts = ts
    img = mp.Image(image_format=mp.ImageFormat.SRGB,
                   data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    res = landmarker.detect_for_video(img, ts)

    h, w = frame.shape[:2]
    txt = "no face"
    if res.face_landmarks:
        lm = res.face_landmarks[0]
        for i in KEY:
            cv2.circle(frame, (int(lm[i].x * w), int(lm[i].y * h)), 3, (0, 255, 0), -1)
        txt = ""
        if res.facial_transformation_matrixes:
            p, y, r = angles(res.facial_transformation_matrixes[0])
            txt = f"pitch {p:.0f} yaw {y:.0f} roll {r:.0f} | "
        txt += f"lip gap {abs(lm[13].y - lm[14].y) * h:.1f}px"

    n += 1
    now = time.perf_counter()
    if now - tf >= 1:
        fps = n / (now - tf); n = 0; tf = now
    cv2.putText(frame, f"{txt}  fps {fps:.0f}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.imshow("smoke test (q to quit)", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
