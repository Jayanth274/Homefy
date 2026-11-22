# Facial Movement Observation (Saachivya Medical, PS3)

A local, offline tool that tracks facial landmarks from a live camera or a saved video and writes **neutral, timestamped observations of facial movement, each with a confidence value**.

It describes geometry only. It does not infer emotion, intent, or any clinical condition. Interpretation is left to a qualified person.

## What it does
1. Reads frames from a webcam or a video file.
2. Locates 478 facial landmarks per frame (MediaPipe Face Landmarker).
3. Measures mouth opening, mouth width, mouth-corner height, eyelid opening (left and right), and brow-to-eyelid distance (left and right).
4. Uses a **trained reliability model** to score how trustworthy the landmarks are in each frame. Frames that are not reliable are marked `not_observable` and are left out of baselines and events.
5. Compares each measurement with that session's own baseline and writes events such as mouth opening, mouth-corner raise or lowering, eyelid closure (brief or sustained), and eyebrow raise or lowering. Each event has a start and end time, a size of change, and a confidence.

The baseline is the session's own typical (median) value on reliable frames. It is not a clinical "neutral face".

## The reliability model
- **Purpose:** predict, from capture conditions alone, whether the landmark detector is likely to be accurate on a frame.
- **Inputs:** head angles (pitch, yaw, roll), face size, distance of the face from the frame edge, brightness, and sharpness. Facial movement itself is deliberately not an input, so smiling, talking or opening the mouth cannot lower the score.
- **Training:** gradient-boosted decision trees written in NumPy, trained on the public 300VW video benchmark with its annotated landmarks as ground truth. A frame is labelled unreliable when the detector's landmark error is large relative to the distance between the eyes.
- **Evaluation:** cross-validated by video, so every test video was scored by a model that never saw it, and compared with a simple rule-based baseline.
- **Output:** a score per frame, and a cut-off that separates `ok` from `low` frames.
- The score is a model score, **not a calibrated probability**.

## Output
Each run writes a session folder `sessions/<name>/`:

A sample output from a low-light webcam recording is in `examples/low_light_demo/` (landmark measurements only, no video).

| File | Content |
|---|---|
| `frames.csv` | per-frame measurements, status and reliability score |
| `events.csv` | observations with timing, baseline, peak, change, confidence and notes |
| `summary.json` | status shares, baselines, event counts and caveat notes |
| `meta.json` | source, model and thresholds used |
| `session.mp4` | the raw video, only when `--record` is used (consent required) |

## Setup (Windows)

    python -m venv .venv
    .venv\Scripts\activate
    python -m pip install -r requirements.txt

The MediaPipe landmark model (`models/face_landmarker.task`) and the trained reliability model (`models/reliability.json`) are included.

## Run

    python observe.py --source cam --show --seconds 60 --record --out live1
    python fix_fps.py live1
    python observe.py --source path\to\clip.mp4 --out myclip
    python -m streamlit run review_app.py

- `--record` saves the raw video. Use it only with the person's consent.
- `fix_fps.py` rewrites a recorded video at its true frame rate. Webcams often report a higher frame rate than they deliver in low light, which would otherwise make the saved video play too fast.
- The review app shows events on a timeline, jumps the video to each event, and stores clinician notes in a separate `notes.csv`.

## Limits
- Event thresholds are default values and have not yet been validated against scripted recordings. Treat events as observations for a person to review, not as detections with known accuracy.
- Low light slows webcams and makes short events weak evidence. Use bright, even light and face the camera.
- Eyelid closure needs a stable baseline. When the baseline is too noisy, the tool says "not assessable" in `summary.json` instead of reporting zero.
- Eyebrow measures are distorted when the head is tilted up or down; such events carry a note. Mouth-corner events that overlap a mouth-opening event also carry a note, because jaw movement can affect that measure.
- The reliability model was trained on adult video and has not been tested on children.
- Head-angle direction conventions come from the landmarker and were not independently verified.

## Privacy
Everything runs offline on the local machine. No video is saved unless `--record` is used. Clinician notes are stored separately from measurements.

## Files
| File | Role |
|---|---|
| `observe.py` | runs a session on a camera or video |
| `engine.py` | landmark engine |
| `features.py` | measurements and reliability features |
| `events.py` | event rules and session summary |
| `review_app.py` | Streamlit review interface |
| `fix_fps.py` | repairs the frame rate of recorded video |
| `extract_300vw.py`, `reliability_model.py`, `train_reliability.py` | build and train the reliability model |
| `smoke_test.py` | quick camera and landmark check |

## Data and citations
The 300VW dataset is for research use and is not included in this repository. The trained model weights are.

- Shen et al., "The First Facial Landmark Tracking in-the-Wild Challenge: Benchmark and Results", ICCV Workshops 2015.
- Chrysos et al., "Offline Deformable Face Tracking in Arbitrary Videos", ICCV Workshops 2015.
- Tzimiropoulos, "Project-Out Cascaded Regression with an Application to Face Alignment", CVPR 2015.
- Google MediaPipe Face Landmarker (Tasks API).