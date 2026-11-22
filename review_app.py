"""Phase 4b: local review screen.  Run:  python -m streamlit run review_app.py
Shows one processed session: summary, timeline, events, a video frame at each event, and clinician notes.
Notes go to a separate file (notes.csv) and are never mixed into the system's observations."""
import csv
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from events import fmt_ms

SESS = Path(__file__).resolve().parent / "sessions"
st.set_page_config(page_title="Facial movement review", layout="wide")
st.title("Facial movement observation: session review")
st.caption("Neutral geometric measurements relative to this session's own baseline. "
           "No emotion or clinical inference is made. All data stays on this computer.")

names = sorted([p.name for p in SESS.glob("*") if (p / "frames.csv").exists() and (p / "events.csv").exists()],
               reverse=True)
if not names:
    st.warning("No processed sessions found. Run observe.py first.")
    st.stop()
name = st.sidebar.selectbox("Session", names)
folder = SESS / name

frames = pd.read_csv(folder / "frames.csv")
events = pd.read_csv(folder / "events.csv")
summary = json.loads((folder / "summary.json").read_text()) if (folder / "summary.json").exists() else {}
meta = json.loads((folder / "meta.json").read_text()) if (folder / "meta.json").exists() else {}

# ---------- summary ----------
share = summary.get("status_share_pct", {})
dur = summary.get("duration_s", float(frames["t_ms"].max() - frames["t_ms"].min()) / 1000)
c = st.columns(6)
c[0].metric("Duration", fmt_ms(dur * 1000))
c[1].metric("Frames", len(frames))
c[2].metric("Average fps", f"{len(frames) / max(dur, 1e-9):.1f}")
c[3].metric("ok frames", f"{share.get('ok', 0)}%")
c[4].metric("low confidence", f"{share.get('low_confidence', 0)}%")
c[5].metric("not observable", f"{share.get('not_observable', 0)}%")
if len(frames) / max(dur, 1e-9) < 15:
    st.warning("Low frame rate (under 15 fps). Brief movements such as blinks may be missed or mis-measured. "
               "Improve lighting and re-record.")
with st.expander("How to read this (limits)"):
    for n in summary.get("notes", []):
        st.write("- " + n)
    st.write("- Confidence is the reliability model's score for the frames in an event (0 to 1). "
             "It is not the probability that a movement really happened.")

# ---------- events (filtered) ----------
ev = events.copy()
if len(ev):
    types = sorted(ev["type"].unique())
    pick_types = st.sidebar.multiselect("Event types", types, default=types)
    ev = ev[ev["type"].isin(pick_types)].reset_index(drop=True)
    ev["start"] = ev["start_ms"].map(fmt_ms)
    ev["end"] = ev["end_ms"].map(fmt_ms)

# ---------- timeline ----------
st.subheader("Timeline")
group = st.radio("Show", ["Measurements (units of eye-to-eye distance)", "Head angles (degrees)"], horizontal=True)
if group.startswith("Measurements"):
    cols = [x for x in ["mouth_open", "mouth_corner_raise", "eye_open_r", "eye_open_l", "brow_gap_r", "brow_gap_l"]
            if x in frames]
    default = [x for x in ["mouth_open", "eye_open_r", "brow_gap_r"] if x in cols]
else:
    cols = [x for x in ["pitch", "yaw", "roll"] if x in frames]
    default = cols
sel = st.multiselect("Lines", cols, default=default)
t = frames["t_ms"].to_numpy(float) / 1000.0
fig = go.Figure()
for col in sel:
    fig.add_trace(go.Scatter(x=t, y=frames[col], mode="lines", name=col))


def spans(mask):
    m = np.concatenate([[0], mask.astype(int), [0]])
    d = np.diff(m)
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1))


for status, color in (("low_confidence", "rgba(150,150,150,0.30)"), ("not_observable", "rgba(220,0,0,0.25)")):
    for a, z in spans((frames["status"] == status).to_numpy())[:300]:
        fig.add_vrect(x0=t[a], x1=t[z], fillcolor=color, line_width=0, layer="below")
if len(ev):
    for r in ev.itertuples():
        fig.add_vrect(x0=r.start_ms / 1000, x1=max(r.end_ms, r.start_ms + 30) / 1000,
                      fillcolor="rgba(0,100,255,0.18)", line_width=0, layer="below")
fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="seconds")
st.plotly_chart(fig)
st.caption("Grey = low confidence, red = not observable, blue = detected event. "
           + ("Head-angle axis names come from the MediaPipe matrix and have not been verified."
              if group.startswith("Head") else ""))

# ---------- events table ----------
st.subheader(f"Events ({len(ev)})")
if len(ev):
    st.dataframe(ev[["start", "end", "type", "side", "duration_ms", "change", "confidence", "note"]],
                 hide_index=True)
    st.download_button("Download events.csv", events.to_csv(index=False), file_name=f"{name}_events.csv")
else:
    st.info("No events detected in this session.")

# ---------- frame at event ----------
st.subheader("Video frame at an event")
video = folder / "session.mp4"
if not video.exists() and meta.get("source") not in (None, "cam") and Path(meta["source"]).exists():
    video = Path(meta["source"])
if not len(ev):
    pass
elif not video.exists():
    st.info("No video was saved for this session (default, for privacy). Re-run observe.py with --record to review frames.")
else:
    labels = [f"{r.start}  {r.type} {r.side}" for r in ev.itertuples()]
    k = st.selectbox("Event", range(len(ev)), format_func=lambda i: labels[i])
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(ev.loc[k, "peak_frame"]))
    ok, img = cap.read()
    cap.release()
    if ok:
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                 caption=f"Frame {int(ev.loc[k, 'peak_frame'])} at the peak of: {ev.loc[k, 'description']}")
    else:
        st.warning("Could not read that frame from the video.")

# ---------- clinician notes (separate file) ----------
st.subheader("Clinician notes (stored separately)")
st.caption(f"Saved to {folder / 'notes.csv'}. Never mixed into the system's observations.")
notes_path = folder / "notes.csv"
with st.form("note", clear_on_submit=True):
    at = st.text_input("Time in recording (mm:ss, optional)")
    txt = st.text_area("Note")
    if st.form_submit_button("Save note") and txt.strip():
        new = not notes_path.exists()
        with open(notes_path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["saved_at", "recording_time", "note"])
            w.writerow([datetime.now().isoformat(timespec="seconds"), at.strip(), txt.strip()])
        st.success("Note saved.")
if notes_path.exists():
    st.dataframe(pd.read_csv(notes_path), hide_index=True)
