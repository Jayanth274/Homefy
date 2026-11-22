"""Turn per-frame measurements into neutral, timestamped observations.
Describes geometry only. No emotion labels, no clinical interpretation.
Baseline = median of the session's own 'ok' frames. Thresholds below are unvalidated defaults."""
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd

GAP_MS = 150          # active runs separated by less than this are merged
PITCH_NOTE_DEG = 15   # brow measures are distorted when the head is pitched more than this

SPECS = [
    dict(type="mouth_opening", col="mouth_open", side="", mode="above_abs", floor=0.06, min_ms=200,
         label="Mouth opening above session baseline"),
    dict(type="mouth_corner_raise", col="mouth_corner_raise", side="", mode="above_abs", floor=0.03, min_ms=200,
         label="Mouth corners raised relative to session baseline"),
    dict(type="mouth_corner_lowering", col="mouth_corner_raise", side="", mode="below_abs", floor=0.03, min_ms=200,
         label="Mouth corners lowered relative to session baseline"),
    dict(type="eyelid_closure", col="eye_open_r", side="right", mode="below_rel", rel=0.5, min_ms=50,
         label="Eye opening below half of session baseline"),
    dict(type="eyelid_closure", col="eye_open_l", side="left", mode="below_rel", rel=0.5, min_ms=50,
         label="Eye opening below half of session baseline"),
    dict(type="eyebrow_raise", col="brow_gap_r", side="right", mode="above_rel", rel=1.15, min_ms=200,
         label="Brow-to-eyelid distance above session baseline"),
    dict(type="eyebrow_raise", col="brow_gap_l", side="left", mode="above_rel", rel=1.15, min_ms=200,
         label="Brow-to-eyelid distance above session baseline"),
    dict(type="eyebrow_lowering", col="brow_gap_r", side="right", mode="below_rel", rel=0.88, min_ms=200,
         label="Brow-to-eyelid distance below session baseline"),
    dict(type="eyebrow_lowering", col="brow_gap_l", side="left", mode="below_rel", rel=0.88, min_ms=200,
         label="Brow-to-eyelid distance below session baseline"),
]
MEASURE_COLS = ["mouth_open", "mouth_width", "mouth_corner_raise", "eye_open_r", "eye_open_l",
                "brow_gap_r", "brow_gap_l"]


def fmt_ms(ms):
    s = ms / 1000.0
    return f"{int(s // 60):02d}:{s % 60:05.2f}"


def baseline(ok_df, col):
    v = ok_df[col].dropna().to_numpy(float)
    if len(v) < 30:
        return None
    b = float(np.median(v))
    return b, float(1.4826 * np.median(np.abs(v - b)))


def runs(active, t, dt, min_ms):
    idx = np.flatnonzero(active)
    if len(idx) == 0:
        return []
    out, s, e = [], idx[0], idx[0]
    for i in idx[1:]:
        if t[i] - t[e] <= GAP_MS:
            e = i
        else:
            out.append((s, e)); s = e = i
    out.append((s, e))
    return [(s, e) for s, e in out if t[e] - t[s] + dt >= min_ms]


def detect_events(df):
    df = df.reset_index(drop=True)
    t = df["t_ms"].to_numpy(float)
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 33.0
    ok = (df["status"] == "ok").to_numpy()
    ok_df = df[ok]
    has_score = "reliability_score" in df and df["reliability_score"].notna().any()
    events = []
    for sp in SPECS:
        col = sp["col"]
        if col not in df:
            continue
        bs = baseline(ok_df, col)
        if bs is None:
            continue
        b, s = bs
        v = df[col].to_numpy(float)
        m = sp["mode"]
        if m == "above_abs":   thr = b + max(3 * s, sp["floor"]); cond = v > thr
        elif m == "below_abs": thr = b - max(3 * s, sp["floor"]); cond = v < thr
        elif m == "above_rel": thr = max(b * sp["rel"], b + 3 * s); cond = v > thr
        else:                  thr = min(b * sp["rel"], b - 3 * s); cond = v < thr
        active = ok & cond & ~np.isnan(v)
        for a, z in runs(active, t, dt, sp["min_ms"]):
            seg = slice(a, z + 1)
            k = a + int(np.nanargmax(np.abs(v[seg] - b)))
            dur = float(t[z] - t[a] + dt)
            typ = sp["type"]
            if typ == "eyelid_closure":
                typ += "_brief" if dur < 500 else "_sustained"
            note = ""
            if typ.startswith("eyebrow") and "pitch" in df and np.nanmedian(np.abs(df["pitch"].to_numpy(float)[seg])) > PITCH_NOTE_DEG:
                note = f"head pitched more than {PITCH_NOTE_DEG} deg; brow measure less reliable"
            events.append({
                "type": typ, "side": sp["side"], "description": sp["label"],
                "start_ms": int(t[a]), "end_ms": int(t[z]), "duration_ms": int(round(dur)),
                "start_frame": int(df["frame"].iloc[a]), "peak_frame": int(df["frame"].iloc[k]),
                "peak_ms": int(t[k]), "baseline": round(b, 4), "peak_value": round(float(v[k]), 4),
                "change": round(float(v[k] - b), 4),
                "confidence": round(float(df["reliability_score"].iloc[seg].mean()), 3) if has_score else "",
                "note": note})
    mo = [(e["start_ms"], e["end_ms"]) for e in events if e["type"] == "mouth_opening"]
    for e in events:
        if e["type"].startswith("mouth_corner") and any(e["start_ms"] <= z + GAP_MS and e["end_ms"] >= a - GAP_MS for a, z in mo):
            e["note"] = "occurs during mouth opening; corner measure may be less reliable while the jaw is open"
    cols = ["type", "side", "description", "start_ms", "end_ms", "duration_ms", "start_frame", "peak_frame",
            "peak_ms", "baseline", "peak_value", "change", "confidence", "note"]
    ev = pd.DataFrame(events, columns=cols)
    return ev.sort_values("start_ms").reset_index(drop=True) if len(ev) else ev


def summarize(df, ev, meta=None):
    n = len(df)
    t = df["t_ms"].to_numpy(float)
    secs = float((t.max() - t.min()) / 1000) if n > 1 else 0.0
    ok_df = df[df["status"] == "ok"]
    s = {"frames": n, "duration_s": round(secs, 2),
         "status_share_pct": {k: round(100 * float(v), 1) for k, v in df["status"].value_counts(normalize=True).items()},
         "baselines": {}, "events": {}, "notes": [
             "Observations are geometric measurements of facial landmarks relative to this session's own baseline.",
             "No emotion, intent or clinical condition is inferred.",
             "Baseline is the median of 'ok' frames in this session; it is not a clinical neutral face.",
             "Event thresholds are unvalidated defaults. Reliability score is a model score, not a calibrated probability.",
             "Eyebrow measures are distorted by head pitch. Reliability model was trained on adult video."]}
    for c in MEASURE_COLS:
        b = baseline(ok_df, c) if c in df else None
        if b:
            s["baselines"][c] = {"median": round(b[0], 4), "robust_sd": round(b[1], 4),
                                 "min": round(float(ok_df[c].min()), 4), "max": round(float(ok_df[c].max()), 4)}
    if len(ev):
        for (typ, side), g in ev.groupby(["type", "side"]):
            s["events"][f"{typ}{'_' + side if side else ''}"] = {
                "count": int(len(g)), "total_s": round(float(g["duration_ms"].sum()) / 1000, 2)}
        if secs > 0:
            for side in ("right", "left"):
                nb = int(((ev["type"] == "eyelid_closure_brief") & (ev["side"] == side)).sum())
                s[f"brief_eyelid_closures_per_min_{side}"] = round(nb / (secs / 60), 1)
    if "reliability_score" in df and df["reliability_score"].notna().any():
        s["mean_reliability_score_ok_frames"] = round(float(ok_df["reliability_score"].mean()), 3) if len(ok_df) else None
    for side, c in (("right", "eye_open_r"), ("left", "eye_open_l")):
        bb = baseline(ok_df, c) if c in df else None
        if bb and bb[0] - 3 * bb[1] <= 0:
            s["notes"].append(f"Eyelid closure not assessable on the {side}: eye-opening baseline too noisy in this session.")
    if meta:
        s["meta"] = meta
    return s


if __name__ == "__main__":
    folder = Path(sys.argv[1])
    d = pd.read_csv(folder / "frames.csv")
    e = detect_events(d)
    e.to_csv(folder / "events.csv", index=False)
    sm = summarize(d, e)
    (folder / "summary.json").write_text(json.dumps(sm, indent=2))
    print(json.dumps({k: v for k, v in sm.items() if k != "notes"}, indent=2))
    print(f"\n{len(e)} events -> {folder / 'events.csv'}")
