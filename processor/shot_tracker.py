# processor/shot_tracker.py
from __future__ import annotations

import os
import math
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .rim_detector import get_rim_center, get_rim_info

log = logging.getLogger(__name__)
try:
    cv2.setNumThreads(0)
except Exception:
    pass
try:
    cv2.ocl.setUseOpenCL(False)
except Exception:
    pass

# ----------------------------
# YOLO (optional)
# ----------------------------
try:
    from ultralytics import YOLO  # type: ignore
except Exception:
    YOLO = None  # type: ignore

_YOLO = None

def _load_yolo():
    global _YOLO
    if _YOLO is not None:
        return _YOLO
    path = os.getenv("YOLO_MODEL_PATH", "")
    if not path or YOLO is None or not Path(path).exists():
        log.info("shot_tracker: YOLO=off (model=%s available=%s)", path, bool(YOLO))
        _YOLO = None
        return None
    try:
        _YOLO = YOLO(path)
        log.info("shot_tracker: YOLO=on (model=%s)", path)
    except Exception as e:
        log.warning("shot_tracker: YOLO load failed: %s", e)
        _YOLO = None
    return _YOLO

YOLO_CONF = float(os.getenv("YOLO_CONF", "0.30"))
YOLO_IOU = float(os.getenv("YOLO_IOU", "0.50"))
YOLO_IMGSZ = int(os.getenv("YOLO_IMGSZ", "640"))
YOLO_DOWNSCALE_MAX_W = int(os.getenv("YOLO_DOWNSCALE_MAX_W", "960"))

# ----------------------------
# Tunables (env-backed)
# ----------------------------
TRACK_STRIDE = int(os.getenv("TRACK_STRIDE", "2"))            # read every Nth frame
MAX_GAP_FRAMES = int(os.getenv("MAX_GAP_FRAMES", "6"))        # gap fill tolerance
EMA_ALPHA = float(os.getenv("EMA_ALPHA", "0.45"))             # smoothing on y
VY_UP_THRESH = float(os.getenv("VY_UP_THRESH", "-18.0"))      # upward velocity to mark release
MIN_DY_PIX = float(os.getenv("MIN_DY_PIX", "80.0"))           # min vertical excursion for a valid arc
MIN_APEX_SEP_FRAMES = int(os.getenv("MIN_APEX_SEP_FRAMES", "8"))
RIM_X_MARGIN = float(os.getenv("RIM_X_MARGIN", "1.4"))        # rim radius multiplier for X gating
MADE_HOLD_FRAMES = int(os.getenv("MADE_HOLD_FRAMES", "10"))   # frames ball must stay below rim after crossing
CAND_PRE_SEC = float(os.getenv("CAND_PRE_SEC", "0.40"))
CAND_POST_SEC = float(os.getenv("CAND_POST_SEC", "1.40"))
DEDUP_SEC = float(os.getenv("DEDUP_SEC", "1.8"))

@dataclass
class TrackPoint:
    f: int
    x: float
    y: float

# ----------------------------
# Ball detection helpers
# ----------------------------

def _ball_center_yolo(frame) -> Optional[Tuple[float, float]]:
    model = _load_yolo()
    if model is None:
        return None
    h, w = frame.shape[:2]
    scale = 1.0
    small = frame
    if w > YOLO_DOWNSCALE_MAX_W:
        scale = YOLO_DOWNSCALE_MAX_W / float(w)
        small = cv2.resize(frame, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    try:
        res = model.predict(small, conf=YOLO_CONF, iou=YOLO_IOU, imgsz=YOLO_IMGSZ, verbose=False)[0]
    except Exception:
        return None
    names = getattr(res, "names", {}) or {}
    best = None
    best_conf = -1.0
    for b in getattr(res, "boxes", []) or []:
        try:
            cls_id = int(b.cls)
            nm = str(names.get(cls_id, "")).lower()
            if nm not in {"sports ball", "basketball", "ball"}:
                continue
            conf = float(b.conf[0].item() if hasattr(b.conf[0], "item") else b.conf[0])
            if conf <= best_conf:
                continue
            x1, y1, x2, y2 = b.xyxy[0].cpu().numpy()
            cx_s, cy_s = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            if scale != 1.0:
                best = (float(cx_s / scale), float(cy_s / scale))
            else:
                best = (float(cx_s), float(cy_s))
            best_conf = conf
        except Exception:
            continue
    return best

def _ball_center_orange(frame) -> Optional[Tuple[float, float]]:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    low = np.array([5, 80, 60], dtype=np.uint8)
    high = np.array([25, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, low, high)
    mask = cv2.medianBlur(mask, 5)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_score = 0.0
    for c in contours:
        area = cv2.contourArea(c)
        if area < 40 or area > 4000:
            continue
        (bx, by), r = cv2.minEnclosingCircle(c)
        if r < 3 or r > 40:
            continue
        per = cv2.arcLength(c, True)
        circ = 0.0 if per == 0 else 4 * math.pi * area / (per * per)
        score = area * circ
        if score > best_score:
            best_score = score
            best = (float(bx), float(by))
    return best

def _detect_ball_center(frame) -> Optional[Tuple[float, float]]:
    c = _ball_center_yolo(frame)
    if c is not None:
        return c
    return _ball_center_orange(frame)

# ----------------------------
# Core pipeline
# ----------------------------

def _video_meta(cap: cv2.VideoCapture) -> Tuple[float, int, int, int]:
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    return fps, total, w, h

def _track_ball(video_path: str) -> Tuple[List[TrackPoint], float, int, Tuple[int,int,int]]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [], 30.0, 0, (0,0,0)
    fps, total, w, h = _video_meta(cap)
    ok, first = cap.read()
    if not ok or first is None:
        cap.release()
        return [], fps, total, (0,0,0)
    try:
        rx, ry, rr = get_rim_info(first)
    except Exception:
        rx, ry = get_rim_center(first)
        rr = max(16, min(w, h)//20)

    pts: List[TrackPoint] = []
    ema_y = None
    f = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if f % TRACK_STRIDE != 0:
            f += 1
            continue
        c = _detect_ball_center(frame)
        if c is not None:
            cx, cy = c
            if ema_y is None:
                ema_y = cy
            else:
                ema_y = EMA_ALPHA * cy + (1 - EMA_ALPHA) * ema_y
            pts.append(TrackPoint(f=f, x=float(cx), y=float(ema_y)))
        f += 1

    cap.release()

    # gap fill (linear) for short gaps
    if len(pts) >= 2:
        filled: List[TrackPoint] = [pts[0]]
        for i in range(1, len(pts)):
            prev = filled[-1]
            cur = pts[i]
            gap = int((cur.f - prev.f)//TRACK_STRIDE) - 1
            if 0 < gap <= MAX_GAP_FRAMES:
                for g in range(1, gap+1):
                    t = g/(gap+1)
                    filled.append(
                        TrackPoint(
                            f=prev.f + g*TRACK_STRIDE,
                            x=prev.x*(1-t)+cur.x*t,
                            y=prev.y*(1-t)+cur.y*t
                        )
                    )
            filled.append(cur)
        pts = filled

    return pts, fps, total, (rx, ry, rr)

def _velocity_y(track: List[TrackPoint], fps: float) -> List[Tuple[int,float]]:
    vy = []
    if len(track) < 2:
        return vy
    for i in range(1, len(track)):
        dt = (track[i].f - track[i-1].f) / fps
        if dt <= 0:
            continue
        vy.append((track[i].f, (track[i].y - track[i-1].y)/dt))
    return vy

def _find_candidates(track: List[TrackPoint], vy: List[Tuple[int,float]], fps: float, rim: Tuple[int,int,int]) -> List[Dict]:
    if not track or not vy:
        return []

    rx, ry, rr = rim
    rr_x = rr * RIM_X_MARGIN

    # map from frame->y for quick lookup
    y_by_frame = {p.f: p.y for p in track}
    x_by_frame = {p.f: p.x for p in track}
    frames = [p.f for p in track]

    cands: List[Dict] = []
    i = 0
    while i < len(vy):
        f, v = vy[i]
        # release: strong upward motion
        if v <= VY_UP_THRESH:
            # find apex: sign change to positive (downward) with min separation
            j = i + 1
            seen_up = True
            apex_f = None
            while j < len(vy):
                f2, v2 = vy[j]
                if f2 - f < MIN_APEX_SEP_FRAMES * TRACK_STRIDE:
                    j += 1
                    continue
                if v2 > 0:  # downward
                    apex_f = f2
                    break
                j += 1
            if apex_f is None:
                i += 1
                continue

            # vertical excursion
            y_release = y_by_frame.get(f, None)
            y_apex = y_by_frame.get(apex_f, None)
            if y_release is None or y_apex is None:
                i += 1
                continue
            if (y_release - y_apex) < MIN_DY_PIX:
                i += 1
                continue

            # rim crossing (downward through rim y near rim x)
            cross_f = None
            k = j
            while k < len(vy):
                fk, vk = vy[k]
                yk = y_by_frame.get(fk, None)
                xk = x_by_frame.get(fk, None)
                if yk is None or xk is None:
                    k += 1
                    continue
                close_x = abs(xk - rx) <= rr_x
                if vk > 0 and close_x and yk >= ry:  # downward & at/below rim
                    cross_f = fk
                    break
                # stop searching 2 seconds after apex
                if (fk - apex_f) / fps > 2.2:
                    break
                k += 1

            # Build a window even if no crossing yet (Gemini may help)
            t_release = f / fps
            t_end = (cross_f or apex_f) / fps + 1.0
            t_start = max(0.0, t_release - CAND_PRE_SEC)

            cid = f"c{len(cands)+1}"
            cands.append({
                "id": cid,
                "start": t_start,
                "end": t_end,
                "release_f": f,
                "apex_f": apex_f,
                "cross_f": cross_f,
                "confidence": 0.92 if cross_f else 0.85,
            })

            # skip ahead ~1.5s to avoid duplicates for the same shot
            skip_frames = int(DEDUP_SEC * fps)
            while i < len(vy) and vy[i][0] < (apex_f + skip_frames):
                i += 1
            continue
        i += 1

    # dedup very-close windows by start time
    cands.sort(key=lambda c: c["start"])
    deduped = []
    last_end = -999.0
    for c in cands:
        if not deduped:
            deduped.append(c)
            last_end = c["end"]
            continue
        if c["start"] <= last_end - 0.20:  # overlapping a lot → merge
            if c["end"] > last_end:
                deduped[-1]["end"] = c["end"]
                deduped[-1]["cross_f"] = deduped[-1]["cross_f"] or c["cross_f"]
        else:
            deduped.append(c)
            last_end = c["end"]

    log.info("shot_tracker: candidates=%s", deduped)
    return deduped

def _local_outcome(video_path: str, cand: Dict, fps: float, rim: Tuple[int,int,int], track: List[TrackPoint]) -> str:
    """
    Simple local rule: if we found a rim-plane crossing, check if the ball stays below rim
    for MADE_HOLD_FRAMES; else if it pops back above quickly → missed; else unknown.
    """
    rx, ry, rr = rim
    if not cand.get("cross_f"):
        return "unknown"

    cross_f = cand["cross_f"]
    # Build quick index for the track
    y_by_f = {p.f: p.y for p in track}

    below_count = 0
    above_after = False
    last_f = cross_f + MADE_HOLD_FRAMES * TRACK_STRIDE
    for p in track:
        if cross_f <= p.f <= last_f:
            if p.y >= ry:
                below_count += 1
            else:
                above_after = True  # jumped back up
        if p.f > last_f:
            break

    if below_count >= MADE_HOLD_FRAMES:
        return "made"
    if above_after:
        return "missed"
    return "unknown"

# ----------------------------
# Public API
# ----------------------------

def detect_shot_candidates(video_path: str) -> List[Dict]:
    track, fps, total, rim = _track_ball(video_path)
    if not track:
        log.info("shot_tracker: no track")
        return []
    vy = _velocity_y(track, fps)
    return _find_candidates(track, vy, fps, rim)

def classify_outcomes(video_path: str, candidates: List[Dict]) -> Dict[str, str]:
    track, fps, total, rim = _track_ball(video_path)
    out: Dict[str,str] = {}
    for c in candidates:
        out[c["id"]] = _local_outcome(video_path, c, fps, rim, track)
        log.info("shot_tracker: candidate %s -> %s", c["id"], out[c["id"]])
    return out
