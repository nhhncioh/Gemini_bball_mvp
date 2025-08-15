"""
trigger.py
----------
Detect candidate basketball-shot timestamps by tracking the ball’s Y-coordinate
and finding the apex of each flight path.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from scipy.signal import find_peaks
from ultralytics import YOLO

from .sort_tracker import Sort   # local lightweight SORT

# ─── YOLO CONFIG ───────────────────────────────────────────────────────────
BALL_MODEL = YOLO("yolov8n.pt")
BALL_ID    = next(i for i, n in BALL_MODEL.names.items() if "sports" in n)
YOLO_CONF  = 0.10
AREA_MIN   = 10

# ─── SORT CONFIG ───────────────────────────────────────────────────────────
tracker = Sort(max_age=40, min_hits=1, iou_threshold=0.10)

# ─── MAIN ──────────────────────────────────────────────────────────────────
def detect_shot_candidates(video: Path) -> Tuple[List[int], float]:
    cap   = cv2.VideoCapture(str(video))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_detections: Dict[int, List[List[float]]] = {}
    track_y: Dict[int, List[Tuple[int, int]]] = {}

    sec, det_cnt, t0 = 0, 0, time.time()

    for f_idx in range(total):
        ok, frame = cap.read()
        if not ok:
            break

        # ── YOLO detect ───────────────────────────────────────────────
        pred = BALL_MODEL.predict(frame, imgsz=640, conf=YOLO_CONF,
                                  verbose=False)[0]

        dets = []
        for box, cls, conf in zip(pred.boxes.xyxy,
                                  pred.boxes.cls,
                                  pred.boxes.conf):
            if int(cls) != BALL_ID:
                continue
            x1, y1, x2, y2 = map(int, box)
            area = (x2 - x1) * (y2 - y1)
            if area < AREA_MIN:
                continue
            dets.append([x1, y1, x2, y2, float(conf)])

        # per-second raw-det logging
        det_cnt += len(dets)
        if time.time() - t0 >= 1.0:
            print(f"⏱️ t={sec:3.1f}s – raw dets this sec: {det_cnt}")
            sec, det_cnt, t0 = sec + 1, 0, time.time()

        if not dets:
            # advance tracker age even when no detections this frame
            tracker.update(np.empty((0, 5)))
            continue

        frame_detections[f_idx] = dets
        tracks = tracker.update(np.asarray(dets))

        for x1, y1, x2, y2, tid in tracks:
            cy = int((y1 + y2) / 2)
            track_y.setdefault(int(tid), []).append((f_idx, cy))

    cap.release()

    # ── peak-finding per track ─────────────────────────────────────────────
    peak_frames: List[int] = []
    for seq in track_y.values():
        if len(seq) < 3:   # shorter sequences are ignored
            continue
        seq.sort()
        frames = np.array([f for f, _ in seq])
        ys     = np.array([cy for _, cy in seq])
        peaks, _ = find_peaks(-ys, prominence=0.04,
                              distance=int(fps * 1.5))
        peak_frames.extend(frames[peaks].tolist())

    # ── stats + fallback ──────────────────────────────────────────────────
    lens = [len(s) for s in track_y.values()]
    if lens:
        print(f"📈 tracks: n={len(lens)}  min={min(lens)}  "
              f"median={int(np.median(lens))}  max={max(lens)}")

    if not peak_frames and frame_detections:
        timeline = {f: int((max(d, key=lambda x: x[-1])[1] +
                            max(d, key=lambda x: x[-1])[3]) / 2)
                    for f, d in frame_detections.items()}
        frames = np.array(sorted(timeline))
        ys     = np.array([timeline[f] for f in frames])
        peaks, _ = find_peaks(-ys, prominence=0.04,
                              distance=int(fps * 0.5))
        peak_frames = frames[peaks].tolist()

    peak_frames.sort()
    print(f"🎯 {len(peak_frames)} shot candidates ({fps:.1f} fps)")
    if peak_frames:
        print("   peak times (s):",
              [round(f / fps, 2) for f in peak_frames])

    return peak_frames, fps
