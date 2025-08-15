from __future__ import annotations
import os
import cv2
import numpy as np
from typing import Tuple, Optional, List

try:
    cv2.setNumThreads(0)
except Exception:
    pass
try:
    cv2.ocl.setUseOpenCL(False)
except Exception:
    pass

# ---- Tunables (env) ---------------------------------------------------------
# If your hoop is on the right half of the frame, set RIM_ROI=right in docker-compose (worker env).
RIM_ROI = os.getenv("RIM_ROI", "auto")       # left | right | auto
RIM_MIN_R = int(os.getenv("RIM_MIN_R", "18"))
RIM_MAX_R = int(os.getenv("RIM_MAX_R", "60"))
RIM_TOP_FRAC = float(os.getenv("RIM_TOP_FRAC", "0.65"))  # only look in top X of frame
RIM_FRAMES = int(os.getenv("RIM_FRAMES", "60"))          # analyze first N frames
RIM_STRIDE = int(os.getenv("RIM_STRIDE", "2"))           # sample every Nth frame

# ---- Internal helpers -------------------------------------------------------
def _roi_slice(shape, side: str):
    h, w = shape[:2]
    top = 0
    bottom = int(h * RIM_TOP_FRAC)
    if side == "left":
        return slice(top, bottom), slice(0, w // 2)
    elif side == "right":
        return slice(top, bottom), slice(w // 2, w)
    else:  # auto
        return slice(top, bottom), slice(0, w)

def _score_circle(frame, x, y, r) -> float:
    """Edge + orange hue score for a proposed rim ring."""
    h, w = frame.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    cv2.circle(mask, (int(x), int(y)), int(r + 4), 255, 3)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # orange-ish (tweak if rim color differs)
    low = np.array([5, 80, 60], dtype=np.uint8)
    high = np.array([25, 255, 255], dtype=np.uint8)
    m_orange = cv2.inRange(hsv, low, high)

    edges = cv2.Canny(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 80, 160)

    ring_orange = cv2.bitwise_and(m_orange, m_orange, mask=mask)
    ring_edges  = cv2.bitwise_and(edges, edges, mask=mask)

    # Weighted sum; edges capture metal ring, hue helps when color is visible
    return float(ring_orange.sum() * 0.6 + ring_edges.sum() * 0.4)

def _detect_rim_frame(frame, side='auto') -> Optional[Tuple[int, int, int, float]]:
    """Find best circle on one frame; returns (x, y, r, score) or None."""
    h, w = frame.shape[:2]
    rs, cs = _roi_slice(frame.shape, side)
    roi = frame[rs, cs]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (9, 9), 2)

    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=40,
        param1=120, param2=18, minRadius=RIM_MIN_R, maxRadius=RIM_MAX_R
    )
    if circles is None:
        return None

    circles = np.uint16(np.around(circles[0]))
    best = None
    best_score = -1.0

    for (cx, cy, r) in circles:
        # shift x back to full-frame coords if we searched the right half
        x = int(cx + (0 if side != 'right' else (w // 2)))
        y = int(cy)
        score = _score_circle(frame, x, y, r)
        if score > best_score:
            best = (x, y, int(r), score)
            best_score = score

    return best

def _median(values: List[int]) -> int:
    if not values:
        return 0
    arr = sorted(values)
    return int(arr[len(arr) // 2])

# ---- Public API (compat) ----------------------------------------------------
def get_rim_center(frame) -> Tuple[int, int]:
    x, y, r = get_rim_info_from_frame(frame)
    return x, y

def get_rim_info_from_frame(frame) -> Tuple[int, int, int]:
    """Find rim on a single frame; prefers the better of left/right half if both found."""
    res_r = _detect_rim_frame(frame, 'right')
    res_l = _detect_rim_frame(frame, 'left')

    cand = res_r or res_l
    if res_r and res_l:
        cand = res_r if res_r[3] >= res_l[3] else res_l

    if cand:
        return cand[0], cand[1], cand[2]

    # Fallback guess: near the upper middle area with a sane radius
    h, w = frame.shape[:2]
    return w // 2, int(h * 0.35), max(24, min(h, w) // 20)

def get_rim_info(video_or_frame) -> Tuple[int, int, int]:
    """
    Robustly estimate rim (rx, ry, rr).
    - If given a numpy frame, analyze that frame.
    - If given a video path, aggregate over the first RIM_FRAMES frames and return medians.
    """
    if isinstance(video_or_frame, str):
        cap = cv2.VideoCapture(video_or_frame)
        if not cap.isOpened():
            raise RuntimeError("Could not open video for rim detection")

        xs: List[int] = []
        ys: List[int] = []
        rs: List[int] = []

        f = 0
        ok, frame = cap.read()
        while ok and f < RIM_FRAMES:
            if f % RIM_STRIDE == 0:
                x, y, r = get_rim_info_from_frame(frame)
                xs.append(x); ys.append(y); rs.append(r)
            ok, frame = cap.read()
            f += 1

        cap.release()

        if not xs:
            raise RuntimeError("Rim not found in early frames")

        return _median(xs), _median(ys), _median(rs)

    # assume it's a frame
    return get_rim_info_from_frame(video_or_frame)
