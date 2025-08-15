from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Optional, Union

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

def _center(box: np.ndarray) -> Tuple[float, float]:
    x1, y1, x2, y2 = box
    return float((x1 + x2) / 2.0), float((y1 + y2) / 2.0)

def detect_shot_times_yolo(
    video: Union[str, Path],
    model_path: Union[str, Path],
    conf: float = 0.25,
    iou: float = 0.45,
    min_gap_sec: float = 1.5,
    max_sample_fps: float = 10.0,
) -> List[float]:
    if YOLO is None:
        return []
    model = YOLO(str(model_path))

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    stride = max(int(round(fps / max_sample_fps)), 1)

    times: List[float] = []
    last_shot_t: float = -1e9
    last_ball_cy: Optional[float] = None
    last_vy_sign: Optional[int] = None

    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % stride != 0:
                frame_idx += 1
                continue

            t = frame_idx / fps
            res = model.predict(frame, conf=conf, iou=iou, verbose=False)[0]

            hoop_box = None
            ball_box = None
            for c in res.boxes:
                cls_name = None
                try:
                    cls_name = res.names.get(int(c.cls))
                except Exception:
                    pass
                if cls_name is None:
                    continue
                box = c.xyxy[0].cpu().numpy()
                name = cls_name.lower()
                if name in {"hoop", "rim", "basket"}:
                    hoop_box = box
                elif name in {"ball", "basketball"}:
                    ball_box = box

            if ball_box is not None:
                _, cy = _center(ball_box)
                if last_ball_cy is not None:
                    vy = cy - last_ball_cy
                    vy_sign = 1 if vy > 0 else (-1 if vy < 0 else 0)
                    apex = last_vy_sign is not None and vy_sign > 0 and last_vy_sign < 0

                    made_zone = False
                    if hoop_box is not None:
                        hx1, hy1, hx2, hy2 = hoop_box
                        bx, by = _center(ball_box)
                        tol = 8.0
                        made_zone = (hx1 - tol) <= bx <= (hx2 + tol) and (hy1 - tol) <= by <= (hy2 + tol)

                    recently = (t - last_shot_t) >= min_gap_sec
                    if apex and recently and (made_zone or hoop_box is not None):
                        times.append(t)
                        last_shot_t = t

                    last_vy_sign = vy_sign
                last_ball_cy = cy

            frame_idx += 1
    finally:
        cap.release()

    return times
