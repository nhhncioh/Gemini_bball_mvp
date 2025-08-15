"""
processor.rim_detector
----------------------
Loads YOLOv8 rim model and returns (rim_y, (x1,x2)).
"""

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

# Optional Ultralytics
try:
    from ultralytics import YOLO
except ModuleNotFoundError:
    YOLO = None  # fallback

RIM_MODEL_PATH = Path(__file__).parent.parent / "models" / "rim_best.pt"

_rim_yolo = None

def _load_rim_yolo():
    global _rim_yolo
    if _rim_yolo is not None:
        return _rim_yolo
    if YOLO is None or not RIM_MODEL_PATH.exists():
        print("⚠️  Ultralytics not available or rim model missing – using default ROI")
        return None
    _rim_yolo = YOLO(str(RIM_MODEL_PATH))
    return _rim_yolo

def detect_rim(frame_bgr: np.ndarray) -> Tuple[int, Tuple[int,int]]:
    """
    Returns
    -------
    rim_y  : vertical centre of rim
    hoop_x : (x1,x2) horizontal span
    """
    model = _load_rim_yolo()
    if model is None:
        return 90, (250, 390)           # fallback ROI

    pred = model.predict(frame_bgr, imgsz=640, verbose=False, conf=0.25)[0]
    if not len(pred):
        return 90, (250, 390)

    box = max(pred.boxes.xyxy, key=lambda b: (b[2]-b[0])*(b[3]-b[1]))
    x1,y1,x2,y2 = map(int, box)
    rim_y = (y1 + y2) // 2
    return rim_y, (x1, x2)
