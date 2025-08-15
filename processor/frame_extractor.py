import cv2
import numpy as np
from pathlib import Path

def extract_frames_at_fps(video_path: Path, fps: int = 3):
    cap = cv2.VideoCapture(str(video_path))
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / original_fps

    sampled_indices = [int(t * original_fps) for t in np.arange(0, duration, 1 / fps)]
    frames, times = [], []

    for idx in sampled_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        success, frame = cap.read()
        if success:
            frames.append(frame)
            times.append(round(idx / original_fps, 2))

    cap.release()
    return frames, times
