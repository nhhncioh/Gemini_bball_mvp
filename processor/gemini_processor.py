# processor/gemini_processor.py
from __future__ import annotations

import io
import json
import os
import math
from typing import Any, Dict, List, Sequence, Tuple, Optional

from PIL import Image, ImageDraw, ImageFont
import google.generativeai as genai
from celery.utils.log import get_task_logger
import cv2
import numpy as np

logger = get_task_logger(__name__)

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp")
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

# Debug logging for environment variables
logger.info(f"Gemini model configured: {GEMINI_MODEL}")
logger.info(f"API key configured: {'✅ Yes' if GOOGLE_API_KEY else '❌ Missing'}")
if GOOGLE_API_KEY:
    logger.info(f"API key prefix: {GOOGLE_API_KEY[:8]}...")


def _cfg_gemini():
    if not GOOGLE_API_KEY:
        logger.error(f"GOOGLE_API_KEY is not set. Available env vars: {list(os.environ.keys())}")
        raise RuntimeError("GOOGLE_API_KEY is not set")
    genai.configure(api_key=GOOGLE_API_KEY)
    return genai.GenerativeModel(GEMINI_MODEL)


def _jpeg_bytes(frame_bgr) -> bytes:
    """Convert OpenCV BGR frame to JPEG bytes"""
    ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        raise RuntimeError("Failed to encode JPEG")
    return bytes(buf)


def extract_frames_at_fps(frames: List[np.ndarray], original_fps: float, target_fps: float = 1.0) -> Tuple[List[np.ndarray], List[float]]:
    """Extract frames at target FPS from full frame list"""
    frame_interval = int(original_fps / target_fps)
    sampled_frames = []
    timestamps = []
    
    for i in range(0, len(frames), frame_interval):
        sampled_frames.append(frames[i])
        timestamps.append(i / original_fps)
    
    logger.info(f"Sampled {len(sampled_frames)} frames at {target_fps}fps from {len(frames)} total frames")
    return sampled_frames, timestamps


def analyze_full_video_with_gemini(
    frames: List[np.ndarray],
    rim_center: Tuple[int, int],
    fps: float = 30.0
) -> Dict[str, Any]:
    """
    Analyze entire video with Gemini in one pass at 1fps
    This mimics uploading the video directly to Gemini
    """
    
    logger.info("Starting full video Gemini analysis at 1fps")
    
    # Sample frames at 1fps
    sampled_frames, timestamps = extract_frames_at_fps(frames, fps, target_fps=1.0)
    
    # Build the prompt - simple and direct
    prompt = f"""
Analyze this complete basketball video sampled at 1fps and identify ALL basketball shots.

The rim is located at position {rim_center} (marked with red circle in frames).

For each shot you find, provide:
1. Start time (in seconds)
2. End time (in seconds)  
3. Outcome (made/missed)
4. Confidence score (0-1)
5. Key observations about the shot

Look for these indicators:
- Ball being released from hands (shot start)
- Ball trajectory arc moving upward then downward
- Ball approaching the rim
- Ball going through hoop (made) or bouncing away (missed)
- Net movement after ball interaction

A MADE shot shows:
- Ball passing through the rim opening
- Net moving downward/inward
- Ball emerging below the basket

A MISSED shot shows:
- Ball hitting rim and bouncing away
- Ball passing beside the rim
- Net moving backward/sideways or no net movement

Return results in this exact JSON format:
{{
  "shots": [
    {{
      "shot_number": 1,
      "start_time": 0.0,
      "end_time": 0.0,
      "outcome": "made",
      "confidence": 0.0,
      "observations": "description"
    }}
  ],
  "total_shots_found": 0,
  "video_duration": {len(frames)/fps:.1f}
}}

IMPORTANT: 
- Find ALL shots in the video
- The video likely contains multiple shots (3-5 expected)
- Each frame represents 1 second of video time
- Analyze the entire sequence carefully
"""

    # Build content for Gemini
    content_parts = [{"text": prompt}]
    
    # Add all sampled frames with timestamps
    for i, (frame, timestamp) in enumerate(zip(sampled_frames, timestamps)):
        # Add timestamp context
        content_parts.append({
            "text": f"\n[Frame {i+1}/{len(sampled_frames)} @ {timestamp:.1f}s]"
        })
        
        # Add visual rim marker to frame
        marked_frame = frame.copy()
        cv2.circle(marked_frame, rim_center, 30, (0, 0, 255), 2)  # Red circle for rim
        
        # Convert to JPEG and add
        jpeg_data = _jpeg_bytes(marked_frame)
        content_parts.append({
            "inline_data": {
                "mime_type": "image/jpeg",
                "data": jpeg_data
            }
        })
    
    # Add final instruction
    content_parts.append({
        "text": "\n\nAnalyze all frames above sequentially. Identify every basketball shot from start to finish. Return the complete JSON response with all shots found."
    })
    
    try:
        # Configure and call Gemini
        model = _cfg_gemini()
        
        # Generate response with low temperature for consistency
        response = model.generate_content(
            content_parts,
            generation_config={
                "temperature": 0.1,
                "top_p": 0.1,
                "response_mime_type": "application/json"
            }
        )
        
        # Parse response
        result_text = response.text
        result = json.loads(result_text)
        
        logger.info(f"Gemini found {result.get('total_shots_found', 0)} shots in full video analysis")
        
        # Log each shot found
        for shot in result.get("shots", []):
            logger.info(f"Shot {shot.get('shot_number')}: {shot.get('start_time')}s-{shot.get('end_time')}s, "
                       f"{shot.get('outcome')} (conf: {shot.get('confidence')})")
        
        return result
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse Gemini JSON response: {e}")
        return {
            "shots": [],
            "total_shots_found": 0,
            "error": f"JSON parse error: {str(e)}"
        }
    except Exception as e:
        logger.error(f"Gemini analysis failed: {e}")
        return {
            "shots": [],
            "total_shots_found": 0,
            "error": str(e)
        }


def process_video_simple(video_path: str) -> Dict[str, Any]:
    """
    Main processing function using full video analysis
    """
    logger.info(f"Processing video with full Gemini analysis: {video_path}")
    
    try:
        # Load video frames
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Resize if too large
            height, width = frame.shape[:2]
            if width > 1280:
                scale = 1280 / width
                new_width = int(width * scale)
                new_height = int(height * scale)
                frame = cv2.resize(frame, (new_width, new_height))
            
            frames.append(frame)
        
        cap.release()
        
        logger.info(f"Loaded {len(frames)} frames at {fps:.1f} FPS")
        
        # Simple rim detection using first frame
        from .rim_detector import get_rim_center
        rim_center = get_rim_center(frames[len(frames)//2])  # Use middle frame
        logger.info(f"Rim detected at: {rim_center}")
        
        # Analyze full video with Gemini
        gemini_result = analyze_full_video_with_gemini(frames, rim_center, fps)
        
        # Convert to events format
        events = []
        for shot in gemini_result.get("shots", []):
            event = {
                "type": "shot",
                "start": shot.get("start_time", 0),
                "end": shot.get("end_time", 0),
                "result": shot.get("outcome", "unknown"),
                "player": "unknown",
                "confidence": shot.get("confidence", 0.8),
                "review": shot.get("observations", ""),
                "shot_number": shot.get("shot_number", 0),
                "_source": "gemini_full_video_1fps"
            }
            events.append(event)
        
        # Sort by start time
        events.sort(key=lambda x: x["start"])
        
        logger.info(f"Processing complete: {len(events)} shots detected")
        
        return {
            "shots": events,
            "total_shots": len(events),
            "rim_center": rim_center,
            "processing_method": "gemini_full_video_1fps",
            "gemini_raw": gemini_result,
            "video_stats": {
                "frames": len(frames),
                "fps": fps,
                "duration": len(frames) / fps
            }
        }
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        return {
            "shots": [],
            "total_shots": 0,
            "error": str(e)
        }


# Keep backward compatibility functions
def classify_with_gemini(
    rim: Tuple[int, int],
    frames: Sequence[Any],
    window: Tuple[int, int],
    **kwargs
) -> Dict[str, Any]:
    """Backward compatibility wrapper"""
    # For compatibility, just analyze the window as if it's a mini-video
    start_f, end_f = window
    window_frames = frames[start_f:end_f+1]
    
    # Simple classification based on window
    return {
        "outcome": "unknown",
        "confidence": 0.5,
        "reason": "Using full video analysis instead of windows",
        "key_evidence": []
    }


def analyze_window_with_gemini(*args, **kwargs) -> Dict[str, Any]:
    """Backward compatibility wrapper"""
    return {
        "shots_found": [],
        "total_shots": 0,
        "window_summary": "Using full video analysis instead of windows",
        "analysis_confidence": 0.0
    }


def convert_window_results_to_events(*args, **kwargs) -> List[Dict[str, Any]]:
    """Backward compatibility wrapper"""
    return []


# Backward compatibility
def classify_shot_with_gemini(*args, **kwargs) -> Dict[str, Any]:
    return classify_with_gemini(*args, **kwargs)