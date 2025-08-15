from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

from celery import Celery
from celery.utils.log import get_task_logger

# Updated imports - use existing modules with new approach
from .shot_detector import load_video_frames  # Keep existing frame loading
from .rim_detector import get_rim_center
from .overlay import render_overlay
from .gemini_processor import analyze_window_with_gemini, convert_window_results_to_events

logger = get_task_logger(__name__)

# Celery
celery_app = Celery(
    "processor",
    broker=os.getenv("CELERY_BROKER_URL", "redis://redis:6379/0"),
    backend=os.getenv("CELERY_RESULT_BACKEND", "redis://redis:6379/0"),
)

DATA_DIR = Path(os.getenv("DATA_DIR", "/data"))
UPLOADS = DATA_DIR / "uploads"
OUTPUTS = DATA_DIR / "outputs"
OUTPUTS.mkdir(parents=True, exist_ok=True)


def _sec(f: int, fps: float) -> float:
    return round(f / max(fps, 1e-6), 3)


def detect_motion_periods(frames: List, fps: float) -> List[Tuple[float, float]]:
    """Detect periods of significant motion in the video"""
    if len(frames) < 10:
        return []
    
    motion_scores = []
    prev_gray = None
    
    # Calculate motion scores
    for i, frame in enumerate(frames):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        
        if prev_gray is not None:
            # Calculate motion as mean absolute difference
            motion = np.mean(cv2.absdiff(gray, prev_gray))
            motion_scores.append(motion)
        else:
            motion_scores.append(0.0)
        
        prev_gray = gray
    
    if not motion_scores:
        return []
    
    # Find motion peaks
    import numpy as np
    motion_array = np.array(motion_scores)
    threshold = np.mean(motion_array) + 1.5 * np.std(motion_array)
    
    # Find periods above threshold
    motion_periods = []
    in_motion = False
    start_time = 0.0
    
    for i, score in enumerate(motion_scores):
        time = i / fps
        
        if score > threshold and not in_motion:
            # Start of motion period
            start_time = max(0, time - 1.0)  # Start 1 second before
            in_motion = True
        elif score <= threshold and in_motion:
            # End of motion period
            end_time = min(len(frames) / fps, time + 1.0)  # End 1 second after
            
            # Only include periods longer than 2 seconds
            if end_time - start_time >= 2.0:
                motion_periods.append((start_time, end_time))
            
            in_motion = False
    
    # Handle case where motion continues to end
    if in_motion:
        end_time = len(frames) / fps
        if end_time - start_time >= 2.0:
            motion_periods.append((start_time, end_time))
    
    logger.info(f"Found {len(motion_periods)} motion periods: {motion_periods}")
    return motion_periods


def create_analysis_windows(video_duration: float, motion_periods: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Create overlapping time windows for Gemini analysis"""
    
    if motion_periods:
        # Use motion periods as base, but ensure good coverage
        windows = []
        
        for start, end in motion_periods:
            # Expand each motion period slightly
            expanded_start = max(0, start - 0.5)
            expanded_end = min(video_duration, end + 0.5)
            
            # Create overlapping windows if period is long
            if expanded_end - expanded_start > 8.0:
                # Split long periods into overlapping 6-second windows
                current = expanded_start
                while current < expanded_end:
                    window_end = min(current + 6.0, expanded_end)
                    windows.append((current, window_end))
                    current += 4.0  # 2-second overlap
            else:
                windows.append((expanded_start, expanded_end))
        
        # Merge overlapping windows
        windows.sort()
        merged = []
        for start, end in windows:
            if merged and start <= merged[-1][1] + 1.0:
                # Merge with previous window
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        
        windows = merged
    else:
        # Fallback: divide video into overlapping windows
        window_size = 8.0  # 8-second windows
        overlap = 4.0      # 4-second overlap
        
        windows = []
        current = 0.0
        while current < video_duration:
            end = min(current + window_size, video_duration)
            if end - current >= 3.0:  # Minimum window size
                windows.append((current, end))
            current += overlap
    
    # Limit to reasonable number of windows
    if len(windows) > 8:
        # Keep the most promising windows (those with motion periods)
        if motion_periods:
            # Prioritize windows that overlap with motion periods
            scored_windows = []
            for window in windows:
                score = 0
                for motion_start, motion_end in motion_periods:
                    overlap = max(0, min(window[1], motion_end) - max(window[0], motion_start))
                    score += overlap
                scored_windows.append((score, window))
            
            # Sort by score and take top 8
            scored_windows.sort(reverse=True)
            windows = [window for score, window in scored_windows[:8]]
        else:
            # Just take first 8 windows
            windows = windows[:8]
    
    logger.info(f"Created {len(windows)} analysis windows: {windows}")
    return windows


def create_window_candidates(frames: List, fps: float) -> List[Dict[str, Any]]:
    """Create analysis windows for Gemini (simplified approach)"""
    
    video_duration = len(frames) / fps
    
    # 1. Detect motion periods
    motion_periods = detect_motion_periods(frames, fps)
    
    # 2. Create analysis windows
    analysis_windows = create_analysis_windows(video_duration, motion_periods)
    
    # 3. Create candidates for each window
    candidates = []
    
    for i, (start_time, end_time) in enumerate(analysis_windows):
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        
        # Ensure valid frame range
        start_frame = max(0, start_frame)
        end_frame = min(len(frames) - 1, end_frame)
        
        if end_frame <= start_frame:
            continue
        
        # Simple ball tracking (just mark that we have frames)
        window_ball_data = [
            {
                'frame': start_frame + j,
                'x': 0,  # Placeholder
                'y': 0,  # Placeholder  
                'confidence': 0.5,
                'time': (start_frame + j) / fps
            }
            for j in range(0, end_frame - start_frame, max(1, (end_frame - start_frame) // 10))
        ]
        
        # Create candidate for this window
        candidate = {
            'id': f"window_{i+1}",
            'start': start_time,
            'end': end_time,
            'start_f': start_frame,
            'end_f': end_frame,
            'confidence': 0.8,  # High confidence - let Gemini decide
            'ball_track': window_ball_data,
            'window_type': 'motion' if any(
                start_time <= mp[0] <= end_time or start_time <= mp[1] <= end_time 
                for mp in motion_periods
            ) else 'coverage',
            'shooting_metrics': {
                'duration_seconds': round(end_time - start_time, 2),
                'ball_tracking_points': len(window_ball_data),
                'analysis_type': 'window_based'
            }
        }
        
        candidates.append(candidate)
        
        logger.info(f"GEMINI WINDOW {candidate['id']}: {start_time:.1f}s-{end_time:.1f}s, "
                   f"type: {candidate['window_type']}")
    
    logger.info(f"Final result: {len(candidates)} analysis windows for Gemini")
    return candidates


@celery_app.task(name="processor.tasks.process_video", bind=True)
def process_video(self, video_path: str) -> Dict[str, Any]:
    """
    Gemini-first video processing approach
    
    Returns:
      {
        "output": str | None,
        "events": List[Dict[str, Any]],
        "error": str | None,
        "traceback": str | None,
        "processing_stats": Dict[str, Any]
      }
    """
    job_id = getattr(self.request, "id", "nojob")
    processing_stats = {
        "job_id": job_id,
        "windows_analyzed": 0,
        "shots_found": 0,
        "gemini_calls": 0,
        "errors": [],
        "approach": "gemini_first"
    }
    
    try:
        logger.info("GEMINI-FIRST processing started: Task %s | video=%s", job_id, video_path)

        src = Path(video_path)
        if not src.exists():
            raise FileNotFoundError(f"Video not found: {src}")

        # --- 1) Load frames ---
        logger.info("Loading video frames...")
        frames, fps = load_video_frames(str(src), max_size=720)
        if not frames:
            raise RuntimeError("No frames decoded from video")
        
        logger.info("Loaded %d frames at %.2f FPS", len(frames), fps)

        # --- 2) Create analysis windows (not individual shots) ---
        logger.info("Creating analysis windows for Gemini...")
        window_candidates = find_shot_candidates(frames, fps)
        processing_stats["windows_analyzed"] = len(window_candidates)
        
        if not window_candidates:
            logger.warning("No analysis windows created")
            return {
                "output": None,
                "events": [],
                "error": None,
                "traceback": None,
                "processing_stats": processing_stats
            }
        
        logger.info("Created %d analysis windows", len(window_candidates))

        # --- 3) Rim detection ---
        logger.info("Detecting rim center...")
        rim_xy = get_rim_center(frames[len(frames)//2])
        logger.info("Rim detected at: %s", rim_xy)

        # --- 4) Gemini window analysis ---
        window_results = []
        total_shots_found = 0
        
        for i, window_candidate in enumerate(window_candidates):
            processing_stats["gemini_calls"] += 1
            
            logger.info("Analyzing window %s: %s", 
                       window_candidate["id"], 
                       f"{window_candidate['start']:.1f}s-{window_candidate['end']:.1f}s")
            
            start_f = int(window_candidate["start_f"])
            end_f = int(window_candidate["end_f"])
            window = (start_f, end_f)
            
            # Get ball tracking data for this window
            ball_track = window_candidate.get("ball_track", [])
            window_info = window_candidate.get("shooting_metrics", {})
            
            try:
                # Analyze window with Gemini
                window_result = analyze_window_with_gemini(
                    rim=rim_xy,
                    frames=frames,
                    window=window,
                    fps=fps,
                    ball_track=ball_track,
                    window_info=window_info,
                    max_images=12,  # More frames for window analysis
                    temperature=0.1,  # Low temperature for consistency
                    top_p=0.1,
                )
                
                window_results.append(window_result)
                shots_in_window = window_result.get('total_shots', 0)
                total_shots_found += shots_in_window
                
                logger.info("Window %s analyzed: %d shots found (confidence: %.2f)", 
                           window_candidate["id"], 
                           shots_in_window,
                           window_result.get('analysis_confidence', 0.0))
                
                # Log individual shots found
                for shot in window_result.get('shots_found', []):
                    logger.info("  - Shot %d: %.1fs-%.1fs, %s (%.2f confidence)",
                               shot.get('shot_number', 0),
                               shot.get('approximate_start_time', 0),
                               shot.get('approximate_end_time', 0),
                               shot.get('outcome', 'unknown'),
                               shot.get('confidence', 0))
                
            except Exception as e:
                logger.error("Gemini window analysis failed for %s: %s", 
                           window_candidate["id"], str(e))
                processing_stats["errors"].append(f"Gemini error for {window_candidate['id']}: {str(e)}")
                
                # Fallback result
                window_result = {
                    "shots_found": [],
                    "total_shots": 0,
                    "window_summary": f"Analysis failed: {str(e)}",
                    "analysis_confidence": 0.0
                }
                window_results.append(window_result)

        # --- 5) Convert window results to standard event format ---
        logger.info("Converting window results to events...")
        events = convert_window_results_to_events(window_results, window_candidates)
        processing_stats["shots_found"] = len(events)
        
        logger.info("Final results: %d shots found across %d windows", 
                   len(events), len(window_candidates))
        
        # Log final shot summary
        for i, event in enumerate(events):
            logger.info("FINAL SHOT %d: %.1fs-%.1fs, %s (confidence: %.2f)",
                       i + 1,
                       event.get('start', 0),
                       event.get('end', 0),
                       event.get('result', 'unknown'),
                       event.get('confidence', 0))

        # --- 6) Enhanced overlay rendering ---
        logger.info("Rendering overlay video...")
        out_name = f"{src.stem}_{job_id[:8]}_gemini_first_overlay.mp4"
        dst = OUTPUTS / out_name
        
        try:
            render_overlay(
                src_video=str(src),
                dst_video=str(dst),
                events=events,
                rim=rim_xy,
                fps=fps,
            )
            output_path = str(dst)
            logger.info("Overlay video saved: %s", output_path)
            
        except Exception as e:
            logger.exception("Overlay rendering failed: %s", e)
            processing_stats["errors"].append(f"Overlay error: {str(e)}")
            output_path = None

        # Final processing stats
        processing_stats.update({
            "total_frames": len(frames),
            "fps": fps,
            "rim_center": rim_xy,
            "processing_successful": True,
            "total_windows": len(window_candidates),
            "total_shots": len(events)
        })

        logger.info("GEMINI-FIRST processing completed successfully for task %s", job_id)

        return {
            "output": output_path,
            "events": events,
            "error": None,
            "traceback": None,
            "processing_stats": processing_stats
        }

    except Exception as e:
        logger.exception("GEMINI-FIRST processing failed for task %s", job_id)
        processing_stats.update({
            "processing_successful": False,
            "final_error": f"{type(e).__name__}: {e}"
        })
        
        return {
            "output": None, 
            "events": [], 
            "error": f"{type(e).__name__}: {e}", 
            "traceback": None,
            "processing_stats": processing_stats
        }


# Additional utility task for debugging
@celery_app.task(name="processor.tasks.analyze_video_windows")
def analyze_video_windows(video_path: str, max_windows: int = 5) -> Dict[str, Any]:
    """
    Debug task to analyze video windows without full processing
    """
    try:
        src = Path(video_path)
        if not src.exists():
            return {"error": f"Video not found: {src}"}
        
        frames, fps = load_video_frames(str(src), max_size=720)
        if not frames:
            return {"error": "No frames loaded"}
        
        rim_center = get_rim_center(frames[len(frames)//2])
        window_candidates = find_shot_candidates(frames, fps)
        
        # Limit windows for debugging
        window_candidates = window_candidates[:max_windows]
        
        analysis = {
            "total_frames": len(frames),
            "fps": fps,
            "rim_center": rim_center,
            "windows_created": len(window_candidates),
            "windows": []
        }
        
        for candidate in window_candidates:
            window_info = {
                "id": candidate["id"],
                "duration": candidate["end"] - candidate["start"],
                "ball_tracking_points": len(candidate.get("ball_track", [])),
                "window_type": candidate.get("window_type", "unknown")
            }
            analysis["windows"].append(window_info)
        
        return analysis
        
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}