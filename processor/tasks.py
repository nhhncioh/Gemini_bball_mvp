# processor/tasks.py
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

from celery import Celery
from celery.utils.log import get_task_logger

# Import the modified processor
from .gemini_processor import process_video_simple
from .overlay import render_overlay

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


@celery_app.task(name="processor.tasks.process_video", bind=True)
def process_video(self, video_path: str) -> Dict[str, Any]:
    """
    Simple Gemini-based video processing at 1fps (full video analysis)
    
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
        "shots_found": 0,
        "errors": [],
        "approach": "gemini_full_video_1fps"
    }
    
    try:
        logger.info("Starting FULL VIDEO GEMINI processing: Task %s | video=%s", job_id, video_path)

        src = Path(video_path)
        if not src.exists():
            raise FileNotFoundError(f"Video not found: {src}")

        # Use full video Gemini processing
        result = process_video_simple(str(src))
        
        if "error" in result:
            raise RuntimeError(result["error"])
        
        events = result.get("shots", [])
        processing_stats["shots_found"] = len(events)
        processing_stats["total_shots"] = result.get("total_shots", 0)
        processing_stats["rim_center"] = result.get("rim_center", (0, 0))
        processing_stats["video_stats"] = result.get("video_stats", {})
        processing_stats["processing_successful"] = True
        
        logger.info("Full video Gemini processing found %d shots", len(events))
        
        # Log each shot for debugging
        for i, event in enumerate(events, 1):
            logger.info(
                "Shot %d: %.1fs-%.1fs, %s (confidence: %.2f)",
                i,
                event.get("start", 0),
                event.get("end", 0),
                event.get("result", "unknown"),
                event.get("confidence", 0)
            )

        # Render overlay if shots found
        output_path = None
        if events:
            try:
                logger.info("Rendering overlay video...")
                out_name = f"{src.stem}_{job_id[:8]}_gemini_full.mp4"
                dst = OUTPUTS / out_name
                
                render_overlay(
                    src_video=str(src),
                    dst_video=str(dst),
                    events=events,
                    rim=processing_stats.get("rim_center", (0, 0)),
                    fps=processing_stats.get("video_stats", {}).get("fps", 30.0),
                )
                output_path = str(dst)
                logger.info("Overlay video saved: %s", output_path)
                
            except Exception as e:
                logger.error("Overlay rendering failed: %s", e)
                processing_stats["errors"].append(f"Overlay error: {str(e)}")
        else:
            logger.warning("No shots found in video")

        logger.info("Full video Gemini processing completed successfully for task %s", job_id)

        return {
            "output": output_path,
            "events": events,
            "error": None,
            "traceback": None,
            "processing_stats": processing_stats
        }

    except Exception as e:
        logger.exception("Full video Gemini processing failed for task %s", job_id)
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