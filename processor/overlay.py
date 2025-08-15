# processor/overlay.py
from __future__ import annotations
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging

log = logging.getLogger(__name__)

def _put_text_with_background(img: np.ndarray, text: str, pos: Tuple[int, int], 
                             font_scale: float = 0.7, color: Tuple[int, int, int] = (255, 255, 255),
                             bg_color: Tuple[int, int, int] = (0, 0, 0), thickness: int = 2,
                             padding: int = 5) -> int:
    """Enhanced text rendering with better background and padding"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Get text size
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    x, y = pos
    # Draw background rectangle with padding
    cv2.rectangle(img, (x - padding, y - text_height - padding), 
                  (x + text_width + padding, y + baseline + padding), bg_color, -1)
    
    # Draw border for better visibility
    cv2.rectangle(img, (x - padding, y - text_height - padding), 
                  (x + text_width + padding, y + baseline + padding), 
                  (min(255, bg_color[0] + 50), min(255, bg_color[1] + 50), min(255, bg_color[2] + 50)), 1)
    
    # Draw text
    cv2.putText(img, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)
    
    return text_height + padding * 2 + 5  # Return height for next line


def _get_result_color(result: str) -> Tuple[int, int, int]:
    """Enhanced color coding for shot results"""
    color_map = {
        "made": (0, 255, 0),      # Bright green
        "missed": (0, 0, 255),    # Bright red
        "unknown": (255, 255, 0), # Yellow
        "uncertain": (255, 165, 0), # Orange
    }
    return color_map.get(result.lower(), (255, 255, 255))  # White default


def _get_quality_color(quality: str) -> Tuple[int, int, int]:
    """Color coding for shot quality"""
    quality_colors = {
        "excellent": (0, 255, 0),     # Green
        "good": (0, 255, 255),        # Cyan
        "poor": (0, 100, 255),        # Orange-red
        "unclear": (128, 128, 128),   # Gray
        "high": (0, 255, 0),          # Green
        "medium": (0, 255, 255),      # Cyan
        "low": (0, 100, 255),         # Orange-red
    }
    return quality_colors.get(quality.lower(), (255, 255, 255))


def _calculate_enhanced_metrics(event: Dict, current_time: float) -> Dict[str, str]:
    """Calculate enhanced shot metrics for display"""
    metrics = {}
    
    # Shot progress with enhanced calculation
    start_time = event.get('start', 0)
    end_time = event.get('end', 0)
    duration = end_time - start_time
    
    if duration > 0:
        progress = min(1.0, max(0.0, (current_time - start_time) / duration))
        metrics['Progress'] = f"{progress*100:.1f}%"
        
        # Phase detection
        if progress < 0.3:
            metrics['Phase'] = "Release"
        elif progress < 0.7:
            metrics['Phase'] = "Flight"
        else:
            metrics['Phase'] = "Rim"
    
    # Enhanced confidence display
    confidence = event.get('confidence', 0)
    metrics['AI Confidence'] = f"{confidence*100:.1f}%"
    
    # Basketball-specific metrics
    basketball_metrics = event.get('basketball_metrics', {})
    if basketball_metrics:
        trajectory = basketball_metrics.get('trajectory_quality', 'unclear')
        metrics['Trajectory'] = trajectory.replace('_', ' ').title()
        
        rim_interaction = basketball_metrics.get('rim_interaction_type', 'unknown')
        if rim_interaction != 'unknown':
            metrics['Rim Contact'] = rim_interaction.replace('_', ' ').title()
    
    # Enhanced detection stats
    analysis_method = event.get('_analysis_method', '')
    if 'enhanced' in analysis_method:
        metrics['Analysis'] = "Enhanced"
    
    shot_quality = event.get('_shot_quality_rating', '')
    if shot_quality:
        metrics['Quality'] = shot_quality.title()
    
    # Motion and tracking info
    motion_intensity = event.get('_motion_intensity', 0)
    if motion_intensity > 0:
        metrics['Motion'] = f"{motion_intensity:.1f}"
    
    tracking_points = event.get('_ball_tracking_points', 0)
    if tracking_points > 0:
        metrics['Tracking'] = f"{tracking_points} pts"
    
    return metrics


def _draw_enhanced_rim_visualization(frame: np.ndarray, rim: Tuple[int, int], 
                                   shot_active: bool = False, result: str = "unknown") -> None:
    """Enhanced rim visualization with dynamic elements"""
    rim_x, rim_y = rim
    
    if shot_active:
        # Active shot - enhanced rim highlighting
        result_color = _get_result_color(result)
        
        # Animated rim circles
        base_radius = 35
        for i in range(3):
            radius = base_radius + i * 8
            thickness = 3 - i
            alpha = 0.8 - i * 0.2
            
            # Create semi-transparent effect
            overlay = frame.copy()
            cv2.circle(overlay, rim, radius, result_color, thickness)
            cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0, frame)
        
        # Crosshair
        cross_size = 20
        cv2.line(frame, (rim_x - cross_size, rim_y), (rim_x + cross_size, rim_y), 
                result_color, 3)
        cv2.line(frame, (rim_x, rim_y - cross_size), (rim_x, rim_y + cross_size), 
                result_color, 3)
        
        # Target zone
        zone_points = np.array([
            [rim_x - 60, rim_y + 25],
            [rim_x + 60, rim_y + 25],
            [rim_x + 50, rim_y - 15],
            [rim_x - 50, rim_y - 15]
        ], np.int32)
        
        # Semi-transparent target zone
        overlay = frame.copy()
        cv2.fillPoly(overlay, [zone_points], (*result_color, 100))
        cv2.addWeighted(frame, 0.7, overlay, 0.3, 0, frame)
        cv2.polylines(frame, [zone_points], True, result_color, 2)
        
    else:
        # Passive rim - subtle indication
        cv2.circle(frame, rim, 30, (128, 128, 128), 2)
        cv2.circle(frame, rim, 5, (128, 128, 128), -1)


def _draw_trajectory_visualization(frame: np.ndarray, event: Dict, current_time: float, 
                                 rim: Tuple[int, int]) -> None:
    """Draw trajectory visualization if ball tracking data is available"""
    
    # Get ball tracking data if available
    start_time = event.get('start', 0)
    end_time = event.get('end', 0)
    
    # Simulate trajectory path based on shot progress
    if start_time <= current_time <= end_time:
        progress = (current_time - start_time) / max(end_time - start_time, 0.1)
        
        # Simple trajectory simulation
        height, width = frame.shape[:2]
        start_x = width // 4      # Approximate shooter position
        start_y = height * 3 // 4
        
        rim_x, rim_y = rim
        
        # Calculate current ball position along arc
        current_x = int(start_x + (rim_x - start_x) * progress)
        
        # Parabolic arc
        arc_height = 150  # Maximum arc height
        current_y = int(start_y - arc_height * 4 * progress * (1 - progress) + (rim_y - start_y) * progress)
        
        # Draw trajectory path
        trajectory_points = []
        for t in np.linspace(0, 1, 20):
            x = int(start_x + (rim_x - start_x) * t)
            y = int(start_y - arc_height * 4 * t * (1 - t) + (rim_y - start_y) * t)
            if 0 <= x < width and 0 <= y < height:
                trajectory_points.append((x, y))
        
        # Draw trajectory line
        if len(trajectory_points) > 1:
            for i in range(len(trajectory_points) - 1):
                alpha = i / len(trajectory_points)
                color = (int(255 * (1 - alpha)), int(255 * alpha), 0)
                cv2.line(frame, trajectory_points[i], trajectory_points[i + 1], color, 2)
        
        # Draw current ball position
        if 0 <= current_x < width and 0 <= current_y < height:
            cv2.circle(frame, (current_x, current_y), 12, (0, 255, 0), -1)
            cv2.circle(frame, (current_x, current_y), 15, (255, 255, 255), 2)


def _draw_coaching_insights(frame: np.ndarray, event: Dict, y_start: int) -> int:
    """Draw coaching insights if available"""
    coaching_insights = event.get('coaching_insights', {})
    if not coaching_insights:
        return y_start
    
    y_offset = y_start
    
    # Shot quality assessment
    shot_quality = coaching_insights.get('shot_quality', '')
    if shot_quality:
        quality_color = _get_quality_color(shot_quality)
        y_offset += _put_text_with_background(
            frame, f"Shot Quality: {shot_quality.replace('_', ' ').title()}", 
            (20, y_offset), font_scale=0.6, color=quality_color, bg_color=(0, 0, 0)
        )
    
    # Timing analysis
    timing_analysis = coaching_insights.get('timing_analysis', {})
    if timing_analysis:
        timing_rating = timing_analysis.get('timing_rating', '')
        if timing_rating:
            timing_color = _get_quality_color(timing_rating)
            y_offset += _put_text_with_background(
                frame, f"Timing: {timing_rating.replace('_', ' ').title()}", 
                (20, y_offset), font_scale=0.6, color=timing_color, bg_color=(0, 0, 0)
            )
    
    # Key feedback (limit to top 2 items)
    technical_feedback = coaching_insights.get('technical_feedback', [])
    for feedback in technical_feedback[:2]:
        if isinstance(feedback, str) and len(feedback) < 60:
            y_offset += _put_text_with_background(
                frame, f"• {feedback}", (25, y_offset),
                font_scale=0.5, color=(200, 255, 200), bg_color=(0, 0, 0)
            )
    
    return y_offset


def render_overlay(src_video: str, dst_video: str, events: List[Dict], 
                  rim: Tuple[int, int], fps: float, **kwargs) -> None:
    """
    Enhanced overlay rendering with advanced basketball analysis visualization
    """
    cap = cv2.VideoCapture(src_video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {src_video}")

    # Get video properties
    src_fps = cap.get(cv2.CAP_PROP_FPS) or fps
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    
    if width == 0 or height == 0:
        cap.release()
        raise RuntimeError("Could not get video dimensions")

    # Create output writer with enhanced codec
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    Path(dst_video).parent.mkdir(parents=True, exist_ok=True)
    out = cv2.VideoWriter(dst_video, fourcc, src_fps, (width, height))
    
    if not out.isOpened():
        cap.release()
        raise RuntimeError(f"Could not create output video: {dst_video}")

    frame_idx = 0
    
    try:
        log.info(f"Starting enhanced overlay rendering: {total_frames} frames")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            current_time = frame_idx / src_fps
            
            # Find active events
            active_events = [
                e for e in events 
                if e.get('start', 0) <= current_time <= e.get('end', float('inf'))
            ]
            
            # Enhanced rim visualization
            has_active_shot = len(active_events) > 0
            main_result = active_events[0].get('result', 'unknown') if active_events else 'unknown'
            _draw_enhanced_rim_visualization(frame, rim, has_active_shot, main_result)
            
            # Process each active event with enhanced visualization
            y_offset = 30
            
            for event in active_events:
                result = event.get('result', 'unknown')
                result_color = _get_result_color(result)
                
                # Enhanced main result label
                confidence = event.get('confidence', 0)
                main_label = f"SHOT: {result.upper()} ({confidence*100:.0f}%)"
                
                y_offset += _put_text_with_background(
                    frame, main_label, (20, y_offset),
                    font_scale=1.2, color=result_color, bg_color=(0, 0, 0), thickness=3, padding=8
                )
                
                # Enhanced shot metrics
                metrics = _calculate_enhanced_metrics(event, current_time)
                for key, value in metrics.items():
                    metric_color = (200, 200, 200)
                    
                    # Special coloring for specific metrics
                    if key == "Quality":
                        metric_color = _get_quality_color(value)
                    elif key == "Trajectory":
                        metric_color = _get_quality_color(value)
                    elif key == "Phase":
                        phase_colors = {"Release": (255, 255, 0), "Flight": (255, 165, 0), "Rim": (255, 0, 0)}
                        metric_color = phase_colors.get(value, (200, 200, 200))
                    
                    metric_text = f"{key}: {value}"
                    y_offset += _put_text_with_background(
                        frame, metric_text, (20, y_offset),
                        font_scale=0.6, color=metric_color, bg_color=(0, 0, 0)
                    )
                
                # Draw trajectory visualization
                _draw_trajectory_visualization(frame, event, current_time, rim)
                
                # Enhanced coaching insights
                y_offset = _draw_coaching_insights(frame, event, y_offset + 10)
                
                y_offset += 25  # Space between events
            
            # Enhanced timestamp and info
            info_texts = [
                f"Time: {current_time:.2f}s / {total_frames/src_fps:.1f}s",
                f"Frame: {frame_idx:,} / {total_frames:,}",
                f"Enhanced AI Basketball Coach"
            ]
            
            # Position timestamp in bottom right
            for i, text in enumerate(info_texts):
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                timestamp_pos = (width - text_size[0] - 15, height - 60 + i * 20)
                
                _put_text_with_background(
                    frame, text, timestamp_pos,
                    font_scale=0.5, color=(255, 255, 255), bg_color=(0, 0, 0)
                )
            
            # Frame enhancement indicator
            if events and any(e.get('_enhancement_used', False) for e in events):
                enhancement_text = "🚀 ENHANCED PROCESSING"
                y_offset += _put_text_with_background(
                    frame, enhancement_text, (20, y_offset),
                    font_scale=0.6, color=(0, 255, 255), bg_color=(0, 0, 0)
                )
            
            # Write enhanced frame
            out.write(frame)
            frame_idx += 1
            
            # Enhanced progress logging
            if frame_idx % 300 == 0:  # Every ~10 seconds at 30fps
                progress = (frame_idx / total_frames) * 100 if total_frames > 0 else 0
                log.info(f"Enhanced overlay progress: {current_time:.1f}s processed ({progress:.1f}%)")
    
    finally:
        cap.release()
        out.release()
    
    log.info(f"Enhanced overlay rendering complete: {dst_video}")
    log.info(f"Processed {frame_idx:,} frames with advanced basketball analysis")


# Keep original simple function for compatibility
def _put(img, txt, xy, scale=0.7, color=(0,255,0), thick=2):
    """Original simple text function for backward compatibility"""
    x, y = xy
    cv2.putText(img, txt, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)