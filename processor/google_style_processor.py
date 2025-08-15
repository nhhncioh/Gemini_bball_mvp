# processor/google_style_processor.py
"""
Google-style AI Basketball Coach processor with Real Ball Tracking
Uses actual ball tracking + Vertex AI clustering + Gemini analysis
"""

from __future__ import annotations

import os
import cv2
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
from dataclasses import dataclass
import json
import logging
import math

from celery import Celery
from celery.utils.log import get_task_logger

# Import our real ball tracker
from .ball_tracker import BallTracker, ShotCandidate as TrackingCandidate, BallPosition
from .rim_detector import get_rim_center
from .overlay import render_overlay
from .gemini_processor import classify_with_gemini

logger = get_task_logger(__name__)

@dataclass
class ShotCandidate:
    """Represents a potential shot with real ball tracking data"""
    start_time: float
    end_time: float
    start_frame: int
    end_frame: int
    confidence: float
    ball_trajectory: List[BallPosition]
    release_point: Optional[BallPosition]
    apex_point: Optional[BallPosition]
    rim_approach_frames: List[int]
    shot_type: str
    physics_metrics: Dict[str, Any]

class VertexAIStyleClustering:
    """
    Google Vertex AI clustering approach with real ball tracking
    Uses ball trajectory analysis instead of just motion detection
    """
    
    def __init__(self):
        self.ball_tracker = BallTracker()
        self.min_shot_duration = 1.0
        self.max_shot_duration = 8.0
        self.min_trajectory_points = 8
    
    def detect_shots_with_ball_tracking(self, frames: List[np.ndarray], fps: float, rim_center: Tuple[int, int]) -> List[ShotCandidate]:
        """Detect shots using real ball tracking instead of motion analysis"""
        logger.info("Starting shot detection with real ball tracking...")
        
        # Track ball through entire video
        ball_trajectory = self.ball_tracker.track_ball_trajectory(frames, fps)
        
        if not ball_trajectory:
            logger.warning("No ball trajectory detected - falling back to motion analysis")
            return self._fallback_motion_detection(frames, fps, rim_center)
        
        logger.info(f"Tracked ball through {len(ball_trajectory)} positions")
        
        # Identify shot candidates from trajectory
        tracking_candidates = self.ball_tracker.identify_shot_candidates(ball_trajectory, rim_center, fps)
        
        if not tracking_candidates:
            logger.warning("No shot candidates from trajectory - adding supplementary detection")
            tracking_candidates = self._supplementary_shot_detection(ball_trajectory, rim_center, fps)
        
        # Convert to our shot candidate format
        shot_candidates = []
        for candidate in tracking_candidates:
            converted = self._convert_tracking_candidate(candidate, fps, rim_center)
            if converted:
                shot_candidates.append(converted)
        
        # Add comprehensive coverage if we have too few candidates
        if len(shot_candidates) < 2:
            logger.info("Adding comprehensive coverage candidates")
            coverage_candidates = self._create_coverage_candidates(frames, ball_trajectory, rim_center, fps)
            shot_candidates.extend(coverage_candidates)
        
        # Merge overlapping candidates
        final_candidates = self._merge_overlapping_candidates(shot_candidates)
        
        logger.info(f"Final result: {len(final_candidates)} shot candidates detected with ball tracking")
        return final_candidates
    
    def _convert_tracking_candidate(self, tracking_candidate, fps: float, rim_center: Tuple[int, int]) -> Optional[ShotCandidate]:
        """Convert ball tracking candidate to our shot candidate format"""
        if not tracking_candidate.trajectory:
            return None
        
        start_time = tracking_candidate.start_frame / fps
        end_time = tracking_candidate.end_frame / fps
        
        # Calculate physics metrics
        physics_metrics = self._calculate_physics_metrics(tracking_candidate.trajectory, rim_center, fps)
        
        return ShotCandidate(
            start_time=start_time,
            end_time=end_time,
            start_frame=tracking_candidate.start_frame,
            end_frame=tracking_candidate.end_frame,
            confidence=tracking_candidate.confidence,
            ball_trajectory=tracking_candidate.trajectory,
            release_point=tracking_candidate.release_point,
            apex_point=tracking_candidate.apex_point,
            rim_approach_frames=tracking_candidate.rim_approach_frames,
            shot_type=tracking_candidate.shot_type,
            physics_metrics=physics_metrics
        )
    
    def _calculate_physics_metrics(self, trajectory: List[BallPosition], rim_center: Tuple[int, int], fps: float) -> Dict[str, Any]:
        """Calculate physics-based metrics for shot analysis"""
        if len(trajectory) < 3:
            return {}
        
        metrics = {}
        
        # Calculate arc height
        y_positions = [pos.y for pos in trajectory]
        max_height = min(y_positions)  # Min Y is max height (inverted coordinates)
        min_height = max(y_positions)
        arc_height = min_height - max_height
        metrics['arc_height_pixels'] = arc_height
        
        # Calculate approximate release angle
        if len(trajectory) >= 3:
            start_pos = trajectory[0]
            mid_pos = trajectory[len(trajectory) // 3]
            
            dx = mid_pos.x - start_pos.x
            dy = mid_pos.y - start_pos.y  # Note: Y increases downward
            
            if dx != 0:
                angle_rad = math.atan(-dy / dx)  # Negative because Y is inverted
                angle_deg = math.degrees(angle_rad)
                metrics['release_angle_degrees'] = angle_deg
        
        # Calculate duration
        if trajectory:
            duration = (trajectory[-1].frame - trajectory[0].frame) / fps
            metrics['duration_seconds'] = duration
        
        # Check if trajectory reaches rim area
        rim_x, rim_y = rim_center
        rim_distances = [
            math.sqrt((pos.x - rim_x)**2 + (pos.y - rim_y)**2)
            for pos in trajectory
        ]
        min_rim_distance = min(rim_distances)
        metrics['final_rim_distance_pixels'] = min_rim_distance
        metrics['reaches_rim'] = min_rim_distance <= 100
        
        return metrics
    
    def _fallback_motion_detection(self, frames: List[np.ndarray], fps: float, rim_center: Tuple[int, int]) -> List[ShotCandidate]:
        """Fallback when ball tracking fails completely"""
        logger.info("Using fallback motion detection")
        
        candidates = []
        video_duration = len(frames) / fps
        
        # Create basic coverage candidates
        segment_duration = 6.0
        overlap = 3.0
        current_time = 0.0
        
        while current_time < video_duration - 3.0:
            end_time = min(current_time + segment_duration, video_duration)
            
            start_frame = int(current_time * fps)
            end_frame = int(end_time * fps)
            
            candidate = ShotCandidate(
                start_time=current_time,
                end_time=end_time,
                start_frame=start_frame,
                end_frame=end_frame,
                confidence=0.3,  # Lower confidence for fallback
                ball_trajectory=[],
                release_point=None,
                apex_point=None,
                rim_approach_frames=[],
                shot_type='inferred',
                physics_metrics={}
            )
            
            candidates.append(candidate)
            current_time += overlap
        
        return candidates[:5]  # Limit to 5 segments
    
    def _supplementary_shot_detection(self, ball_trajectory: List[BallPosition], rim_center: Tuple[int, int], fps: float) -> List[TrackingCandidate]:
        """Add supplementary detection when primary methods miss shots"""
        logger.info("Adding supplementary shot detection")
        
        candidates = []
        
        if not ball_trajectory:
            return candidates
        
        # Look for ball positions near rim
        rim_x, rim_y = rim_center
        rim_threshold = 150
        
        rim_frames = []
        for pos in ball_trajectory:
            distance = math.sqrt((pos.x - rim_x)**2 + (pos.y - rim_y)**2)
            if distance <= rim_threshold:
                rim_frames.append(pos.frame)
        
        if rim_frames:
            # Group consecutive rim frames
            groups = []
            current_group = [rim_frames[0]]
            
            for frame in rim_frames[1:]:
                if frame - current_group[-1] <= 30:
                    current_group.append(frame)
                else:
                    groups.append(current_group)
                    current_group = [frame]
            groups.append(current_group)
            
            # Create candidates from groups
            for group in groups:
                if len(group) >= 3:
                    start_frame = max(0, group[0] - 60)
                    end_frame = min(len(ball_trajectory), group[-1] + 60)
                    
                    trajectory_positions = [
                        pos for pos in ball_trajectory
                        if start_frame <= pos.frame <= end_frame
                    ]
                    
                    if trajectory_positions:
                        candidate = TrackingCandidate(
                            start_frame=start_frame,
                            end_frame=end_frame,
                            trajectory=trajectory_positions,
                            release_point=trajectory_positions[0],
                            apex_point=None,
                            rim_approach_frames=group,
                            confidence=0.6,
                            shot_type='supplementary'
                        )
                        candidates.append(candidate)
        
        return candidates
    
    def _create_coverage_candidates(self, frames: List[np.ndarray], ball_trajectory: List[BallPosition], rim_center: Tuple[int, int], fps: float) -> List[ShotCandidate]:
        """Create additional candidates to ensure comprehensive coverage"""
        logger.info("Creating coverage candidates for comprehensive analysis")
        
        candidates = []
        video_duration = len(frames) / fps
        
        # Create strategic time windows
        time_windows = [
            (2.0, 8.0),    # Early game action
            (8.0, 15.0),   # Mid sequence
            (15.0, 22.0),  # Later action
            (max(0, video_duration - 10.0), video_duration)  # End sequence
        ]
        
        for start_time, end_time in time_windows:
            if start_time >= video_duration:
                break
                
            end_time = min(end_time, video_duration)
            if end_time - start_time < 3.0:
                continue
            
            start_frame = int(start_time * fps)
            end_frame = int(end_time * fps)
            
            # Find ball positions in this window
            window_trajectory = [
                pos for pos in ball_trajectory
                if start_frame <= pos.frame <= end_frame
            ]
            
            candidate = ShotCandidate(
                start_time=start_time,
                end_time=end_time,
                start_frame=start_frame,
                end_frame=end_frame,
                confidence=0.5,
                ball_trajectory=window_trajectory,
                release_point=window_trajectory[0] if window_trajectory else None,
                apex_point=None,
                rim_approach_frames=[],
                shot_type='coverage',
                physics_metrics={}
            )
            
            candidates.append(candidate)
        
        return candidates
    
    def _merge_overlapping_candidates(self, candidates: List[ShotCandidate]) -> List[ShotCandidate]:
        """Merge candidates that overlap significantly"""
        if len(candidates) <= 1:
            return candidates
        
        candidates.sort(key=lambda c: c.start_time)
        merged = [candidates[0]]
        
        for candidate in candidates[1:]:
            last_merged = merged[-1]
            
            # Check for significant overlap
            overlap_start = max(last_merged.start_time, candidate.start_time)
            overlap_end = min(last_merged.end_time, candidate.end_time)
            overlap_duration = max(0, overlap_end - overlap_start)
            
            min_duration = min(
                last_merged.end_time - last_merged.start_time,
                candidate.end_time - candidate.start_time
            )
            
            if overlap_duration > min_duration * 0.5:  # 50% overlap
                # Merge candidates
                merged_candidate = ShotCandidate(
                    start_time=min(last_merged.start_time, candidate.start_time),
                    end_time=max(last_merged.end_time, candidate.end_time),
                    start_frame=min(last_merged.start_frame, candidate.start_frame),
                    end_frame=max(last_merged.end_frame, candidate.end_frame),
                    confidence=max(last_merged.confidence, candidate.confidence),
                    ball_trajectory=last_merged.ball_trajectory + candidate.ball_trajectory,
                    release_point=last_merged.release_point or candidate.release_point,
                    apex_point=last_merged.apex_point or candidate.apex_point,
                    rim_approach_frames=last_merged.rim_approach_frames + candidate.rim_approach_frames,
                    shot_type='merged',
                    physics_metrics={**last_merged.physics_metrics, **candidate.physics_metrics}
                )
                merged[-1] = merged_candidate
            else:
                merged.append(candidate)
        
        return merged


class GeminiCoachingAnalyzer:
    """
    Google-style Gemini integration for coaching analysis with real ball tracking
    """
    
    def __init__(self):
        self.rim_center = None
    
    def analyze_shot_with_coaching(self, frames: List[np.ndarray], candidate: ShotCandidate, 
                                 rim_center: Tuple[int, int], fps: float) -> Dict[str, Any]:
        """Analyze shot and provide coaching insights using real ball tracking data"""
        
        # Prepare window for Gemini analysis
        window = (candidate.start_frame, candidate.end_frame)
        
        # Prepare ball tracking data from real trajectory
        ball_track = []
        for pos in candidate.ball_trajectory:
            ball_track.append({
                'frame': pos.frame,
                'x': pos.x,
                'y': pos.y,
                'confidence': pos.confidence,
                'time': pos.frame / fps
            })
        
        # Get basic classification from Gemini
        basic_result = classify_with_gemini(
            rim=rim_center,
            frames=frames,
            window=window,
            fps=fps,
            ball_track=ball_track,
            shooting_metrics=candidate.physics_metrics,
            max_images=10,
            temperature=0.1
        )
        
        # Enhance with coaching analysis
        coaching_analysis = self._generate_coaching_insights(
            basic_result, candidate, rim_center
        )
        
        # Combine results
        result = {
            **basic_result,
            'coaching_insights': coaching_analysis,
            'shot_metrics': {
                'duration': candidate.end_time - candidate.start_time,
                'trajectory_points': len(candidate.ball_trajectory),
                'tracking_confidence': candidate.confidence,
                'shot_type': candidate.shot_type,
                'physics_metrics': candidate.physics_metrics,
                'rim_approach_frames': len(candidate.rim_approach_frames)
            }
        }
        
        return result
    
    def _generate_coaching_insights(self, basic_result: Dict, candidate: ShotCandidate, 
                                  rim_center: Tuple[int, int]) -> Dict[str, Any]:
        """Generate coaching insights using real ball tracking data"""
        
        outcome = basic_result.get('outcome', 'unknown')
        confidence = basic_result.get('confidence', 0.0)
        duration = candidate.end_time - candidate.start_time
        
        insights = {
            'shot_quality': 'good' if confidence > 0.8 else 'needs_improvement',
            'timing_analysis': {
                'shot_duration': duration,
                'timing_rating': 'good' if 1.0 <= duration <= 4.0 else 'too_fast_or_slow'
            },
            'technical_feedback': [],
            'improvement_suggestions': [],
            'ball_tracking_insights': []
        }
        
        # Ball tracking specific insights
        if candidate.ball_trajectory:
            trajectory_length = len(candidate.ball_trajectory)
            insights['ball_tracking_insights'].append(f"Tracked ball through {trajectory_length} positions")
            
            if candidate.release_point:
                insights['ball_tracking_insights'].append(f"Release detected at frame {candidate.release_point.frame}")
            
            if candidate.apex_point:
                insights['ball_tracking_insights'].append(f"Shot apex at frame {candidate.apex_point.frame}")
            
            if candidate.rim_approach_frames:
                insights['ball_tracking_insights'].append(f"Ball approached rim in {len(candidate.rim_approach_frames)} frames")
        
        # Physics-based insights
        if candidate.physics_metrics:
            arc_height = candidate.physics_metrics.get('arc_height_pixels', 0)
            if arc_height > 100:
                insights['technical_feedback'].append("Good shot arc height detected")
            elif arc_height < 50:
                insights['improvement_suggestions'].append("Try shooting with a higher arc")
            
            release_angle = candidate.physics_metrics.get('release_angle_degrees')
            if release_angle:
                if 35 <= release_angle <= 55:
                    insights['technical_feedback'].append(f"Excellent release angle: {release_angle:.1f}°")
                else:
                    insights['improvement_suggestions'].append(f"Adjust release angle (current: {release_angle:.1f}°, optimal: 45°)")
            
            reaches_rim = candidate.physics_metrics.get('reaches_rim', False)
            if not reaches_rim:
                insights['improvement_suggestions'].append("Shot didn't reach rim area - increase power")
        
        # Outcome-based feedback
        if outcome == 'made':
            insights['technical_feedback'].append("Successful shot! Ball went through the rim.")
            if candidate.shot_type == 'complete':
                insights['technical_feedback'].append("Complete trajectory tracked - excellent shot mechanics.")
        elif outcome == 'missed':
            insights['technical_feedback'].append("Shot missed the target.")
            insights['improvement_suggestions'].append("Focus on follow-through and consistent release.")
            
            if candidate.shot_type == 'partial':
                insights['improvement_suggestions'].append("Incomplete trajectory detected - work on consistent shooting motion.")
        
        return insights


def load_video_frames(video_path: str, max_size: int = 720) -> Tuple[List[np.ndarray], float]:
    """Load video frames - same as before"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [], 30.0
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Downscale if needed
        height, width = frame.shape[:2]
        if max(height, width) > max_size:
            scale = max_size / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        frames.append(frame)
    
    cap.release()
    return frames, fps


def process_video_google_style(video_path: str) -> Dict[str, Any]:
    """
    Main processing function using Google's approach with real ball tracking
    """
    logger.info("Starting Google-style AI Basketball Coach processing with real ball tracking")
    
    # 1. Load video
    frames, fps = load_video_frames(video_path)
    if not frames:
        return {"error": "Could not load video frames"}
    
    logger.info(f"Loaded {len(frames)} frames at {fps:.1f} FPS")
    
    # 2. Detect rim
    rim_center = get_rim_center(frames[len(frames)//2])
    logger.info(f"Rim detected at: {rim_center}")
    
    # 3. Use real ball tracking + Vertex AI style clustering
    clustering_engine = VertexAIStyleClustering()
    
    # Detect shots using real ball tracking
    shot_candidates = clustering_engine.detect_shots_with_ball_tracking(frames, fps, rim_center)
    
    logger.info(f"Found {len(shot_candidates)} shot candidates via ball tracking")
    
    # 4. Gemini analysis with coaching insights
    coaching_analyzer = GeminiCoachingAnalyzer()
    
    shot_results = []
    for i, candidate in enumerate(shot_candidates):
        logger.info(f"Analyzing shot candidate {i+1}: {candidate.start_time:.1f}s-{candidate.end_time:.1f}s")
        
        analysis = coaching_analyzer.analyze_shot_with_coaching(
            frames, candidate, rim_center, fps
        )
        
        # Convert to standard event format
        event = {
            "type": "shot",
            "start": round(candidate.start_time, 3),
            "end": round(candidate.end_time, 3),
            "result": analysis.get("outcome", "unknown"),
            "player": "unknown",
            "confidence": analysis.get("confidence", 0.0),
            "review": analysis.get("reason", "Shot analyzed by AI Basketball Coach with ball tracking"),
            "key_evidence": analysis.get("key_evidence", []),
            
            # Enhanced data from ball tracking
            "coaching_insights": analysis.get("coaching_insights", {}),
            "shot_metrics": analysis.get("shot_metrics", {}),
            "ball_tracking_data": {
                "trajectory_points": len(candidate.ball_trajectory),
                "release_point": {
                    "x": candidate.release_point.x if candidate.release_point else 0,
                    "y": candidate.release_point.y if candidate.release_point else 0,
                    "frame": candidate.release_point.frame if candidate.release_point else 0
                } if candidate.release_point else None,
                "apex_point": {
                    "x": candidate.apex_point.x if candidate.apex_point else 0,
                    "y": candidate.apex_point.y if candidate.apex_point else 0,
                    "frame": candidate.apex_point.frame if candidate.apex_point else 0
                } if candidate.apex_point else None,
                "rim_approach_frames": candidate.rim_approach_frames,
                "shot_type": candidate.shot_type
            },
            "physics_metrics": candidate.physics_metrics,
            "_analysis_method": "google_style_ball_tracking",
            "_tracking_confidence": candidate.confidence
        }
        
        shot_results.append(event)
        
        logger.info(f"Shot {i+1} classified as: {event['result']} (confidence: {event['confidence']:.2f})")
    
    return {
        "shots": shot_results,
        "total_shots": len(shot_results),
        "processing_method": "google_style_ball_tracking",
        "rim_center": rim_center,
        "video_stats": {
            "frames": len(frames),
            "fps": fps,
            "duration": len(frames) / fps
        }
    }