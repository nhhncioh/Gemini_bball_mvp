# processor/ball_tracker.py
"""
Real ball tracking using color detection and motion analysis
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class BallPosition:
    """Represents ball position at a specific frame"""
    frame: int
    x: float
    y: float
    confidence: float
    time: float

@dataclass
class TrajectorySegment:
    """Represents a segment of ball trajectory"""
    start_idx: int
    end_idx: int
    positions: List[BallPosition]
    is_ascending: bool
    max_height: float
    duration: float

class BallTracker:
    """Track basketball through video using color detection"""
    
    def __init__(self):
        # Basketball color ranges (orange/brown)
        self.lower_orange = np.array([5, 100, 100])
        self.upper_orange = np.array([15, 255, 255])
        
        # Alternative color range for different lighting
        self.lower_brown = np.array([10, 50, 50])
        self.upper_brown = np.array([20, 255, 200])
        
        # Detection parameters
        self.min_ball_area = 100
        self.max_ball_area = 5000
        self.min_circularity = 0.6
        
    def track_ball(self, frames: List[np.ndarray], fps: float) -> List[BallPosition]:
        """Track ball through all frames"""
        trajectory = []
        
        for i, frame in enumerate(frames):
            if i % 100 == 0:
                logger.info(f"Processing frame {i}/{len(frames)}")
            
            position = self.detect_ball_in_frame(frame)
            
            if position:
                ball_pos = BallPosition(
                    frame=i,
                    x=position[0],
                    y=position[1],
                    confidence=position[2] if len(position) > 2 else 0.8,
                    time=i / fps
                )
            else:
                # Use interpolation or last known position
                if trajectory:
                    last_pos = trajectory[-1]
                    ball_pos = BallPosition(
                        frame=i,
                        x=last_pos.x,
                        y=last_pos.y,
                        confidence=0.3,  # Low confidence for interpolated
                        time=i / fps
                    )
                else:
                    # Default position if no ball detected yet
                    h, w = frame.shape[:2]
                    ball_pos = BallPosition(
                        frame=i,
                        x=w // 2,
                        y=h // 2,
                        confidence=0.1,
                        time=i / fps
                    )
            
            trajectory.append(ball_pos)
        
        # Smooth trajectory
        trajectory = self._smooth_trajectory(trajectory)
        
        logger.info(f"Built trajectory with {len(trajectory)} positions")
        return trajectory
    
    def detect_ball_in_frame(self, frame: np.ndarray) -> Optional[Tuple[float, float, float]]:
        """Detect ball in a single frame using color and shape detection"""
        
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create masks for orange and brown colors
        mask_orange = cv2.inRange(hsv, self.lower_orange, self.upper_orange)
        mask_brown = cv2.inRange(hsv, self.lower_brown, self.upper_brown)
        
        # Combine masks
        mask = cv2.bitwise_or(mask_orange, mask_brown)
        
        # Morphological operations to remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best_ball = None
        best_score = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by area
            if area < self.min_ball_area or area > self.max_ball_area:
                continue
            
            # Calculate circularity
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            if circularity < self.min_circularity:
                continue
            
            # Get center and radius
            (x, y), radius = cv2.minEnclosingCircle(contour)
            
            # Score based on circularity and size
            score = circularity * np.sqrt(area)
            
            if score > best_score:
                best_score = score
                confidence = min(0.95, circularity)
                best_ball = (float(x), float(y), confidence)
        
        return best_ball
    
    def _smooth_trajectory(self, trajectory: List[BallPosition], window: int = 5) -> List[BallPosition]:
        """Smooth the trajectory using moving average"""
        if len(trajectory) < window:
            return trajectory
        
        smoothed = []
        half_window = window // 2
        
        for i in range(len(trajectory)):
            start = max(0, i - half_window)
            end = min(len(trajectory), i + half_window + 1)
            
            window_positions = trajectory[start:end]
            
            # Average position in window
            avg_x = np.mean([p.x for p in window_positions])
            avg_y = np.mean([p.y for p in window_positions])
            
            # Keep original confidence and time
            smoothed_pos = BallPosition(
                frame=trajectory[i].frame,
                x=avg_x,
                y=avg_y,
                confidence=trajectory[i].confidence,
                time=trajectory[i].time
            )
            smoothed.append(smoothed_pos)
        
        return smoothed
    
    def identify_shot_candidates(self, trajectory: List[BallPosition], 
                                rim_center: Tuple[int, int], fps: float) -> List[Dict[str, Any]]:
        """Identify shot candidates from ball trajectory"""
        
        if len(trajectory) < 30:  # Need minimum frames
            return []
        
        # Segment trajectory into potential shots
        segments = self._segment_trajectory(trajectory)
        
        candidates = []
        for i, segment in enumerate(segments):
            if self._is_shot_segment(segment, rim_center):
                candidate = self._create_candidate_from_segment(segment, i, rim_center, fps)
                candidates.append(candidate)
        
        return candidates
    
    def _segment_trajectory(self, trajectory: List[BallPosition]) -> List[TrajectorySegment]:
        """Segment trajectory into distinct shot attempts based on velocity changes"""
        if len(trajectory) < 3:
            return []
        
        # Calculate velocities
        velocities_y = []
        for i in range(1, len(trajectory)):
            dt = trajectory[i].time - trajectory[i-1].time
            if dt > 0:
                vy = (trajectory[i].y - trajectory[i-1].y) / dt
                velocities_y.append(vy)
            else:
                velocities_y.append(0)
        
        # Find significant velocity changes (potential shot boundaries)
        segments = []
        start_idx = 0
        
        for i in range(1, len(velocities_y)):
            # Detect sign change in velocity (apex of trajectory)
            if velocities_y[i-1] * velocities_y[i] < 0:
                # Create segment
                change_idx = i
                
                # Check if any velocities in the range are negative (ascending)
                velocity_slice = velocities_y[max(0, start_idx-1):change_idx]
                is_ascending = any(v < 0 for v in velocity_slice) if velocity_slice else False
                
                if change_idx - start_idx > 10:  # Minimum segment size
                    segment = TrajectorySegment(
                        start_idx=start_idx,
                        end_idx=change_idx,
                        positions=trajectory[start_idx:change_idx+1],
                        is_ascending=is_ascending,
                        max_height=min(p.y for p in trajectory[start_idx:change_idx+1]),
                        duration=(trajectory[change_idx].time - trajectory[start_idx].time)
                    )
                    segments.append(segment)
                    start_idx = change_idx
        
        # Add final segment if substantial
        if len(trajectory) - start_idx > 10:
            velocity_slice = velocities_y[start_idx:] if start_idx < len(velocities_y) else []
            is_ascending = any(v < 0 for v in velocity_slice) if velocity_slice else False
            
            segment = TrajectorySegment(
                start_idx=start_idx,
                end_idx=len(trajectory)-1,
                positions=trajectory[start_idx:],
                is_ascending=is_ascending,
                max_height=min(p.y for p in trajectory[start_idx:]),
                duration=(trajectory[-1].time - trajectory[start_idx].time)
            )
            segments.append(segment)
        
        return segments
    
    def _is_shot_segment(self, segment: TrajectorySegment, rim_center: Tuple[int, int]) -> bool:
        """Determine if a segment represents a shot attempt"""
        
        # Check if segment has upward motion
        if not segment.is_ascending:
            return False
        
        # Check if trajectory gets near rim
        rim_x, rim_y = rim_center
        min_distance = float('inf')
        
        for pos in segment.positions:
            distance = np.sqrt((pos.x - rim_x)**2 + (pos.y - rim_y)**2)
            min_distance = min(min_distance, distance)
        
        # Shot should come within reasonable distance of rim
        if min_distance > 200:  # pixels
            return False
        
        # Check for reasonable arc height
        start_y = segment.positions[0].y
        height_change = start_y - segment.max_height
        
        if height_change < 50:  # Minimum arc height in pixels
            return False
        
        return True
    
    def _create_candidate_from_segment(self, segment: TrajectorySegment, index: int,
                                      rim_center: Tuple[int, int], fps: float) -> Dict[str, Any]:
        """Create shot candidate from trajectory segment"""
        
        # Calculate shot metrics
        start_pos = segment.positions[0]
        end_pos = segment.positions[-1]
        
        # Find apex
        apex_pos = min(segment.positions, key=lambda p: p.y)
        
        # Calculate arc height
        arc_height = start_pos.y - apex_pos.y
        
        # Distance to rim at end
        rim_x, rim_y = rim_center
        final_distance = np.sqrt((end_pos.x - rim_x)**2 + (end_pos.y - rim_y)**2)
        
        # Build ball tracking data
        ball_track = []
        for pos in segment.positions:
            ball_track.append({
                'frame': pos.frame,
                'x': pos.x,
                'y': pos.y,
                'confidence': pos.confidence,
                'time': pos.time
            })
        
        return {
            'id': f'shot_{index+1}',
            'start': start_pos.time,
            'end': end_pos.time,
            'start_f': start_pos.frame,
            'end_f': end_pos.frame,
            'confidence': 0.85,  # Base confidence
            'ball_track': ball_track,
            'shooting_metrics': {
                'arc_height_pixels': arc_height,
                'final_rim_distance_pixels': final_distance,
                'duration_seconds': segment.duration,
                'apex_frame': apex_pos.frame,
                'release_frame': start_pos.frame,
                'release_point': (start_pos.x, start_pos.y),
                'apex_point': (apex_pos.x, apex_pos.y),
                'end_point': (end_pos.x, end_pos.y)
            }
        }

# Create a default instance
default_tracker = BallTracker()