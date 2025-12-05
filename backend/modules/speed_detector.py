"""
Speed Detector - Calculate speed of detected objects
"""
import numpy as np
from typing import Dict, Optional, List, Tuple
from datetime import datetime, timedelta
from collections import deque
from backend.utils.logger import setup_logger

logger = setup_logger('speed_detector')


class SpeedDetector:
    """Calculate speed of detected objects using frame tracking"""
    
    def __init__(self, fps: int = 30, smoothing_window: int = 5):
        """
        Initialize speed detector
        
        Args:
            fps: Frames per second of camera
            smoothing_window: Number of frames for speed smoothing
        """
        self.fps = fps
        self.smoothing_window = smoothing_window
        
        # Track objects across frames
        # Format: {object_id: deque([(timestamp, x, y, distance), ...])}
        self.object_tracks = {}
        self.next_object_id = 0
        
        # Speed history for smoothing
        self.speed_history = {}
        
        logger.info(f"Speed detector initialized: {fps} FPS, smoothing window={smoothing_window}")
    
    def track_object(
        self, 
        detection: Dict, 
        distance: float, 
        timestamp: datetime = None
    ) -> Tuple[int, Optional[float]]:
        """
        Track object and calculate speed
        
        Args:
            detection: Detection dictionary with bbox
            distance: Distance to object in meters
            timestamp: Timestamp of detection
        
        Returns:
            Tuple of (object_id, speed_mps) where speed is in meters/second
        """
        timestamp = timestamp or datetime.now()
        
        # Get center of bounding box
        bbox = detection['bbox']
        center_x = (bbox['x1'] + bbox['x2']) / 2
        center_y = (bbox['y1'] + bbox['y2']) / 2
        
        # Find matching object or create new
        object_id = self._find_matching_object(center_x, center_y, bbox)
        
        if object_id is None:
            # New object
            object_id = self.next_object_id
            self.next_object_id += 1
            self.object_tracks[object_id] = deque(maxlen=self.smoothing_window)
            self.speed_history[object_id] = deque(maxlen=self.smoothing_window)
        
        # Add to track
        self.object_tracks[object_id].append((timestamp, center_x, center_y, distance))
        
        # Calculate speed if we have enough history
        speed = self._calculate_speed(object_id)
        
        if speed is not None:
            self.speed_history[object_id].append(speed)
            # Return smoothed speed
            smoothed_speed = np.mean(list(self.speed_history[object_id]))
            return object_id, smoothed_speed
        
        return object_id, None
    
    def _find_matching_object(
        self, 
        center_x: float, 
        center_y: float, 
        bbox: Dict, 
        max_distance: float = 100
    ) -> Optional[int]:
        """
        Find existing object that matches this detection
        
        Args:
            center_x: X coordinate of detection center
            center_y: Y coordinate of detection center
            bbox: Bounding box dictionary
            max_distance: Maximum pixel distance for matching
        
        Returns:
            Object ID if found, None otherwise
        """
        best_match = None
        best_distance = max_distance
        
        for obj_id, track in self.object_tracks.items():
            if len(track) == 0:
                continue
            
            # Get last position
            last_timestamp, last_x, last_y, last_dist = track[-1]
            
            # Calculate distance
            pixel_distance = np.sqrt((center_x - last_x) ** 2 + (center_y - last_y) ** 2)
            
            # Check if within matching distance
            if pixel_distance < best_distance:
                # Also check time difference (max 1 second)
                time_diff = (datetime.now() - last_timestamp).total_seconds()
                if time_diff < 1.0:
                    best_distance = pixel_distance
                    best_match = obj_id
        
        return best_match
    
    def _calculate_speed(self, object_id: int) -> Optional[float]:
        """
        Calculate speed for tracked object
        
        Args:
            object_id: Object ID
        
        Returns:
            Speed in meters/second, or None if insufficient data
        """
        track = self.object_tracks.get(object_id)
        
        if track is None or len(track) < 2:
            return None
        
        # Get first and last positions
        first_time, first_x, first_y, first_dist = track[0]
        last_time, last_x, last_y, last_dist = track[-1]
        
        # Time difference in seconds
        time_diff = (last_time - first_time).total_seconds()
        
        if time_diff == 0:
            return None
        
        # Pixel displacement
        pixel_disp = np.sqrt((last_x - first_x) ** 2 + (last_y - first_y) ** 2)
        
        # Convert pixel displacement to meters
        # Assuming average distance for conversion
        avg_distance = (first_dist + last_dist) / 2
        
        # Estimate meters per pixel at this distance
        # This is approximate and depends on camera FOV
        # For a typical webcam with 60° H-FOV at distance D:
        # FOV_width_meters = 2 * D * tan(60°/2) ≈ 1.15 * D
        # meters_per_pixel = FOV_width_meters / image_width
        
        fov_half_angle = np.radians(30)  # 60° / 2
        fov_width = 2 * avg_distance * np.tan(fov_half_angle)
        image_width = 1280  # Assume 1280px width
        meters_per_pixel = fov_width / image_width
        
        # Real displacement in meters
        real_displacement = pixel_disp * meters_per_pixel
        
        # Speed in meters/second
        speed = real_displacement / time_diff
        
        logger.debug(f"Object {object_id}: {speed:.2f} m/s "
                    f"(displacement={real_displacement:.2f}m, time={time_diff:.2f}s)")
        
        return speed
    
    def get_speed_kmh(self, speed_mps: float) -> float:
        """Convert speed from m/s to km/h"""
        return speed_mps * 3.6
    
    def get_speed_knots(self, speed_mps: float) -> float:
        """Convert speed from m/s to knots"""
        return speed_mps * 1.944
    
    def cleanup_old_tracks(self, max_age_seconds: int = 5):
        """
        Remove old tracks that haven't been updated
        
        Args:
            max_age_seconds: Maximum age for tracks in seconds
        """
        current_time = datetime.now()
        objects_to_remove = []
        
        for obj_id, track in self.object_tracks.items():
            if len(track) == 0:
                objects_to_remove.append(obj_id)
                continue
            
            last_time, _, _, _ = track[-1]
            age = (current_time - last_time).total_seconds()
            
            if age > max_age_seconds:
                objects_to_remove.append(obj_id)
        
        for obj_id in objects_to_remove:
            del self.object_tracks[obj_id]
            if obj_id in self.speed_history:
                del self.speed_history[obj_id]
        
        if objects_to_remove:
            logger.debug(f"Cleaned up {len(objects_to_remove)} old tracks")
    
    def get_stats(self) -> Dict:
        """Get speed detector statistics"""
        return {
            'active_tracks': len(self.object_tracks),
            'fps': self.fps,
            'smoothing_window': self.smoothing_window
        }


# Example usage
if __name__ == "__main__":
    detector = SpeedDetector(fps=30)
    
    # Simulate tracking an object
    test_detections = [
        {'bbox': {'x1': 100, 'y1': 100, 'x2': 150, 'y2': 130}},  # Frame 0
        {'bbox': {'x1': 120, 'y1': 100, 'x2': 170, 'y2': 130}},  # Frame 1 (moved right)
        {'bbox': {'x1': 140, 'y1': 100, 'x2': 190, 'y2': 130}},  # Frame 2 (moved right)
    ]
    
    distances = [1000, 1020, 1040]  # meters
    
    import time
    for i, (det, dist) in enumerate(zip(test_detections, distances)):
        obj_id, speed = detector.track_object(det, dist)
        
        if speed:
            print(f"Frame {i}: Object {obj_id} speed = {speed:.2f} m/s "
                  f"({detector.get_speed_kmh(speed):.2f} km/h)")
        else:
            print(f"Frame {i}: Object {obj_id} (calculating...)")
        
        time.sleep(0.1)  # Simulate frame delay
    
    print(f"Stats: {detector.get_stats()}")
