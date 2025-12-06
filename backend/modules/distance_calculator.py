"""
Distance Calculator - Calculate distance to detected aircraft using camera and GPS
"""
import math
import numpy as np
from typing import Dict, Optional, Tuple
from backend.utils.logger import setup_logger

logger = setup_logger('distance_calculator')


class DistanceCalculator:
    """Calculate distance to detected objects using camera parameters and GPS"""
    
    def __init__(self, camera_focal_length: float = None, sensor_width: float = None):
        """
        Initialize distance calculator
        
        Args:
            camera_focal_length: Camera focal length in mm (default: estimated from FOV)
            sensor_width: Camera sensor width in mm (default: 6.17mm for typical webcam)
        """
        # Default values for typical webcam
        self.sensor_width = sensor_width or 6.17  # mm
        self.camera_focal_length = camera_focal_length or 3.6  # mm (typical webcam)
        
        # Known aircraft dimensions (approximate)
        self.aircraft_sizes = {
            'small_aircraft': 10.0,  # meters (Cessna)
            'medium_aircraft': 40.0,  # meters (Boeing 737)
            'large_aircraft': 70.0,  # meters (Boeing 777)
            'drone': 0.5  # meters (typical consumer drone)
        }
        
        logger.info(f"Distance calculator initialized: focal={self.camera_focal_length}mm, sensor={self.sensor_width}mm")
    
    def calculate_distance_pinhole(
        self, 
        pixel_width: int, 
        real_width: float, 
        image_width: int
    ) -> float:
        """
        Calculate distance using pinhole camera model
        
        Formula: Distance = (Real_Width × Focal_Length × Image_Width) / (Pixel_Width × Sensor_Width)
        
        Args:
            pixel_width: Width of object in pixels
            real_width: Real width of object in meters
            image_width: Image width in pixels
        
        Returns:
            Distance in meters
        """
        if pixel_width == 0:
            return 0.0
        
        # Convert to meters
        focal_length_m = self.camera_focal_length / 1000
        sensor_width_m = self.sensor_width / 1000
        
        distance = (real_width * focal_length_m * image_width) / (pixel_width * sensor_width_m)
        
        return distance
    
    def estimate_aircraft_size(self, detection: Dict, classification: str) -> float:
        """
        Estimate aircraft real-world size based on classification
        
        Args:
            detection: Detection dictionary
            classification: Aircraft classification
        
        Returns:
            Estimated size in meters
        """
        if classification == 'drone':
            return self.aircraft_sizes['drone']
        elif classification == 'bird':
            return 0.5  # Average bird wingspan
        elif classification == 'aircraft':
            # Estimate based on relative size in frame
            bbox = detection['bbox']
            relative_width = bbox['width'] / 1280  # Assuming 1280px width
            
            if relative_width < 0.1:
                return self.aircraft_sizes['small_aircraft']
            elif relative_width < 0.3:
                return self.aircraft_sizes['medium_aircraft']
            else:
                return self.aircraft_sizes['large_aircraft']
        else:
            # Unknown - use medium aircraft as default
            return self.aircraft_sizes['medium_aircraft']
    
    def calculate_distance(
        self, 
        detection: Dict, 
        classification: str, 
        image_width: int = 1280
    ) -> float:
        """
        Calculate distance to detected object
        
        Args:
            detection: Detection dictionary with bbox
            classification: Object classification
            image_width: Image width in pixels
        
        Returns:
            Distance in meters
        """
        try:
            bbox = detection['bbox']
            pixel_width = bbox['width']
            
            # Estimate real width
            real_width = self.estimate_aircraft_size(detection, classification)
            
            # Calculate distance using pinhole model
            distance = self.calculate_distance_pinhole(pixel_width, real_width, image_width)
            
            logger.debug(f"Distance calculated: {distance:.2f}m for {classification} "
                        f"(pixel_width={pixel_width}, real_width={real_width:.2f}m)")
            
            return max(distance, 0.0)  # Ensure non-negative
            
        except Exception as e:
            logger.error(f"Error calculating distance: {e}")
            return 0.0
    
    def calculate_altitude_difference(
        self, 
        camera_altitude: float, 
        detection_angle: float, 
        distance: float
    ) -> float:
        """
        Calculate altitude difference using trigonometry
        
        Args:
            camera_altitude: Camera altitude in meters
            detection_angle: Vertical angle of detection (degrees from horizon)
            distance: Horizontal distance in meters
        
        Returns:
            Altitude difference in meters (positive = above camera)
        """
        angle_rad = math.radians(detection_angle)
        altitude_diff = distance * math.tan(angle_rad)
        
        return altitude_diff
    
    def calculate_gps_distance(
        self, 
        lat1: float, 
        lon1: float, 
        lat2: float, 
        lon2: float
    ) -> float:
        """
        Calculate distance between two GPS coordinates using Haversine formula
        
        Args:
            lat1, lon1: First coordinate
            lat2, lon2: Second coordinate
        
        Returns:
            Distance in meters
        """
        R = 6371000  # Earth radius in meters
        
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lon = math.radians(lon2 - lon1)
        
        a = (math.sin(delta_lat / 2) ** 2 +
             math.cos(lat1_rad) * math.cos(lat2_rad) *
             math.sin(delta_lon / 2) ** 2)
        
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        
        distance = R * c
        
        return distance
    
    def estimate_coordinates(
        self, 
        camera_lat: float, 
        camera_lon: float, 
        compass_heading: float, 
        distance: float
    ) -> Tuple[float, float]:
        """
        Estimate GPS coordinates of detected object
        
        Args:
            camera_lat: Camera latitude
            camera_lon: Camera longitude
            compass_heading: Compass heading in degrees (0-360)
            distance: Distance to object in meters
        
        Returns:
            Tuple of (latitude, longitude)
        """
        R = 6371000  # Earth radius in meters
        
        bearing = math.radians(compass_heading)
        lat1 = math.radians(camera_lat)
        lon1 = math.radians(camera_lon)
        
        lat2 = math.asin(
            math.sin(lat1) * math.cos(distance / R) +
            math.cos(lat1) * math.sin(distance / R) * math.cos(bearing)
        )
        
        lon2 = lon1 + math.atan2(
            math.sin(bearing) * math.sin(distance / R) * math.cos(lat1),
            math.cos(distance / R) - math.sin(lat1) * math.sin(lat2)
        )
        
        return math.degrees(lat2), math.degrees(lon2)


# Example usage
if __name__ == "__main__":
    calculator = DistanceCalculator()
    
    # Test distance calculation
    test_detection = {
        'bbox': {
            'width': 150,  # pixels
            'height': 50
        },
        'confidence': 0.85
    }
    
    distance = calculator.calculate_distance(test_detection, 'aircraft', image_width=1280)
    print(f"Estimated distance: {distance:.2f} meters")
    
    # Test GPS distance
    gps_dist = calculator.calculate_gps_distance(37.7749, -122.4194, 37.8044, -122.2712)
    print(f"GPS distance: {gps_dist:.2f} meters")
    
    # Test coordinate estimation
    lat, lon = calculator.estimate_coordinates(37.7749, -122.4194, 45, 1000)
    print(f"Estimated coordinates: {lat:.6f}, {lon:.6f}")
