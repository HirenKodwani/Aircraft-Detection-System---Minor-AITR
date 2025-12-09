"""
Aircraft Classifier - Classify detected objects
"""
import numpy as np
from typing import Dict, Optional
from backend.utils.logger import setup_logger

logger = setup_logger('aircraft_classifier')


class AircraftClassifier:
    """Classify detected objects as aircraft, drone, bird, or unknown"""
    
    def __init__(self):
        """Initialize classifier"""
        self.classification_count = 0
        logger.info("Aircraft classifier initialized")
    
    def classify(self, detection: Dict, frame_shape: tuple, distance: Optional[float] = None) -> str:
        """
        Classify detected object
        
        Args:
            detection: Detection dictionary with bbox and confidence
            frame_shape: Shape of the frame (height, width, channels)
            distance: Estimated distance in meters (optional)
        
        Returns:
            Classification: 'aircraft', 'drone', 'bird', 'unknown'
        """
        try:
            bbox = detection['bbox']
            confidence = detection['confidence']
            
            # Calculate object size metrics
            width = bbox['width']
            height = bbox['height']
            area = width * height
            aspect_ratio = width / height if height > 0 else 1
            
            # Calculate relative size (percentage of frame)
            frame_height, frame_width = frame_shape[:2]
            relative_area = area / (frame_height * frame_width)
            
            # Classification logic
            classification = self._classify_by_features(
                width, height, area, aspect_ratio, relative_area, 
                confidence, distance
            )
            
            self.classification_count += 1
            
            logger.debug(f"Classified as {classification}: "
                        f"size={area}px, aspect={aspect_ratio:.2f}, "
                        f"relative_area={relative_area:.4f}")
            
            return classification
            
        except Exception as e:
            logger.error(f"Error classifying object: {e}")
            return 'unknown'
    
    def _classify_by_features(
        self, 
        width: int, 
        height: int, 
        area: int, 
        aspect_ratio: float, 
        relative_area: float, 
        confidence: float,
        distance: Optional[float]
    ) -> str:
        """
        Classify based on features
        
        Classification rules:
        - Drone: Small, hovering capability, irregular shape, close distance
        - Aircraft: Large, elongated (wings), far distance
        - Bird: Very small, irregular movement, low altitude
        """
        
        # Very small objects are likely birds
        if relative_area < 0.0005:  # < 0.05% of frame
            return 'bird'
        
        # Distance-based classification (if available)
        if distance is not None:
            if distance < 100:  # < 100m
                # Close objects with small size = drone
                if relative_area < 0.01:  # < 1% of frame
                    return 'drone'
            elif distance > 500:  # > 500m
                # Far objects = likely aircraft
                return 'aircraft'
        
        # Size-based classification
        if relative_area < 0.002:  # < 0.2% of frame
            # Small objects
            if aspect_ratio > 2.0:
                # Elongated = small aircraft or drone in flight
                return 'drone'
            else:
                return 'bird'
        
        elif relative_area < 0.02:  # 0.2% - 2% of frame
            # Medium objects
            if aspect_ratio > 2.5:
                # Very elongated = aircraft with visible wings
                return 'aircraft'
            else:
                # Could be drone or close aircraft
                return 'drone'
        
        else:  # > 2% of frame
            # Large objects
            if aspect_ratio > 2.0:
                # Large and elongated = aircraft
                return 'aircraft'
            else:
                # Large but not elongated = close drone or unknown
                return 'drone' if confidence > 0.5 else 'unknown'
    
    def get_classification_confidence(self, classification: str, detection: Dict) -> float:
        """
        Get confidence score for classification
        
        Args:
            classification: Classification result
            detection: Detection dictionary
        
        Returns:
            Confidence score (0-1)
        """
        base_confidence = detection['confidence']
        
        # Adjust confidence based on classification certainty
        if classification == 'bird':
            # Birds are easy to classify by size
            return min(base_confidence * 1.1, 1.0)
        elif classification == 'unknown':
            # Unknown classifications have lower confidence
            return base_confidence * 0.7
        else:
            # Aircraft and drones
            return base_confidence
    
    def get_stats(self) -> Dict:
        """Get classification statistics"""
        return {
            'total_classifications': self.classification_count
        }


# Example usage
if __name__ == "__main__":
    classifier = AircraftClassifier()
    
    # Test classification
    test_detection = {
        'bbox': {
            'x1': 100,
            'y1': 50,
            'x2': 300,
            'y2': 120,
            'width': 200,
            'height': 70
        },
        'confidence': 0.85
    }
    
    frame_shape = (720, 1280, 3)
    
    result = classifier.classify(test_detection, frame_shape, distance=1000)
    print(f"Classification: {result}")
    
    confidence = classifier.get_classification_confidence(result, test_detection)
    print(f"Confidence: {confidence:.2f}")
    
    print(f"Stats: {classifier.get_stats()}")
