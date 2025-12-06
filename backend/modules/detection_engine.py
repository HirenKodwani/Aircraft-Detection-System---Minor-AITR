"""
Detection Engine - YOLOv8-based aircraft detection
"""
import cv2
import numpy as np
from ultralytics import YOLO
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from pathlib import Path

from backend.utils.logger import setup_logger
from backend.config.config import Config

logger = setup_logger('detection_engine')


class DetectionEngine:
    """YOLOv8-based object detection for aircraft"""
    
    def __init__(self, model_path: str = None, conf_threshold: float = 0.25):
        """
        Initialize detection engine
        
        Args:
            model_path: Path to YOLO model (default: yolov8n.pt)
            conf_threshold: Confidence threshold for detections
        """
        self.conf_threshold = conf_threshold
        self.model = None
        self.model_path = model_path or 'yolov8n.pt'
        
        # Aircraft-related class IDs from COCO dataset
        # 4: airplane, 32: bird (to filter out)
        self.aircraft_classes = [4]  # COCO: airplane
        self.bird_class = 32
        
        self.detection_count = 0
        
        logger.info(f"Detection engine initialized with threshold {conf_threshold}")
    
    def load_model(self) -> bool:
        """
        Load YOLO model
        
        Returns:
            True if successful
        """
        try:
            logger.info(f"Loading YOLO model from {self.model_path}")
            self.model = YOLO(self.model_path)
            logger.info("YOLO model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading YOLO model: {e}")
            logger.warning("Model will be downloaded on first use")
            return False
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess frame for detection
        
        Args:
            frame: Input frame
        
        Returns:
            Preprocessed frame
        """
        # YOLO handles preprocessing internally, but we can add custom preprocessing
        # For sky detection bias, we could enhance contrast in sky regions
        return frame
    
    def detect_sky_region(self, frame: np.ndarray) -> np.ndarray:
        """
        Detect sky region in frame (helps focus detection)
        
        Args:
            frame: Input frame
        
        Returns:
            Binary mask of sky region
        """
        # Convert to HSV for better sky detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Sky color ranges (blue/gray)
        # Blue sky
        lower_blue = np.array([90, 50, 50])
        upper_blue = np.array([130, 255, 255])
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
        
        # Gray sky (cloudy)
        lower_gray = np.array([0, 0, 100])
        upper_gray = np.array([180, 50, 220])
        mask_gray = cv2.inRange(hsv, lower_gray, upper_gray)
        
        # Combine masks
        sky_mask = cv2.bitwise_or(mask_blue, mask_gray)
        
        # Morphological operations to clean up
        kernel = np.ones((5, 5), np.uint8)
        sky_mask = cv2.morphologyEx(sky_mask, cv2.MORPH_CLOSE, kernel)
        sky_mask = cv2.morphologyEx(sky_mask, cv2.MORPH_OPEN, kernel)
        
        return sky_mask
    
    def detect_objects(self, frame: np.ndarray, focus_sky: bool = True) -> List[Dict]:
        """
        Detect aircraft in frame using YOLO
        
        Args:
            frame: Input frame
            focus_sky: Whether to focus detection on sky regions
        
        Returns:
            List of detection dictionaries
        """
        if self.model is None:
            if not self.load_model():
                logger.error("Cannot detect without model")
                return []
        
        try:
            # Get sky mask if focusing on sky
            sky_mask = None
            if focus_sky:
                sky_mask = self.detect_sky_region(frame)
            
            # Run YOLO detection
            results = self.model(frame, conf=self.conf_threshold, verbose=False)
            
            detections = []
            
            # Process results
            for result in results:
                boxes = result.boxes
                
                for box in boxes:
                    # Get class ID
                    cls_id = int(box.cls[0])
                    
                    # TEMPORARILY DISABLED: Filter for aircraft classes only
                    # This allows detecting people/cars/etc for testing
                    # if cls_id not in self.aircraft_classes:
                    #     continue
                    
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # Check if detection is in sky region
                    if focus_sky and sky_mask is not None:
                        # Check center of bounding box
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2
                        
                        if center_y < sky_mask.shape[0] and center_x < sky_mask.shape[1]:
                            if sky_mask[center_y, center_x] == 0:
                                # Not in sky region, skip
                                continue
                    
                    # Get confidence
                    confidence = float(box.conf[0])
                    
                    # Calculate bounding box dimensions
                    width = x2 - x1
                    height = y2 - y1
                    
                    detection = {
                        'bbox': {
                            'x1': x1,
                            'y1': y1,
                            'x2': x2,
                            'y2': y2,
                            'width': width,
                            'height': height
                        },
                        'confidence': confidence,
                        'class_id': cls_id,
                        'class_name': result.names[cls_id],
                        'timestamp': datetime.now(),
                        'in_sky': True if focus_sky else None
                    }
                    
                    detections.append(detection)
                    self.detection_count += 1
            
            logger.debug(f"Detected {len(detections)} objects in frame")
            return detections
            
        except Exception as e:
            logger.error(f"Error during detection: {e}")
            return []
    
    def draw_detections(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        Draw detection bounding boxes on frame
        
        Args:
            frame: Input frame
            detections: List of detections
        
        Returns:
            Frame with drawn detections
        """
        output_frame = frame.copy()
        
        for detection in detections:
            bbox = detection['bbox']
            confidence = detection['confidence']
            class_name = detection['class_name']
            
            # Draw bounding box
            cv2.rectangle(
                output_frame,
                (bbox['x1'], bbox['y1']),
                (bbox['x2'], bbox['y2']),
                (0, 255, 0),  # Green
                2
            )
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            
            # Background for text
            cv2.rectangle(
                output_frame,
                (bbox['x1'], bbox['y1'] - label_size[1] - 10),
                (bbox['x1'] + label_size[0], bbox['y1']),
                (0, 255, 0),
                -1
            )
            
            # Draw text
            cv2.putText(
                output_frame,
                label,
                (bbox['x1'], bbox['y1'] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                2
            )
        
        return output_frame
    
    def get_detection_stats(self) -> Dict:
        """Get detection statistics"""
        return {
            'total_detections': self.detection_count,
            'model_loaded': self.model is not None,
            'confidence_threshold': self.conf_threshold
        }


# Example usage
if __name__ == "__main__":
    import time
    
    # Initialize detection engine
    engine = DetectionEngine(conf_threshold=0.3)
    engine.load_model()
    
    # Test with camera
    from backend.modules.camera_module import CameraModule
    
    camera = CameraModule()
    if camera.initialize():
        print("Camera initialized, starting detection...")
        camera.start_capture()
        
        # Process a few frames
        for i in range(10):
            frame_data = camera.get_frame()
            if frame_data:
                frame = frame_data['frame']
                
                # Detect objects
                detections = engine.detect_objects(frame)
                
                if detections:
                    print(f"Frame {i}: Found {len(detections)} aircraft")
                    for det in detections:
                        print(f"  - {det['class_name']}: {det['confidence']:.2f}")
                    
                    # Draw and save
                    output = engine.draw_detections(frame, detections)
                    cv2.imwrite(f'detection_{i}.jpg', output)
                
                time.sleep(1)
        
        camera.stop_capture()
        print(f"Detection stats: {engine.get_detection_stats()}")
