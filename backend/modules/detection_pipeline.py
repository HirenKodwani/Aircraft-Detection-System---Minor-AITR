"""
Detection Pipeline - Integrates all detection modules
"""
import cv2
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path

from backend.modules.detection_engine import DetectionEngine
from backend.modules.aircraft_classifier import AircraftClassifier
from backend.modules.distance_calculator import DistanceCalculator
from backend.modules.speed_detector import SpeedDetector
from backend.modules.aircraft_verifier import AircraftVerifier
from backend.modules.threat_alert import ThreatAlertSystem
from backend.database.db_manager import get_db_manager
from backend.utils.logger import setup_logger
from backend.config.config import Config

logger = setup_logger('detection_pipeline')


class DetectionPipeline:
    """
    Complete detection pipeline integrating all modules
    For browser integration via WebSocket/REST API
    """
    
    def __init__(self, session_id: int, save_frames: bool = False):
        """
        Initialize detection pipeline
        
        Args:
            session_id: Current detection session ID
            save_frames: Whether to save detection frames to disk
        """
        self.session_id = session_id
        self.save_frames = save_frames
        
        # Initialize all modules
        self.detection_engine = DetectionEngine(conf_threshold=0.3)
        self.classifier = AircraftClassifier()
        self.distance_calc = DistanceCalculator()
        self.speed_detector = SpeedDetector(fps=Config.CAMERA_FPS)
        self.verifier = AircraftVerifier() if Config.ENABLE_AIRCRAFT_VERIFICATION else None
        self.threat_alert = ThreatAlertSystem()
        
        # Database
        self.db = get_db_manager()
        
        # State
        self.frame_count = 0
        self.total_detections = 0
        self.threat_count = 0
        
        # Current GPS position (from mobile device)
        self.current_gps = None
        self.current_compass = None
        
        logger.info(f"Detection pipeline initialized for session {session_id}")
    
    def update_gps_position(self, latitude: float, longitude: float, altitude: float = 0):
        """
        Update current GPS position from mobile device
        
        Args:
            latitude: Current latitude
            longitude: Current longitude
            altitude: Current altitude in meters
        """
        self.current_gps = {
            'latitude': latitude,
            'longitude': longitude,
            'altitude': altitude
        }
        logger.debug(f"GPS updated: {latitude:.6f}, {longitude:.6f}")
    
    def update_compass(self, heading: float):
        """
        Update compass heading from mobile device
        
        Args:
            heading: Compass heading in degrees (0-360)
        """
        self.current_compass = heading
        logger.debug(f"Compass updated: {heading:.2f}°")
    
    def process_frame(self, frame: np.ndarray, frame_number: int = None) -> Dict:
        """
        Process a single frame through the detection pipeline
        
        Args:
            frame: Input frame from camera
            frame_number: Frame number (optional)
        
        Returns:
            Dictionary with detection results for browser display
        """
        self.frame_count += 1
        frame_num = frame_number or self.frame_count
        
        results = {
            'frame_number': frame_num,
            'timestamp': datetime.now().isoformat(),
            'detections': [],
            'total_count': 0
        }
        
        try:
            # Step 1: Detect objects using YOLO
            detections = self.detection_engine.detect_objects(frame, focus_sky=True)
            
            if not detections:
                return results
            
            # Step 2: Process each detection
            frame_shape = frame.shape
            processed_detections = []
            
            for detection in detections:
                # Classify object type
                classification = self.classifier.classify(detection, frame_shape)
                
                # Skip birds (not relevant for aircraft detection)
                if classification == 'bird':
                    continue
                
                # Calculate distance
                distance = self.distance_calc.calculate_distance(
                    detection, 
                    classification,
                    image_width=frame.shape[1]
                )
                
                # Track and calculate speed
                object_id, speed = self.speed_detector.track_object(
                    detection,
                    distance,
                    detection['timestamp']
                )
                
                # Prepare detection data for database and browser
                detection_data = {
                    'object_id': object_id,
                    'object_type': classification,
                    'confidence': detection['confidence'],
                    'bbox': detection['bbox'],
                    'distance_meters': distance,
                    'speed_mps': speed if speed else 0,
                    'speed_kmh': self.speed_detector.get_speed_kmh(speed) if speed else 0,
                    'timestamp': detection['timestamp'].isoformat()
                }
                
                # Add GPS coordinates if available
                est_lat, est_lon = None, None
                if self.current_gps and self.current_compass and distance > 0:
                    est_lat, est_lon = self.distance_calc.estimate_coordinates(
                        self.current_gps['latitude'],
                        self.current_gps['longitude'],
                        self.current_compass,
                        distance
                    )
                    detection_data['estimated_location'] = {
                        'latitude': est_lat,
                        'longitude': est_lon
                    }
                
                # Aircraft verification and threat detection
                aircraft_info = None
                if self.verifier and est_lat and est_lon:
                    aircraft_info = self.verifier.verify_aircraft(
                        est_lat, est_lon,
                        self.current_gps.get('altitude', 0) + distance  # Estimated altitude
                    )
                    detection_data['verified'] = aircraft_info is not None
                    if aircraft_info:
                        detection_data['flight_info'] = {
                            'flight_number': aircraft_info.get('flight_number'),
                            'aircraft_type': aircraft_info.get('aircraft_type'),
                            'is_commercial': aircraft_info.get('is_commercial')
                        }
                
                # Threat classification
                threat_level = self.verifier.classify_threat_level(
                    aircraft_info, classification
                ) if self.verifier else 'unknown'
                
                detection_data['threat_level'] = threat_level
                
                processed_detections.append(detection_data)
                
                # Save to database (will get detection_id)
                detection_id = self._save_detection_to_db(detection_data, frame_num)
                
                # Process threat alert
                if detection_id:
                    alert = self.threat_alert.process_detection(
                        detection_id,
                        threat_level,
                        aircraft_info,
                        classification
                    )
                    
                    detection_data['alert'] = alert
                    
                    if threat_level in ['high', 'critical']:
                        self.threat_count += 1
                        # Broadcast alert via WebSocket (will be handled by API)
                        detection_data['broadcast_alert'] = self.threat_alert.create_alert_for_browser(alert)
            
            # Update results
            results['detections'] = processed_detections
            results['total_count'] = len(processed_detections)
            self.total_detections += len(processed_detections)
            
            # Save frame with detections if enabled
            if self.save_frames and processed_detections:
                self._save_detection_frame(frame, detections, frame_num)
            
            # Update session statistics in database
            self.db.update_session(self.session_id, {
                'total_detections': self.total_detections
            })
            
            logger.info(f"Frame {frame_num}: {len(processed_detections)} detections")
            
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            results['error'] = str(e)
        
        return results
    
    def _save_detection_to_db(self, detection_data: Dict, frame_number: int) -> Optional[int]:
        """Save detection to database and return detection_id"""
        try:
            bbox = detection_data['bbox']
            
            # Update status based on threat level
            threat_level = detection_data.get('threat_level', 'unknown')
            if threat_level in ['high', 'critical']:
                status = 'threat_detected'
            elif detection_data.get('verified'):
                status = 'verified'
            else:
                status = 'detected'
            
            db_data = {
                'session_id': self.session_id,
                'detection_timestamp': datetime.fromisoformat(detection_data['timestamp']),
                'object_type': detection_data['object_type'],
                'confidence_score': detection_data['confidence'],
                'bounding_box_x': bbox['x1'],
                'bounding_box_y': bbox['y1'],
                'bounding_box_width': bbox['width'],
                'bounding_box_height': bbox['height'],
                'frame_number': frame_number,
                'detection_status': status,
                'is_verified': detection_data.get('verified', False)
            }
            
            detection_id = self.db.add_detection(db_data)
            
            # Save geolocation if available
            if self.current_gps and detection_id:
                geo_data = {
                    'detection_id': detection_id,
                    'device_id': 1,  # TODO: Get from device manager
                    'device_latitude': self.current_gps['latitude'],
                    'device_longitude': self.current_gps['longitude'],
                    'device_altitude': self.current_gps.get('altitude', 0),
                    'compass_heading': self.current_compass or 0,
                    'compass_azimuth': self.current_compass or 0,
                    'estimated_distance_meters': detection_data['distance_meters'],
                    'estimated_knots': detection_data.get('speed_kmh', 0) * 0.539957  # km/h to knots
                }
                self.db.add_geolocation(geo_data)
            
            return detection_id
            
        except Exception as e:
            logger.error(f"Error saving detection to database: {e}")
            return None
    
    def _save_detection_frame(self, frame: np.ndarray, detections: List[Dict], frame_number: int):
        """Save frame with detection overlays"""
        try:
            # Draw detections
            output_frame = self.detection_engine.draw_detections(frame, detections)
            
            # Create save directory
            save_dir = Path(Config.FRAME_STORAGE_PATH) / f"session_{self.session_id}"
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Save frame
            filename = f"detection_{frame_number}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            filepath = save_dir / filename
            
            cv2.imwrite(str(filepath), output_frame)
            logger.debug(f"Saved detection frame: {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving detection frame: {e}")
    
    def get_stats(self) -> Dict:
        """Get pipeline statistics for browser display"""
        return {
            'session_id': self.session_id,
            'frames_processed': self.frame_count,
            'total_detections': self.total_detections,
            'detection_engine': self.detection_engine.get_detection_stats(),
            'classifier': self.classifier.get_stats(),
            'speed_detector': self.speed_detector.get_stats(),
            'gps_available': self.current_gps is not None,
            'compass_available': self.current_compass is not None
        }
    
    def cleanup(self):
        """Cleanup resources"""
        logger.info(f"Pipeline cleanup: {self.total_detections} total detections")
        self.speed_detector.cleanup_old_tracks()


# Example usage
if __name__ == "__main__":
    from backend.modules.camera_module import CameraModule
    
    # Initialize camera
    camera = CameraModule()
    
    if camera.initialize():
        # Create detection pipeline (session_id would come from database)
        pipeline = DetectionPipeline(session_id=1, save_frames=True)
        
        # Simulate GPS position
        pipeline.update_gps_position(37.7749, -122.4194, 100)
        pipeline.update_compass(180.0)
        
        # Start camera
        camera.start_capture()
        
        # Process frames
        import time
        for i in range(30):  # Process 30 frames
            frame_data = camera.get_frame()
            if frame_data:
                results = pipeline.process_frame(frame_data['frame'], frame_data['frame_number'])
                print(f"Frame {results['frame_number']}: {results['total_count']} detections")
                
                for det in results['detections']:
                    print(f"  - {det['object_type']}: {det['confidence']:.2f}, "
                          f"distance={det['distance_meters']:.1f}m, "
                          f"speed={det['speed_kmh']:.1f} km/h")
            
            time.sleep(0.1)
        
        # Cleanup
        camera.stop_capture()
        pipeline.cleanup()
        
        print(f"\nFinal stats: {pipeline.get_stats()}")
