"""
Camera Detection Engine - Modular Integration
Uses your Camera module.py as the core detection engine
Provides easy integration with backend modules (database, API, threat assessment)
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

# Import your camera module
from importlib import import_module
camera_module = import_module('Camera module')


class CameraDetectionEngine:
    """
    Wrapper around your Camera module.py that adds backend integration
    Can be used standalone OR with database/API modules
    """
    
    def __init__(self, enable_database=True, enable_api=True):
        """
        Initialize detection engine
        
        Args:
            enable_database: Enable SQLite logging
            enable_api: Enable AviationStack verification
        """
        self.enable_database = enable_database
        self.enable_api = enable_api
        
        # Optional backend imports
        self.db = None
        self.verifier = None
        self.session_id = None
        
        if enable_database:
            from backend.database.db_manager import get_db_manager
            self.db = get_db_manager()
            self.db.create_tables()
            self.session_id = self.db.create_session({
                'device_id': 1,
                'start_time': __import__('datetime').datetime.now(),
                'session_status': 'active'
            })
        
        if enable_api:
            from backend.modules.aircraft_verifier import AircraftVerifier
            self.verifier = AircraftVerifier()
    
    def process_detection(self, detection_data):
        """
        Process a detection from your Camera module
        Adds database logging and API verification
        
        Args:
            detection_data: Dict with keys: bbox, conf, class, timestamp
        
        Returns:
            Enhanced detection with threat_level, verified, etc.
        """
        result = detection_data.copy()
        
        # API Verification (if enabled and GPS available)
        if self.verifier and 'gps_lat' in detection_data:
            aircraft_info = self.verifier.verify_aircraft(
                detection_data['gps_lat'],
                detection_data['gps_lon'],
                detection_data.get('gps_alt', 0)
            )
            
            if aircraft_info:
                result['verified'] = True
                result['flight_number'] = aircraft_info.get('flight_number')
                result['airline'] = aircraft_info.get('airline')
                result['threat_level'] = self.verifier.classify_threat_level(
                    aircraft_info, 
                    detection_data['class']
                )
            else:
                result['verified'] = False
                # Drone = high threat, unknown aircraft = medium
                if 'drone' in detection_data['class'] or 'uav' in detection_data['class']:
                    result['threat_level'] = 'high'
                else:
                    result['threat_level'] = 'medium'
        else:
            result['verified'] = False
            result['threat_level'] = 'medium'
        
        # Database logging (if enabled)
        if self.db and self.session_id:
            bbox = detection_data['bbox']
            self.db.add_detection({
                'session_id': self.session_id,
                'detection_timestamp': detection_data['timestamp'],
                'object_type': detection_data['class'],
                'confidence_score': detection_data['conf'],
                'bounding_box_x': int(bbox[0]),
                'bounding_box_y': int(bbox[1]),
                'bounding_box_width': int(bbox[2] - bbox[0]),
                'bounding_box_height': int(bbox[3] - bbox[1]),
                'frame_number': detection_data.get('frame_number', 0),
                'is_verified': result.get('verified', False),
                'detection_status': 'threat' if result['threat_level'] in ['high', 'critical'] else 'detected'
            })
        
        return result
    
    def cleanup(self):
        """End session and cleanup"""
        if self.db and self.session_id:
            self.db.update_session(self.session_id, {
                'end_time': __import__('datetime').datetime.now(),
                'session_status': 'completed'
            })


# Example: How to use with your Camera module
if __name__ == "__main__":
    # Initialize with backend features
    engine = CameraDetectionEngine(
        enable_database=True,  # Log to SQLite
        enable_api=True        # Verify with AviationStack
    )
    
    # Your Camera module detection happens here...
    # When you get a detection, process it:
    detection = {
        'bbox': [100, 100, 200, 200],  # x1, y1, x2, y2
        'conf': 0.85,
        'class': 'airplane',
        'timestamp': __import__('datetime').datetime.now(),
        'frame_number': 1,
        # Optional GPS data for API verification
        # 'gps_lat': 28.6139,
        # 'gps_lon': 77.2090,
        # 'gps_alt': 100
    }
    
    # Process with backend
    enhanced = engine.process_detection(detection)
    print(f"Threat Level: {enhanced['threat_level']}")
    print(f"Verified: {enhanced['verified']}")
    
    # Cleanup when done
    engine.cleanup()
