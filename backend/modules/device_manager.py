"""
Device Manager - Centralized device initialization and session management
"""
from datetime import datetime
from typing import Optional, Dict
import uuid

from backend.utils.logger import setup_logger
from backend.modules.camera_module import CameraModule
from backend.config.config import Config

logger = setup_logger('device_manager')


class DeviceManager:
    """Manages all devices and detection sessions"""
    
    def __init__(self, db_connection=None):
        """
        Initialize device manager
        
        Args:
            db_connection: Database connection for storing device info
        """
        self.db = db_connection
        self.devices = {}
        self.active_sessions = {}
        self.camera = None
        logger.info("Device manager initialized")
    
    def register_camera_device(self, device_id: int = None, name: str = "Main Camera") -> Optional[str]:
        """
        Register a camera device
        
        Args:
            device_id: Camera device ID
            name: Device name
        
        Returns:
            Device UUID if successful, None otherwise
        """
        try:
            # Generate unique device UUID
            device_uuid = str(uuid.uuid4())
            
            # Initialize camera
            camera = CameraModule(device_id=device_id)
            if not camera.initialize():
                logger.error("Failed to initialize camera")
                return None
            
            # Get camera info
            info = camera.get_camera_info()
            
            # Store device info
            device_data = {
                'uuid': device_uuid,
                'name': name,
                'type': 'camera',
                'camera_model': info.get('backend', 'Unknown'),
                'camera_resolution': info.get('resolution', 'Unknown'),
                'has_gps': False,
                'has_compass': False,
                'status': 'active',
                'created_at': datetime.now(),
                'camera_instance': camera
            }
            
            self.devices[device_uuid] = device_data
            self.camera = camera
            
            logger.info(f"Camera device registered: {device_uuid} ({name})")
            logger.info(f"Resolution: {info.get('resolution')}, FPS: {info.get('fps')}")
            
            # TODO: Store in database
            # self._store_device_in_db(device_data)
            
            return device_uuid
            
        except Exception as e:
            logger.error(f"Error registering camera device: {e}")
            return None
    
    def register_mobile_device(self, device_id: str, device_name: str = "Mobile Sensor") -> bool:
        """
        Register a mobile sensor device
        
        Args:
            device_id: Mobile device ID
            device_name: Device name
        
        Returns:
            True if successful
        """
        try:
            device_data = {
                'uuid': device_id,
                'name': device_name,
                'type': 'mobile_sensor',
                'has_gps': True,
                'has_compass': True,
                'status': 'active',
                'created_at': datetime.now()
            }
            
            self.devices[device_id] = device_data
            logger.info(f"Mobile device registered: {device_id} ({device_name})")
            
            # TODO: Store in database
            # self._store_device_in_db(device_data)
            
            return True
            
        except Exception as e:
            logger.error(f"Error registering mobile device: {e}")
            return False
    
    def create_detection_session(self, device_uuid: str, location: Dict = None) -> Optional[str]:
        """
        Create a new detection session
        
        Args:
            device_uuid: Device UUID
            location: Initial location data (latitude, longitude)
        
        Returns:
            Session ID if successful, None otherwise
        """
        try:
            if device_uuid not in self.devices:
                logger.error(f"Device {device_uuid} not registered")
                return None
            
            session_id = str(uuid.uuid4())
            
            session_data = {
                'session_id': session_id,
                'device_uuid': device_uuid,
                'start_time': datetime.now(),
                'end_time': None,
                'location_latitude': location.get('latitude') if location else None,
                'location_longitude': location.get('longitude') if location else None,
                'session_status': 'active',
                'total_detections': 0
            }
            
            self.active_sessions[session_id] = session_data
            
            logger.info(f"Detection session created: {session_id} for device {device_uuid}")
            
            # TODO: Store in database
            # self._store_session_in_db(session_data)
            
            return session_id
            
        except Exception as e:
            logger.error(f"Error creating detection session: {e}")
            return None
    
    def end_detection_session(self, session_id: str) -> bool:
        """
        End a detection session
        
        Args:
            session_id: Session ID
        
        Returns:
            True if successful
        """
        try:
            if session_id not in self.active_sessions:
                logger.warning(f"Session {session_id} not found")
                return False
            
            session = self.active_sessions[session_id]
            session['end_time'] = datetime.now()
            session['session_status'] = 'completed'
            
            logger.info(f"Detection session ended: {session_id}")
            logger.info(f"Total detections: {session['total_detections']}")
            
            # TODO: Update in database
            # self._update_session_in_db(session)
            
            return True
            
        except Exception as e:
            logger.error(f"Error ending detection session: {e}")
            return False
    
    def start_camera_capture(self) -> bool:
        """Start camera capture"""
        if not self.camera:
            logger.error("No camera device registered")
            return False
        
        try:
            self.camera.start_capture()
            logger.info("Camera capture started")
            return True
        except Exception as e:
            logger.error(f"Error starting camera capture: {e}")
            return False
    
    def stop_camera_capture(self) -> bool:
        """Stop camera capture"""
        if not self.camera:
            logger.warning("No camera device registered")
            return False
        
        try:
            self.camera.stop_capture()
            logger.info("Camera capture stopped")
            return True
        except Exception as e:
            logger.error(f"Error stopping camera capture: {e}")
            return False
    
    def get_device_info(self, device_uuid: str) -> Optional[Dict]:
        """Get device information"""
        device = self.devices.get(device_uuid)
        if not device:
            return None
        
        # Remove camera instance from response
        info = {k: v for k, v in device.items() if k != 'camera_instance'}
        return info
    
    def get_all_devices(self) -> Dict:
        """Get all registered devices"""
        return {
            uuid: {k: v for k, v in device.items() if k != 'camera_instance'}
            for uuid, device in self.devices.items()
        }
    
    def get_active_sessions(self) -> Dict:
        """Get all active sessions"""
        return {
            sid: session
            for sid, session in self.active_sessions.items()
            if session['session_status'] == 'active'
        }
    
    def cleanup(self):
        """Cleanup all devices and sessions"""
        logger.info("Cleaning up device manager")
        
        # Stop camera
        if self.camera:
            self.camera.stop_capture()
        
        # End all active sessions
        for session_id in list(self.active_sessions.keys()):
            self.end_detection_session(session_id)
        
        logger.info("Device manager cleanup complete")


# Example usage
if __name__ == "__main__":
    # Initialize device manager
    manager = DeviceManager()
    
    # Register camera device
    camera_uuid = manager.register_camera_device(name="Main Detection Camera")
    if camera_uuid:
        print(f"Camera registered: {camera_uuid}")
        print("Camera info:", manager.get_device_info(camera_uuid))
        
        # Create detection session
        session_id = manager.create_detection_session(
            camera_uuid,
            location={'latitude': 37.7749, 'longitude': -122.4194}
        )
        print(f"Session created: {session_id}")
        
        # Start camera capture
        manager.start_camera_capture()
        
        # Simulate some work
        import time
        time.sleep(2)
        
        # Stop camera and end session
        manager.stop_camera_capture()
        manager.end_detection_session(session_id)
    
    # Register mobile device
    manager.register_mobile_device('mobile_001', 'iPhone 14')
    
    # Print all devices
    print("All devices:", manager.get_all_devices())
    
    # Cleanup
    manager.cleanup()
