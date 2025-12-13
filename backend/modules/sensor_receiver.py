"""
Sensor Receiver Module
Receives and processes GPS + compass data from mobile devices via WebSocket
"""
from datetime import datetime
from typing import Dict, Optional
from backend.utils.logger import setup_logger

logger = setup_logger('sensor_receiver')


class SensorReceiver:
    """Handles incoming sensor data from mobile devices"""
    
    def __init__(self, db_connection=None):
        """
        Initialize sensor receiver
        
        Args:
            db_connection: Database connection for storing sensor data
        """
        self.db = db_connection
        self.active_devices = {}
        self.data_buffer = []
        logger.info("Sensor receiver initialized")
    
    def validate_sensor_data(self, data: dict) -> tuple[bool, Optional[str]]:
        """
        Validate incoming sensor data
        
        Args:
            data: Sensor data dictionary
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        required_fields = ['device_id', 'latitude', 'longitude', 'timestamp']
        
        # Check required fields
        for field in required_fields:
            if field not in data:
                return False, f"Missing required field: {field}"
        
        # Validate latitude (-90 to 90)
        try:
            lat = float(data['latitude'])
            if not -90 <= lat <= 90:
                return False, f"Invalid latitude: {lat}"
        except (ValueError, TypeError):
            return False, "Latitude must be a number"
        
        # Validate longitude (-180 to 180)
        try:
            lon = float(data['longitude'])
            if not -180 <= lon <= 180:
                return False, f"Invalid longitude: {lon}"
        except (ValueError, TypeError):
            return False, "Longitude must be a number"
        
        # Validate compass heading if present (0-360)
        if 'compass_heading' in data:
            try:
                heading = float(data['compass_heading'])
                if not 0 <= heading <= 360:
                    return False, f"Invalid compass heading: {heading}"
            except (ValueError, TypeError):
                return False, "Compass heading must be a number"
        
        # Validate GPS accuracy if present
        if 'accuracy' in data:
            try:
                accuracy = float(data['accuracy'])
                if accuracy < 0:
                    return False, "GPS accuracy cannot be negative"
            except (ValueError, TypeError):
                return False, "GPS accuracy must be a number"
        
        return True, None
    
    def process_sensor_data(self, data: dict) -> dict:
        """
        Process and normalize sensor data
        
        Args:
            data: Raw sensor data from mobile device
        
        Returns:
            Processed sensor data dictionary
        """
        # Validate data
        is_valid, error = self.validate_sensor_data(data)
        if not is_valid:
            logger.warning(f"Invalid sensor data: {error}")
            return {'status': 'error', 'message': error}
        
        # Extract and normalize data
        processed = {
            'device_id': data['device_id'],
            'device_latitude': float(data['latitude']),
            'device_longitude': float(data['longitude']),
            'device_altitude': float(data.get('altitude', 0)),
            'compass_heading': float(data.get('compass_heading', 0)),
            'compass_azimuth': float(data.get('compass_heading', 0)),  # Same as heading for now
            'gps_accuracy': float(data.get('accuracy', 0)),
            'timestamp': data.get('timestamp', datetime.now().isoformat()),
            'received_at': datetime.now()
        }
        
        # Update active devices tracking
        device_id = data['device_id']
        self.active_devices[device_id] = {
            'last_update': datetime.now(),
            'latitude': processed['device_latitude'],
            'longitude': processed['device_longitude']
        }
        
        logger.debug(f"Processed sensor data from device {device_id}: " 
                    f"Lat={processed['device_latitude']:.6f}, "
                    f"Lon={processed['device_longitude']:.6f}, "
                    f"Heading={processed['compass_heading']:.2f}°")
        
        return {'status': 'success', 'data': processed}
    
    def store_sensor_data(self, processed_data: dict) -> bool:
        """
        Store processed sensor data in database
        
        Args:
            processed_data: Processed sensor data
        
        Returns:
            True if successful
        """
        if not self.db:
            logger.warning("No database connection available")
            self.data_buffer.append(processed_data)
            return False
        
        try:
            # This will be implemented when database connection is set up
            # For now, just log
            logger.info(f"Would store sensor data: {processed_data}")
            return True
        except Exception as e:
            logger.error(f"Error storing sensor data: {e}")
            return False
    
    def get_active_devices(self) -> dict:
        """Get list of currently active devices"""
        return self.active_devices
    
    def handle_device_disconnect(self, device_id: str):
        """Handle device disconnection"""
        if device_id in self.active_devices:
            del self.active_devices[device_id]
            logger.info(f"Device {device_id} disconnected")
    
    def get_latest_position(self, device_id: str) -> Optional[dict]:
        """
        Get latest position for a specific device
        
        Args:
            device_id: Device identifier
        
        Returns:
            Latest position data or None
        """
        return self.active_devices.get(device_id)


# Example usage
if __name__ == "__main__":
    receiver = SensorReceiver()
    
    # Test data
    test_data = {
        'device_id': 'mobile_001',
        'latitude': 37.7749,
        'longitude': -122.4194,
        'altitude': 15.5,
        'compass_heading': 180.5,
        'accuracy': 10.2,
        'timestamp': datetime.now().isoformat()
    }
    
    # Process data
    result = receiver.process_sensor_data(test_data)
    print("Processing result:", result)
    
    # Get active devices
    print("Active devices:", receiver.get_active_devices())
