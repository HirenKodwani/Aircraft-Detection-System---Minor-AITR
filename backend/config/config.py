"""
Configuration settings for Aircraft Detection System
"""
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Main configuration class"""
    
    # Database Configuration - SWITCHED TO SQLITE FOR WEB DEPLOYMENT
    DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///aircraft_detections.db')
    
    # Camera Configuration
    CAMERA_DEVICE_ID = int(os.getenv('CAMERA_DEVICE_ID', '0'))
    CAMERA_WIDTH = int(os.getenv('CAMERA_WIDTH', '1280'))
    CAMERA_HEIGHT = int(os.getenv('CAMERA_HEIGHT', '720'))
    CAMERA_FPS = int(os.getenv('CAMERA_FPS', '30'))
    
    # API Configuration
    API_HOST = os.getenv('API_HOST', '0.0.0.0')
    API_PORT = int(os.getenv('API_PORT', '5000'))
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
    
    # WebSocket Configuration
    WEBSOCKET_PING_INTERVAL = 25
    WEBSOCKET_PING_TIMEOUT = 60
    
    # Sensor Data Configuration
    SENSOR_UPDATE_INTERVAL_MS = int(os.getenv('SENSOR_UPDATE_INTERVAL_MS', '1000'))
    GPS_ACCURACY_THRESHOLD = float(os.getenv('GPS_ACCURACY_THRESHOLD', '50.0'))
    
    # Logging Configuration
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE = os.getenv('LOG_FILE', 'logs/aircraft_detection.log')
    
    # Frame Storage
    FRAME_STORAGE_PATH = os.getenv('FRAME_STORAGE_PATH', 'data/frames')
    SAVE_FRAMES = os.getenv('SAVE_FRAMES', 'False').lower() == 'true'
    
    # Detection Configuration - UPGRADED TO MEDIUM MODEL
    YOLO_MODEL_PATH = os.getenv('YOLO_MODEL_PATH', 'yolov8m.pt')  # 50% better accuracy
    DETECTION_CONFIDENCE = float(os.getenv('DETECTION_CONFIDENCE', '0.25'))  # Lower for better recall
    FOCUS_SKY_DETECTION = os.getenv('FOCUS_SKY_DETECTION', 'True').lower() == 'true'
    
    # Performance
    MAX_DETECTION_FPS = int(os.getenv('MAX_DETECTION_FPS', '10'))
    
    # External API Configuration
    AVIATIONSTACK_API_KEY = os.getenv('AVIATIONSTACK_API_KEY', '')
    ENABLE_AIRCRAFT_VERIFICATION = os.getenv('ENABLE_AIRCRAFT_VERIFICATION', 'True').lower() == 'true'
    
    # Threat Detection
    VOCAL_ALARM_ENABLED = os.getenv('VOCAL_ALARM_ENABLED', 'True').lower() == 'true'
    ALARM_VOLUME = float(os.getenv('ALARM_VOLUME', '0.8'))
