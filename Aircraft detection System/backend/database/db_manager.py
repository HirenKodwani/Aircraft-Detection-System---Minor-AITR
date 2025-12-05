"""
Database Manager - SQLAlchemy ORM for database operations
"""
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime, Text, DECIMAL
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session
from datetime import datetime
from typing import List, Optional, Dict
import json

from backend.config.config import Config
from backend.utils.logger import setup_logger

logger = setup_logger('db_manager')

Base = declarative_base()


# ORM Models
class Device(Base):
    __tablename__ = 'devices'
    
    device_id = Column(Integer, primary_key=True)
    device_name = Column(String(100))
    device_type = Column(String(50))
    camera_model = Column(String(100))
    camera_resolution = Column(String(30))
    has_gps = Column(Boolean, default=False)
    has_compass = Column(Boolean, default=False)
    installation_location = Column(String(200))
    installation_latitude = Column(DECIMAL(9, 6))
    installation_longitude = Column(DECIMAL(9, 6))
    status = Column(String(20), default='active')
    created_at = Column(DateTime, default=datetime.now)


class DetectionSession(Base):
    __tablename__ = 'detection_sessions'
    
    session_id = Column(Integer, primary_key=True)
    device_id = Column(Integer)
    start_time = Column(DateTime)
    end_time = Column(DateTime)
    location_latitude = Column(DECIMAL(9, 6))
    location_longitude = Column(DECIMAL(9, 6))
    session_status = Column(String(20), default='active')
    total_detections = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.now)


class GeolocationData(Base):
    __tablename__ = 'geolocation_data'
    
    geo_id = Column(Integer, primary_key=True)
    detection_id = Column(Integer)
    device_id = Column(Integer)
    device_latitude = Column(DECIMAL(9, 6))
    device_longitude = Column(DECIMAL(9, 6))
    device_altitude = Column(DECIMAL(8, 2))
    compass_heading = Column(DECIMAL(5, 2))
    compass_azimuth = Column(DECIMAL(5, 2))
    estimated_distance_meters = Column(DECIMAL(10, 2))
    estimated_altitude_meters = Column(DECIMAL(8, 2))
    estimated_knots = Column(DECIMAL(6, 2))
    gps_accuracy = Column(DECIMAL(5, 2))
    created_at = Column(DateTime, default=datetime.now)


class DetectedObject(Base):
    __tablename__ = 'detected_objects'
    
    detection_id = Column(Integer, primary_key=True)
    session_id = Column(Integer)
    detection_timestamp = Column(DateTime)
    object_type = Column(String(50))
    confidence_score = Column(DECIMAL(4, 3))
    bounding_box_x = Column(Integer)
    bounding_box_y = Column(Integer)
    bounding_box_width = Column(Integer)
    bounding_box_height = Column(Integer)
    frame_number = Column(Integer)
    image_path = Column(String(255))
    detection_status = Column(String(30), default='pending')
    is_verified = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.now)


class DatabaseManager:
    """Manages database connections and operations"""
    
    def __init__(self, db_url: str = None):
        """
        Initialize database manager
        
        Args:
            db_url: Database URL (default from config)
        """
        self.db_url = db_url or Config.DATABASE_URL
        self.engine = None
        self.Session = None
        
        logger.info(f"Database manager initialized")
    
    def connect(self) -> bool:
        """
        Connect to database
        
        Returns:
            True if successful
        """
        try:
            self.engine = create_engine(self.db_url, pool_pre_ping=True)
            self.Session = scoped_session(sessionmaker(bind=self.engine))
            
            # Test connection
            with self.engine.connect() as conn:
                logger.info("Database connection established")
            
            return True
            
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            return False
    
    def create_tables(self):
        """Create all tables"""
        try:
            Base.metadata.create_all(self.engine)
            logger.info("Database tables created")
        except Exception as e:
            logger.error(f"Error creating tables: {e}")
    
    # Device operations
    def add_device(self, device_data: Dict) -> Optional[int]:
        """Add a new device"""
        try:
            session = self.Session()
            device = Device(**device_data)
            session.add(device)
            session.commit()
            device_id = device.device_id
            session.close()
            logger.info(f"Device added: {device_id}")
            return device_id
        except Exception as e:
            logger.error(f"Error adding device: {e}")
            session.rollback()
            session.close()
            return None
    
    def get_device(self, device_id: int) -> Optional[Dict]:
        """Get device by ID"""
        try:
            session = self.Session()
            device = session.query(Device).filter_by(device_id=device_id).first()
            session.close()
            
            if device:
                return {
                    'device_id': device.device_id,
                    'device_name': device.device_name,
                    'device_type': device.device_type,
                    'status': device.status
                }
            return None
        except Exception as e:
            logger.error(f"Error getting device: {e}")
            return None
    
    # Session operations
    def create_session(self, session_data: Dict) -> Optional[int]:
        """Create detection session"""
        try:
            session = self.Session()
            det_session = DetectionSession(**session_data)
            session.add(det_session)
            session.commit()
            session_id = det_session.session_id
            session.close()
            logger.info(f"Session created: {session_id}")
            return session_id
        except Exception as e:
            logger.error(f"Error creating session: {e}")
            session.rollback()
            session.close()
            return None
    
    def update_session(self, session_id: int, updates: Dict) -> bool:
        """Update detection session"""
        try:
            session = self.Session()
            session.query(DetectionSession).filter_by(session_id=session_id).update(updates)
            session.commit()
            session.close()
            return True
        except Exception as e:
            logger.error(f"Error updating session: {e}")
            session.rollback()
            session.close()
            return False
    
    # Detection operations
    def add_detection(self, detection_data: Dict) -> Optional[int]:
        """Add detected object"""
        try:
            session = self.Session()
            detection = DetectedObject(**detection_data)
            session.add(detection)
            session.commit()
            detection_id = detection.detection_id
            session.close()
            logger.debug(f"Detection added: {detection_id}")
            return detection_id
        except Exception as e:
            logger.error(f"Error adding detection: {e}")
            session.rollback()
            session.close()
            return None
    
    def get_detections(self, session_id: int = None, limit: int = 100) -> List[Dict]:
        """Get detections"""
        try:
            session = self.Session()
            query = session.query(DetectedObject)
            
            if session_id:
                query = query.filter_by(session_id=session_id)
            
            detections = query.order_by(DetectedObject.detection_timestamp.desc()).limit(limit).all()
            session.close()
            
            return [{
                'detection_id': d.detection_id,
                'session_id': d.session_id,
                'timestamp': d.detection_timestamp.isoformat() if d.detection_timestamp else None,
                'object_type': d.object_type,
                'confidence': float(d.confidence_score) if d.confidence_score else 0,
                'bbox': {
                    'x': d.bounding_box_x,
                    'y': d.bounding_box_y,
                    'width': d.bounding_box_width,
                    'height': d.bounding_box_height
                },
                'image_path': d.image_path
            } for d in detections]
            
        except Exception as e:
            logger.error(f"Error getting detections: {e}")
            return []
    
    # Geolocation operations
    def add_geolocation(self, geo_data: Dict) -> Optional[int]:
        """Add geolocation data"""
        try:
            session = self.Session()
            geo = GeolocationData(**geo_data)
            session.add(geo)
            session.commit()
            geo_id = geo.geo_id
            session.close()
            return geo_id
        except Exception as e:
            logger.error(f"Error adding geolocation: {e}")
            session.rollback()
            session.close()
            return None
    
    def close(self):
        """Close database connection"""
        if self.Session:
            self.Session.remove()
        if self.engine:
            self.engine.dispose()
        logger.info("Database connection closed")


# Singleton instance
_db_manager = None

def get_db_manager() -> DatabaseManager:
    """Get database manager singleton"""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
        _db_manager.connect()
    return _db_manager
