"""
Camera Module for Aircraft Detection System
Handles camera initialization, frame capture, and preprocessing
"""
import cv2
import numpy as np
from datetime import datetime
from typing import Optional, Tuple
import threading
import queue
from pathlib import Path

from backend.utils.logger import setup_logger
from backend.config.config import Config

logger = setup_logger('camera_module')


class CameraModule:
    """Manages camera access and frame capture"""
    
    def __init__(self, device_id: int = None, width: int = None, height: int = None, fps: int = None):
        """
        Initialize camera module
        
        Args:
            device_id: Camera device ID (default from config)
            width: Frame width (default from config)
            height: Frame height (default from config)
            fps: Frames per second (default from config)
        """
        self.device_id = device_id or Config.CAMERA_DEVICE_ID
        self.width = width or Config.CAMERA_WIDTH
        self.height = height or Config.CAMERA_HEIGHT
        self.fps = fps or Config.CAMERA_FPS
        
        self.camera = None
        self.is_running = False
        self.frame_count = 0
        self.frame_buffer = queue.Queue(maxsize=30)
        self.capture_thread = None
        
        logger.info(f"Camera module initialized: Device {self.device_id}, Resolution {self.width}x{self.height}, FPS {self.fps}")
    
    def initialize(self) -> bool:
        """
        Initialize and configure the camera
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.camera = cv2.VideoCapture(self.device_id)
            
            if not self.camera.isOpened():
                logger.error(f"Failed to open camera device {self.device_id}")
                return False
            
            # Set camera properties
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.camera.set(cv2.CAP_PROP_FPS, self.fps)
            
            # Verify settings
            actual_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = int(self.camera.get(cv2.CAP_PROP_FPS))
            
            logger.info(f"Camera configured: {actual_width}x{actual_height} @ {actual_fps} FPS")
            
            return True
            
        except Exception as e:
            logger.error(f"Error initializing camera: {e}")
            return False
    
    def start_capture(self):
        """Start continuous frame capture in background thread"""
        if self.is_running:
            logger.warning("Camera capture already running")
            return
        
        if not self.camera or not self.camera.isOpened():
            logger.error("Camera not initialized. Call initialize() first")
            return
        
        self.is_running = True
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        logger.info("Camera capture started")
    
    def _capture_loop(self):
        """Internal loop for continuous frame capture"""
        while self.is_running:
            ret, frame = self.camera.read()
            
            if not ret:
                logger.warning("Failed to read frame from camera")
                continue
            
            self.frame_count += 1
            timestamp = datetime.now()
            
            # Add frame to buffer (drop oldest if full)
            if self.frame_buffer.full():
                try:
                    self.frame_buffer.get_nowait()
                except queue.Empty:
                    pass
            
            self.frame_buffer.put({
                'frame': frame,
                'frame_number': self.frame_count,
                'timestamp': timestamp,
                'width': frame.shape[1],
                'height': frame.shape[0]
            })
    
    def get_frame(self, timeout: float = 1.0) -> Optional[dict]:
        """
        Get the latest frame from buffer
        
        Args:
            timeout: Maximum time to wait for frame (seconds)
        
        Returns:
            Dictionary with frame data or None if timeout
        """
        try:
            return self.frame_buffer.get(timeout=timeout)
        except queue.Empty:
            logger.warning("No frame available within timeout")
            return None
    
    def capture_single_frame(self) -> Optional[Tuple[np.ndarray, dict]]:
        """
        Capture a single frame directly (blocking)
        
        Returns:
            Tuple of (frame, metadata) or None if failed
        """
        if not self.camera or not self.camera.isOpened():
            logger.error("Camera not initialized")
            return None
        
        ret, frame = self.camera.read()
        
        if not ret:
            logger.error("Failed to capture frame")
            return None
        
        self.frame_count += 1
        metadata = {
            'frame_number': self.frame_count,
            'timestamp': datetime.now(),
            'width': frame.shape[1],
            'height': frame.shape[0]
        }
        
        return frame, metadata
    
    def save_frame(self, frame: np.ndarray, filepath: str) -> bool:
        """
        Save frame to disk
        
        Args:
            frame: Frame to save
            filepath: Path to save file
        
        Returns:
            True if successful
        """
        try:
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(filepath, frame)
            logger.debug(f"Frame saved to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error saving frame: {e}")
            return False
    
    def stop_capture(self):
        """Stop frame capture and release camera"""
        self.is_running = False
        
        if self.capture_thread:
            self.capture_thread.join(timeout=2.0)
        
        if self.camera:
            self.camera.release()
            logger.info("Camera released")
    
    def get_camera_info(self) -> dict:
        """Get camera device information"""
        if not self.camera or not self.camera.isOpened():
            return {}
        
        return {
            'device_id': self.device_id,
            'resolution': f"{int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))}",
            'fps': int(self.camera.get(cv2.CAP_PROP_FPS)),
            'backend': self.camera.getBackendName(),
            'frame_count': self.frame_count
        }
    
    def __del__(self):
        """Cleanup on deletion"""
        self.stop_capture()


# Example usage
if __name__ == "__main__":
    # Initialize camera
    camera = CameraModule()
    
    if camera.initialize():
        print("Camera info:", camera.get_camera_info())
        
        # Start continuous capture
        camera.start_capture()
        
        # Capture a few frames
        for i in range(5):
            frame_data = camera.get_frame()
            if frame_data:
                print(f"Frame {frame_data['frame_number']}: {frame_data['width']}x{frame_data['height']} at {frame_data['timestamp']}")
        
        # Stop capture
        camera.stop_capture()
