"""
Threat Alert System - Vocal alarms and threat notifications
"""
import os
from typing import Dict
from datetime import datetime
from backend.config.config import Config
from backend.utils.logger import setup_logger

logger = setup_logger('threat_alert')


class ThreatAlertSystem:
    """Manage threat alerts and vocal alarms"""
    
    def __init__(self, enable_vocal=True):
        """
        Initialize threat alert system
        
        Args:
            enable_vocal: Enable vocal alarms (from config if not specified)
        """
        self.enable_vocal = enable_vocal and Config.VOCAL_ALARM_ENABLED
        self.volume = Config.ALARM_VOLUME
        
        # Track active threats
        self.active_threats = {}
        self.threat_history = []
        
        logger.info(f"Threat alert system initialized (vocal={'enabled' if self.enable_vocal else 'disabled'})")
    
    def process_detection(
        self, 
        detection_id: int,
        threat_level: str,
        aircraft_info: Dict = None,
        classification: str = 'unknown'
    ) -> Dict:
        """
        Process detection and trigger appropriate alerts
        
        Args:
            detection_id: Detection ID
            threat_level: Threat level (none/low/medium/high/critical)
            aircraft_info: Verified aircraft info
            classification: Detection classification
        
        Returns:
            Alert information dict
        """
        alert = {
            'detection_id': detection_id,
            'threat_level': threat_level,
            'timestamp': datetime.now().isoformat(),
            'classification': classification,
            'vocal_alarm': False,
            'alert_message': ''
        }
        
        # Determine alert action
        if threat_level == 'critical':
            # Military aircraft - CRITICAL ALARM
            alert['vocal_alarm'] = True
            alert['alert_message'] = f"CRITICAL ALERT: Military aircraft detected!"
            alert['alarm_sound'] = 'military_alert.mp3'
            
            if self.enable_vocal:
                self._trigger_vocal_alarm('critical', alert['alert_message'])
            
            logger.warning(f"CRITICAL THREAT: Military aircraft detected (ID: {detection_id})")
        
        elif threat_level == 'high':
            # UAV/Drone - HIGH THREAT
            alert['vocal_alarm'] = True
            alert['alert_message'] = f"THREAT DETECTED: Unidentified UAV/drone!"
            alert['alarm_sound'] = 'uav_alert.mp3'
            
            if self.enable_vocal:
                self._trigger_vocal_alarm('high', alert['alert_message'])
            
            logger.warning(f"HIGH THREAT: UAV detected (ID: {detection_id})")
        
        elif threat_level == 'medium':
            # Unverified aircraft
            alert['alert_message'] = f"Unverified aircraft detected"
            logger.info(f"MEDIUM THREAT: Unverified aircraft (ID: {detection_id})")
        
        elif threat_level == 'none':
            # Commercial - no alert
            if aircraft_info:
                alert['alert_message'] = f"Commercial flight: {aircraft_info.get('flight_number', 'Unknown')}"
            logger.debug(f"No threat: Commercial aircraft (ID: {detection_id})")
        
        # Track threat
        if threat_level in ['high', 'critical']:
            self.active_threats[detection_id] = alert
        
        # Add to history
        self.threat_history.append(alert)
        
        return alert
    
    def _trigger_vocal_alarm(self, level: str, message: str):
        """
        Trigger vocal alarm (text-to-speech or audio file)
        
        Args:
            level: Threat level
            message: Alarm message
        """
        try:
            # For browser-based deployment, we'll send WebSocket event
            # The frontend will handle the actual audio playback
            logger.info(f"Vocal alarm triggered: [{level.upper()}] {message}")
            
            # Optional: Use pyttsx3 for local text-to-speech
            # try:
            #     import pyttsx3
            #     engine = pyttsx3.init()
            #     engine.setProperty('volume', self.volume)
            #     engine.say(message)
            #     engine.runAndWait()
            # except ImportError:
            #     logger.warning("pyttsx3 not installed, skipping TTS")
            
        except Exception as e:
            logger.error(f"Error triggering vocal alarm: {e}")
    
    def get_active_threats(self) -> Dict:
        """Get currently active threats"""
        return {
            'count': len(self.active_threats),
            'threats': list(self.active_threats.values())
        }
    
    def clear_threat(self, detection_id: int):
        """Clear a threat from active list"""
        if detection_id in self.active_threats:
            del self.active_threats[detection_id]
            logger.info(f"Threat cleared: {detection_id}")
    
    def get_threat_statistics(self) -> Dict:
        """Get threat statistics"""
        stats = {
            'total_alerts': len(self.threat_history),
            'active_threats': len(self.active_threats),
            'by_level': {}
        }
        
        for alert in self.threat_history:
            level = alert['threat_level']
            stats['by_level'][level] = stats['by_level'].get(level, 0) + 1
        
        return stats
    
    def create_alert_for_browser(self, alert: Dict) -> Dict:
        """
        Format alert for browser consumption via WebSocket
        
        Args:
            alert: Alert information
        
        Returns:
            Browser-formatted alert
        """
        return {
            'type': 'threat_alert',
            'threat_level': alert['threat_level'],
            'message': alert['alert_message'],
            'vocal_alarm': alert['vocal_alarm'],
            'alarm_sound': alert.get('alarm_sound'),
            'timestamp': alert['timestamp'],
            'detection_id': alert['detection_id']
        }


# Example usage
if __name__ == "__main__":
    alert_system = ThreatAlertSystem(enable_vocal=True)
    
    # Test military aircraft alert
    print("\n1. Testing military aircraft alert:")
    military_alert = alert_system.process_detection(
        detection_id=1,
        threat_level='critical',
        classification='aircraft'
    )
    print(f"Alert: {military_alert}")
    
    # Test UAV alert
    print("\n2. Testing UAV alert:")
    uav_alert = alert_system.process_detection(
        detection_id=2,
        threat_level='high',
        classification='drone'
    )
    print(f"Alert: {uav_alert}")
    
    # Test commercial aircraft (no threat)
    print("\n3. Testing commercial aircraft:")
    commercial_alert = alert_system.process_detection(
        detection_id=3,
        threat_level='none',
        aircraft_info={'flight_number': 'AA123'},
        classification='aircraft'
    )
    print(f"Alert: {commercial_alert}")
    
    # Get stats
    print("\n4. Threat statistics:")
    stats = alert_system.get_threat_statistics()
    print(f"Stats: {stats}")
    
    print("\n5. Active threats:")
    active = alert_system.get_active_threats()
    print(f"Active: {active}")
