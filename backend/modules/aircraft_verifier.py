"""
Aircraft Verification - Verify detected aircraft using AviationStack API
"""
import requests
from typing import Dict, Optional
from datetime import datetime, timedelta
from backend.config.config import Config
from backend.utils.logger import setup_logger

logger = setup_logger('aircraft_verifier')


class AircraftVerifier:
    """Verify aircraft using AviationStack API"""
    
    def __init__(self, api_key: str = None):
        """
        Initialize aircraft verifier
        
        Args:
            api_key: AviationStack API key (from config if not provided)
        """
        self.api_key = api_key or Config.AVIATIONSTACK_API_KEY
        self.base_url = "http://api.aviationstack.com/v1"
        
        # Cache to avoid repeated API calls
        self.cache = {}
        self.cache_duration = timedelta(minutes=5)
        
        logger.info("Aircraft verifier initialized")
    
    def verify_aircraft(
        self, 
        latitude: float, 
        longitude: float, 
        altitude: float = None
    ) -> Optional[Dict]:
        """
        Verify if there's a known aircraft at given coordinates
        
        Args:
            latitude: Aircraft latitude
            longitude: Aircraft longitude
            altitude: Aircraft altitude in meters (optional)
        
        Returns:
            Aircraft information dict or None
        """
        if not self.api_key:
            logger.warning("AviationStack API key not configured")
            return None
        
        try:
            # Check cache first
            cache_key = f"{latitude:.4f},{longitude:.4f}"
            if cache_key in self.cache:
                cached_data, cache_time = self.cache[cache_key]
                if datetime.now() - cache_time < self.cache_duration:
                    logger.debug(f"Using cached data for {cache_key}")
                    return cached_data
            
            # Call AviationStack API - flights endpoint
            params = {
                'access_key': self.api_key,
                'limit': 10
            }
            
            response = requests.get(
                f"{self.base_url}/flights",
                params=params,
                timeout=5
            )
            
            if response.status_code == 200:
                data = response.json()
                
                if 'data' in data and len(data['data']) > 0:
                    # Find closest aircraft
                    closest = self._find_closest_aircraft(
                        data['data'],
                        latitude,
                        longitude,
                        altitude
                    )
                    
                    if closest:
                        # Cache result
                        self.cache[cache_key] = (closest, datetime.now())
                        return closest
            else:
                logger.warning(f"AviationStack API error: {response.status_code}")
            
            return None
            
        except Exception as e:
            logger.error(f"Error verifying aircraft: {e}")
            return None
    
    def _find_closest_aircraft(
        self, 
        flights: list, 
        lat: float, 
        lon: float, 
        alt: float = None
    ) -> Optional[Dict]:
        """
        Find closest aircraft from flight list
        
        Args:
            flights: List of flight data from API
            lat: Target latitude
            lon: Target longitude
            alt: Target altitude
        
        Returns:
            Closest aircraft info or None
        """
        closest = None
        min_distance = float('inf')
        
        for flight in flights:
            if not flight.get('live'):
                continue
            
            live = flight['live']
            if not live.get('latitude') or not live.get('longitude'):
                continue
            
            # Calculate distance
            flight_lat = live['latitude']
            flight_lon = live['longitude']
            
            # Simple distance calculation (Pythagorean approximation)
            distance = ((lat - flight_lat) ** 2 + (lon - flight_lon) ** 2) ** 0.5
            
            # If altitude available, consider it
            if alt and live.get('altitude'):
                alt_diff = abs(alt - live['altitude']) / 1000  # Normalize
                distance += alt_diff
            
            # If within reasonable range (0.1 degrees ~ 11km)
            if distance < 0.1 and distance < min_distance:
                min_distance = distance
                closest = self._parse_flight_data(flight)
        
        return closest
    
    def _parse_flight_data(self, flight: Dict) -> Dict:
        """
        Parse flight data from AviationStack response
        
        Args:
            flight: Flight data from API
        
        Returns:
            Parsed aircraft information
        """
        aircraft_info = {
            'flight_number': flight.get('flight', {}).get('iata', 'Unknown'),
            'callsign': flight.get('flight', {}).get('icao', ''),
            'airline': flight.get('airline', {}).get('name', 'Unknown'),
            'aircraft_type': flight.get('aircraft', {}).get('iata', 'Unknown'),
            'registration': flight.get('aircraft', {}).get('registration', ''),
            'origin': flight.get('departure', {}).get('iata', ''),
            'destination': flight.get('arrival', {}).get('iata', ''),
            'altitude': flight.get('live', {}).get('altitude', 0),
            'speed': flight.get('live', {}).get('speed_horizontal', 0),
            'is_commercial': True,  # AviationStack only tracks commercial flights
            'is_military': False,
            'is_uav': False,
            'verified': True,
            'verification_source': 'aviationstack'
        }
        
        return aircraft_info
    
    def classify_threat_level(self, aircraft_info: Dict, detection_classification: str) -> str:
        """
        Classify threat level based on aircraft info and detection
        
        Args:
            aircraft_info: Verified aircraft info (None if unverified)
            detection_classification: Classification from detector (aircraft/drone)
        
        Returns:
            Threat level: 'none', 'low', 'medium', 'high', 'critical'
        """
        # If verified commercial aircraft
        if aircraft_info and aircraft_info.get('is_commercial'):
            return 'none'  # No threat
        
        # If detected as drone/UAV
        if detection_classification == 'drone' or (aircraft_info and aircraft_info.get('is_uav')):
            return 'high'  # All UAVs are threats
        
        # If military aircraft detected
        if aircraft_info and aircraft_info.get('is_military'):
            return 'critical'  # Military = critical threat
        
        # Unverified aircraft
        if not aircraft_info and detection_classification == 'aircraft':
            return 'medium'  # Unknown aircraft = medium threat
        
        return 'low'  # Default low threat


# Example usage
if __name__ == "__main__":
    verifier = AircraftVerifier()
    
    # Test verification
    result = verifier.verify_aircraft(37.7749, -122.4194, 10000)
    
    if result:
        print("Aircraft found:")
        print(f"  Flight: {result['flight_number']}")
        print(f"  Type: {result['aircraft_type']}")
        print(f"  Commercial: {result['is_commercial']}")
        
        threat = verifier.classify_threat_level(result, 'aircraft')
        print(f"  Threat Level: {threat}")
    else:
        print("No aircraft found at coordinates")
        
        # Test unverified threat classification
        threat = verifier.classify_threat_level(None, 'drone')
        print(f"Unverified drone threat level: {threat}")
