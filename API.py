#!/usr/bin/env python3
"""
Aircraft Tracker - ADSB.lol API
Fetches real-time aircraft data for ANY location with ANY coordinates
"""

import requests
import json
import pandas as pd
from datetime import datetime

class AircraftTracker:
    """Simple Aircraft Tracker for any coordinates"""
    
    def __init__(self):
        self.api_url = "https://api.adsb.lol/v2/point"
        self.session = requests.Session()
    
    def get_aircraft(self, latitude, longitude, radius=100):
        """
        Fetch aircraft data for any coordinates
        
        Args:
            latitude: Any latitude (-90 to 90)
            longitude: Any longitude (-180 to 180)
            radius: Search radius in nautical miles (1-250)
        
        Returns:
            Dictionary with aircraft data or None if error
        """
        
        # Build URL
        url = f"{self.api_url}/{latitude}/{longitude}/{radius}"
        
        try:
            print(f"\n🌍 Location: ({latitude}, {longitude})")
            print(f"📏 Radius: {radius} nautical miles")
            print(f"🔗 Fetching: {url}\n")
            
            response = self.session.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Status: SUCCESS")
                return data
            else:
                print(f"❌ Status Code: {response.status_code}")
                return None
        
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            return None
    
    def display_data(self, data):
        """Display aircraft data in clean format"""
        
        if not data or 'ac' not in data:
            print("❌ No data received")
            return
        
        aircraft_list = data.get('ac', [])
        total = len(aircraft_list)
        
        print(f"\n{'='*140}")
        print(f"FOUND {total} AIRCRAFT")
        print(f"{'='*140}\n")
        
        if total == 0:
            print("📭 No aircraft in this area. Try a larger radius or different location.\n")
            return
        
        # Display in table format
        print(f"{'NO':<5} {'CALLSIGN':<15} {'REGISTRATION':<15} {'TYPE':<12} {'LAT':<12} {'LON':<12} {'ALTITUDE':<12} {'SPEED':<10} {'TRACK':<8}")
        print(f"{'-'*140}")
        
        for idx, aircraft in enumerate(aircraft_list, 1):
            callsign = str(aircraft.get('call', 'N/A')).strip()[:15]
            registration = str(aircraft.get('r', 'N/A'))[:15]
            aircraft_type = str(aircraft.get('t', 'N/A'))[:12]
            latitude = aircraft.get('lat', 'N/A')
            longitude = aircraft.get('lon', 'N/A')
            altitude = aircraft.get('alt_baro', aircraft.get('alt_geom', 'N/A'))
            speed = aircraft.get('gs', 'N/A')
            track = aircraft.get('track', 'N/A')
            
            # Format numbers
            if isinstance(latitude, (int, float)):
                latitude = f"{latitude:.4f}"
            if isinstance(longitude, (int, float)):
                longitude = f"{longitude:.4f}"
            if isinstance(altitude, (int, float)):
                altitude = f"{int(altitude)} ft"
            if isinstance(speed, (int, float)):
                speed = f"{int(speed)} kt"
            if isinstance(track, (int, float)):
                track = f"{int(track)}°"
            
            print(f"{idx:<5} {callsign:<15} {registration:<15} {aircraft_type:<12} {str(latitude):<12} {str(longitude):<12} {str(altitude):<12} {str(speed):<10} {str(track):<8}")
        
        print(f"\n{'='*140}\n")


def main():
    """Main function - Interactive mode"""
    
    tracker = AircraftTracker()
    
    print("\n" + "🛫 "*20)
    print("AIRCRAFT TRACKER - WORKS WITH ANY COORDINATES")
    print("🛫 "*20)
    
    while True:
        try:
            print("\n📍 Enter your coordinates:")
            latitude = float(input("   Latitude (-90 to 90): "))
            longitude = float(input("   Longitude (-180 to 180): "))
            radius = int(input("   Radius in nm (1-250) [100]: ") or "100")
            
            # Validate
            if not (-90 <= latitude <= 90):
                print("❌ Latitude must be -90 to 90")
                continue
            if not (-180 <= longitude <= 180):
                print("❌ Longitude must be -180 to 180")
                continue
            if not (1 <= radius <= 250):
                print("❌ Radius must be 1-250")
                continue
            
            # Fetch data
            data = tracker.get_aircraft(latitude, longitude, radius)
            
            if data:
                # Display
                tracker.display_data(data)
               
            
            # Continue?
            cont = input("🔄 Search another location? (yes/no): ").strip().lower()
            if cont not in ['yes', 'y']:
                print("\n👋 Thank you!\n")
                break
        
        except ValueError:
            print("❌ Invalid input - please enter valid numbers")
        except KeyboardInterrupt:
            print("\n\n👋 Program interrupted\n")
            break


# One-line usage functions
def get_aircraft_simple(latitude, longitude, radius=100):
    """Simple function - just call this with any coordinates"""
    tracker = AircraftTracker()
    data = tracker.get_aircraft(latitude, longitude, radius)
    if data:
        tracker.display_data(data)
    return data


# Direct usage examples:
if __name__ == "__main__":
    
    # Method 1: Interactive mode (recommended)
    # Uncomment the line below to use interactive mode
    main()
    
    
    # Method 2: Direct use - uncomment any of these to test
    # =========================================================
    
    # Example 1: New York (40.7128, -74.0060)
    # data = get_aircraft_simple(40.7128, -74.0060, 100)
    
    # Example 2: London (51.5074, -0.1278)
    # data = get_aircraft_simple(51.5074, -0.1278, 100)
    
    # Example 3: Tokyo (35.6762, 139.6503)
    # data = get_aircraft_simple(35.6762, 139.6503, 100)
    
    # Example 4: Delhi, India (28.7041, 77.1025)
    # data = get_aircraft_simple(28.7041, 77.1025, 150)
    
    # Example 5: Mumbai, India (19.0760, 72.8855)
    # data = get_aircraft_simple(19.0760, 72.8855, 150)
    
    # Example 6: Any custom coordinates
    # data = get_aircraft_simple(YOUR_LATITUDE, YOUR_LONGITUDE, YOUR_RADIUS)
