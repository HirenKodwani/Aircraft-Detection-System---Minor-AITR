# AERIES - Aerial Recognition & Intelligence System
## Complete User Manual

---

## Table of Contents
1. [Introduction](#1-introduction)
2. [System Requirements](#2-system-requirements)
3. [Installation Guide](#3-installation-guide)
4. [Configuration](#4-configuration)
5. [User Interface Guide](#5-user-interface-guide)
6. [File Reference](#6-file-reference)
7. [API Reference](#7-api-reference)
8. [Troubleshooting](#8-troubleshooting)
9. [Deployment Guide](#9-deployment-guide)

---

## 1. Introduction

### 1.1 What is AERIES?
AERIES (Aerial Recognition & Intelligence System) is a real-time aircraft detection and tracking system that combines:
- **Computer Vision**: YOLOv8-based aircraft detection from camera feed
- **Live Tracking**: Real-time aircraft positions via ADSB.lol API
- **Dark Military UI**: Tactical dashboard with radar aesthetics

### 1.2 Key Features
✈️ **Camera Detection**
- Real-time aircraft/drone detection using YOLOv8n
- Sky coverage validation
- Threat level classification
- Bounding box visualization

🗺️ **Map Tracking**
- Live aircraft positions from ADSB.lol
- Detailed flight information (callsign, origin, destination, altitude, speed)
- Custom aircraft markers with popups
- GPS-based user location

🎨 **Military UI**
- Dark tactical theme
- HUD-style interface
- Real-time clock and status indicators
- Landscape layout (camera + map)

### 1.3 Technology Stack
- **Backend**: Python 3.11, Flask, Flask-SocketIO
- **Computer Vision**: OpenCV, Ultralytics YOLOv8
- **Frontend**: HTML5, CSS3, JavaScript, Leaflet.js
- **External APIs**: ADSB.lol (aircraft tracking)

---

## 2. System Requirements

### 2.1 Hardware Requirements
**Minimum:**
- CPU: Dual-core processor
- RAM: 2GB
- Webcam: Any USB or built-in camera
- Storage: 500MB free space

**Recommended:**
- CPU: Quad-core processor
- RAM: 4GB+
- Webcam: HD camera (720p+)
- Internet: Stable connection for API calls

### 2.2 Software Requirements
- **Operating System**: Windows 10/11, Linux, macOS
- **Python**: Version 3.11 (specified in runtime.txt)
- **Browser**: Chrome, Firefox, Edge (modern browsers with webcam support)

### 2.3 Browser Permissions Required
- Camera access
- Geolocation access

---

## 3. Installation Guide

### 3.1 Local Setup (Windows/Linux/Mac)

#### Step 1: Install Python 3.11
Download from: https://www.python.org/downloads/

Verify installation:
```bash
python --version
# Should show: Python 3.11.x
```

#### Step 2: Clone/Download Project
```bash
cd "C:\Users\ADMIN\Downloads\Aircraft detection System"
```

#### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

**Dependencies installed:**
- Flask & Flask-SocketIO (web server)
- OpenCV (image processing)
- Ultralytics (YOLOv8)
- Requests, Pandas (data handling)
- Gunicorn, Gevent (production server)

#### Step 4: Verify YOLO Model
Ensure `yolov8n.pt` exists in project root (6.5MB file)

#### Step 5: Run Server
```bash
python demo_backend.py
```

Expected output:
```
==================================================
   AERIES - Ultra Light Backend
   Processing: 1 frame/sec (prevents crashes)
==================================================
   http://localhost:5000
==================================================
```

#### Step 6: Access Application
Open in browser: http://localhost:5000

---

## 4. Configuration

### 4.1 Environment Variables (.env)
Create `.env` file in project root (use `.env.example` as template):

```env
# Optional configurations
PORT=5000                    # Server port (default: 5000)
YOLO_CONFIG_DIR=/tmp/ultralytics  # YOLO cache directory
```

### 4.2 Backend Configuration (demo_backend.py)
Key parameters you can adjust:

```python
# Line 30-33: Detection thresholds
MODEL_PATH = "yolov8n.pt"         # YOLO model file
CONF_THRESHOLD = 0.25             # Detection confidence (0.0-1.0)
SKY_RATIO_THRESHOLD = 0.20        # Sky coverage threshold
PREFERRED_CLASSES = {"airplane", "aeroplane", "drone", "uav"}

# Line 37: Frame rate limiting
MIN_INTERVAL = 1.0  # Process 1 frame per second (prevents crashes)
```

**Adjusting Performance:**
- Lower `CONF_THRESHOLD` (e.g., 0.15) → More detections, more false positives
- Higher `CONF_THRESHOLD` (e.g., 0.40) → Fewer detections, higher accuracy
- Increase `MIN_INTERVAL` → Lower CPU usage, slower updates
- Decrease `MIN_INTERVAL` → Higher CPU usage, faster updates

### 4.3 Frontend Configuration (dashboard.html)
Camera settings (Line 336-341):

```javascript
video: {
    width: { ideal: 640 },      // Camera resolution
    height: { ideal: 480 },
    facingMode: "environment"   // Rear camera on mobile
}
```

Frame processing (Line 390):
```javascript
}, 100);  // Send frame every 100ms (10 FPS)
```

**Note:** Backend limits processing to 1 FPS regardless of frontend send rate (prevents crashes).

---

## 5. User Interface Guide

### 5.1 Landing Page (landing.html)
**URL:** http://localhost:5000

**Components:**
- Animated radar display
- "INITIALIZE SYSTEM" button → Navigates to Dashboard
- Dark military theme

### 5.2 Dashboard (dashboard.html)
**URL:** http://localhost:5000/dashboard

#### Layout
```
┌─────────────────────────────────────────────────┐
│ AERIES | Status Bar | Clock | GPS Info          │
├──────────────┬──────────────────────────────────┤
│              │                                  │
│  OPTICAL     │         MAP VIEW                 │
│  SENSOR      │                                  │
│  (Camera)    │    (Aircraft markers)            │
│              │                                  │
├──────────────┤                                  │
│              │                                  │
│  TARGET      │                                  │
│  FEED        │                                  │
│  (Detections)│                                  │
│              │                                  │
└──────────────┴──────────────────────────────────┘
```

#### 5.2.1 Header Bar
- **Connection Status**: Green dot = connected to server
- **GPS Status**: Green dot = location acquired
- **UTC Clock**: Real-time UTC timestamp

#### 5.2.2 Optical Sensor (Camera Module)
**Initialize Camera:**
1. Click "INITIALIZE CAMERA" button
2. Grant camera permission when prompted
3. Point camera at sky

**Camera Status:**
- "REQUESTING ACCESS..." → Waiting for permission
- "SENSOR ACTIVE - PROCESSING" → Camera active
- "WARN: POINT TO SKY" → Insufficient sky coverage
- "TARGET ACQUISITION ACTIVE" → Ready for detection

**Terminate Feed:**
- Click "TERMINATE FEED" to stop camera

**Sky Detection:**
- System checks for blue/white sky pixels
- Minimum 20% sky coverage required
- Red warning text appears if insufficient

**Aircraft Detection:**
- Green bounding boxes around detected aircraft
- Label shows: "airplane 0.85" (class + confidence)
- Updates every 1 second

#### 5.2.3 Target Feed (Detections List)
Displays two types of detections:

**[CAM] Camera Detections** (Cyan color)
- Class: airplane/drone
- Confidence: 0-100%
- Threat level: HIGH/MEDIUM

**[MAP] Map Detections** (Green color)
- Callsign: Flight identifier
- Altitude: Feet
- Speed: Knots

Maximum 50 recent detections displayed (newest first).

#### 5.2.4 Map View
**Features:**
- Dark inverted map theme
- User location: Green circle marker
- Aircraft: Cyan airplane icons

**Aircraft Popups:**
Click any aircraft marker to see:
- ✈ Callsign
- Registration number
- Aircraft type
- Origin airport
- Destination airport
- Altitude (feet)
- Speed (knots)
- Heading (degrees)

**Controls:**
- Scroll to zoom
- Drag to pan
- Zoom controls (bottom-right corner)

---

## 6. File Reference

### 6.1 Core Application Files

#### demo_backend.py (Main Backend Server)
**Purpose:** Primary application server
**Size:** 7.8KB
**Lines:** 226

**Key Functions:**
- `get_model()` - Lazy loads YOLO model
- `estimate_sky_ratio(frame)` - Calculates sky coverage percentage
- `draw_detection(frame, box, conf, cls_name)` - Draws bounding boxes
- `handle_frame(data)` - Processes camera frames
- `handle_aircraft(data)` - Fetches nearby aircraft

**Routes:**
- `/` → landing.html
- `/dashboard` → dashboard.html
- `/health` → Health check endpoint

**Socket.IO Events:**
- `connect` - Client connection
- `process_frame` - Receive camera frame, return annotated frame
- `get_nearby_aircraft` - Fetch aircraft data
- `disconnect` - Client disconnection

**Configuration:**
- Rate limiting: 1 frame/second (prevents crashes)
- YOLO image size: 320px (fast processing)
- Max detections: 10 per frame
- JPEG quality: 60%

#### API.py (Aircraft Tracking)
**Purpose:** ADSB.lol API integration
**Size:** 6.7KB
**Lines:** 189

**Class:** AircraftTracker
- `get_aircraft(lat, lon, radius)` - Fetch aircraft within radius
- `display_data(data)` - Console output formatting

**API Endpoint:** https://api.adsb.lol/v2/point/{lat}/{lon}/{radius}
**Radius:** 1-250 nautical miles
**Response:** JSON with aircraft array

**Data Fields:**
- `call` - Callsign
- `r` - Registration
- `t` - Aircraft type
- `lat`, `lon` - Coordinates
- `alt_baro` - Altitude (feet)
- `gs` - Ground speed (knots)
- `track` - Heading (degrees)
- `flight.origin` - Departure airport
- `flight.destination` - Arrival airport

#### landing.html (Entry Page)
**Purpose:** Application landing page
**Size:** 5.7KB

**Features:**
- Animated radar SVG
- "INITIALIZE SYSTEM" button
- Dark military aesthetic
- Auto-redirects to /dashboard on click

#### dashboard.html (Main UI)
**Purpose:** Primary user interface
**Size:** 19.4KB
**Lines:** 548

**Sections:**
1. Header (status indicators)
2. Sidebar (camera + detections)
3. Map view (Leaflet.js)

**Libraries:**
- Leaflet.js 1.9.4 (map rendering)
- Socket.IO 4.5.4 (WebSocket communication)

**JavaScript Functions:**
- `toggleCamera()` - Start/stop camera
- `processLoop()` - Capture and send frames
- `fetchAircraft()` - Request aircraft data
- `addDetection()` - Add to detections list

### 6.2 Configuration Files

#### requirements.txt
**Purpose:** Python dependencies
**Size:** 286 bytes

**Dependencies:**
```
flask>=2.0.0                  # Web framework
flask-socketio>=5.0.0         # WebSocket support
opencv-python-headless>=4.5.0 # Computer vision
numpy>=1.21.0                 # Array operations
ultralytics>=8.0.0            # YOLOv8
requests>=2.25.0              # HTTP requests
pandas>=1.3.0                 # Data manipulation
gunicorn>=20.0.0              # Production server
gevent>=21.0.0                # Async I/O
```

#### runtime.txt
**Purpose:** Python version for deployment
**Content:** `python-3.11.9`

#### Procfile
**Purpose:** Render/Heroku deployment configuration
**Content:**
```
web: gunicorn -k geventwebsocket.gunicorn.workers.GeventWebSocketWorker -w 1 -b 0.0.0.0:$PORT demo_backend:app --timeout 120
```

**Explanation:**
- `gunicorn` - Production WSGI server
- `-k geventwebsocket...` - WebSocket worker
- `-w 1` - Single worker (low RAM usage)
- `--timeout 120` - 2-minute timeout for model loading

#### .env.example
**Purpose:** Environment variables template

#### .gitignore
**Purpose:** Git exclusion rules
**Excludes:** `__pycache__/`, `.env`, `*.db`, `*.log`

### 6.3 Model Files

#### yolov8n.pt
**Purpose:** YOLOv8 Nano model weights
**Size:** 6.5MB
**Classes:** 80 COCO classes (airplane, person, car, etc.)
**Source:** Ultralytics

**Model Info:**
- Architecture: YOLOv8n (nano variant)
- Input size: 320px or 640px
- Speed: ~100ms inference on CPU
- Accuracy: mAP 37.3%

### 6.4 Data Directories

#### captures/
**Purpose:** Saved camera frames (user-created)
**Auto-created:** Yes

#### logs/
**Purpose:** Application logs (if logging enabled)
**Auto-created:** Yes

---

## 7. API Reference

### 7.1 HTTP Routes

#### GET /
**Description:** Landing page
**Response:** HTML (landing.html)
**Status:** 200 OK

#### GET /dashboard
**Description:** Main application dashboard
**Response:** HTML (dashboard.html)
**Status:** 200 OK

#### GET /health
**Description:** Health check endpoint
**Response:** JSON `{"status": "ok"}`
**Status:** 200 OK

### 7.2 Socket.IO Events

#### Client → Server Events

**connect**
- **Trigger:** Client connects to server
- **Payload:** None
- **Response:** `status` event with `{"message": "connected"}`

**process_frame**
- **Trigger:** Client sends camera frame
- **Payload:**
  ```javascript
  {
    "frame": "data:image/jpeg;base64,/9j/4AAQ..."
  }
  ```
- **Response:** `annotated_frame` event
- **Rate Limit:** Server processes max 1 frame/second

**get_nearby_aircraft**
- **Trigger:** Client requests aircraft data
- **Payload:**
  ```javascript
  {
    "latitude": 28.7041,
    "longitude": 77.1025,
    "radius": 50  // nautical miles
  }
  ```
- **Response:** `nearby_aircraft` event

**disconnect**
- **Trigger:** Client disconnects
- **Payload:** None
- **Response:** None

#### Server → Client Events

**status**
- **Trigger:** On connection
- **Payload:**
  ```javascript
  {
    "message": "connected"
  }
  ```

**annotated_frame**
- **Trigger:** After processing camera frame
- **Payload:**
  ```javascript
  {
    "frame": "data:image/jpeg;base64,/9j/4AAQ...",
    "sky_ok": true,
    "detections": [
      {
        "class": "airplane",
        "confidence": 0.87,
        "threat": "medium"
      }
    ]
  }
  ```

**nearby_aircraft**
- **Trigger:** After fetching aircraft data
- **Payload:**
  ```javascript
  {
    "aircraft": [
      {
        "callsign": "AI101",
        "registration": "VT-TCU",
        "type": "A320",
        "lat": 28.5562,
        "lon": 77.1008,
        "altitude": 35000,
        "speed": 450,
        "track": 270,
        "origin": "DEL",
        "destination": "BOM"
      }
    ]
  }
  ```

---

## 8. Troubleshooting

### 8.1 Common Issues

#### Issue: Camera not showing feed

**Symptoms:**
- "NO SIGNAL" displayed
- Image not updating
- Button stuck on "TERMINATE FEED"

**Solutions:**
1. **Check browser console (F12)**
   - Look for `[CAMERA]` logs
   - Check for permission errors

2. **Grant camera permission**
   - Click lock icon in address bar
   - Allow camera access
   - Refresh page

3. **Check camera device**
   - Ensure webcam is connected
   - Try different camera if multiple available
   - Test camera in other apps

4. **Hard refresh browser**
   - Windows: Ctrl + Shift + R
   - Mac: Cmd + Shift + R

5. **Restart server**
   ```bash
   # Stop: Ctrl+C
   python demo_backend.py
   ```

#### Issue: Computer crashes/freezes

**Cause:** Too much processing load

**Solutions:**
1. **Increase MIN_INTERVAL** (demo_backend.py line 37)
   ```python
   MIN_INTERVAL = 2.0  # Process every 2 seconds instead of 1
   ```

2. **Use smaller YOLO image** (demo_backend.py line 169)
   ```python
   results = model.predict(..., imgsz=224, ...)  # Even smaller
   ```

3. **Reduce max detections** (demo_backend.py line 169)
   ```python
   results = model.predict(..., max_det=5, ...)  # Fewer detections
   ```

4. **Close other applications**
   - Free up RAM and CPU resources

#### Issue: No aircraft showing on map

**Symptoms:**
- Map loads but no markers
- GPS indicator red

**Solutions:**
1. **Grant location permission**
   - Click permission prompt
   - Allow location access

2. **Check GPS signal**
   - Move near window
   - Wait 10-30 seconds for GPS lock

3. **Check console for errors**
   - Look for `[AIRCRAFT]` logs
   - Verify API response

4. **Try larger radius**
   - Modify dashboard.html line 457:
   ```javascript
   radius: 200  // Increase search radius
   ```

5. **Check ADSB.lol API status**
   - Test manually: https://api.adsb.lol/v2/point/28.7/77.1/100
   - Should return JSON with aircraft array

#### Issue: WebSocket connection fails

**Symptoms:**
- "LINK LOST" in status bar
- Console shows Socket.IO errors
- No frame processing

**Solutions:**
1. **Check server is running**
   - Terminal should show "Running on http://..."

2. **Restart server**
   - Stop (Ctrl+C) and restart

3. **Check firewall**
   - Allow Python through Windows Firewall

4. **Try different port**
   - Modify .env: `PORT=5001`
   - Restart server

#### Issue: Sky detection too strict

**Symptoms:**
- Always shows "POINT TO SKY" warning
- Even when pointing at clear sky

**Solution:**
Lower threshold in demo_backend.py line 32:
```python
SKY_RATIO_THRESHOLD = 0.10  # More lenient (was 0.20)
```

#### Issue: YOLO model download fails

**Symptoms:**
- Error on first run
- "Model not found"

**Solutions:**
1. **Auto-download** (happens on first run):
   ```python
   # Ultralytics automatically downloads yolov8n.pt
   ```

2. **Manual download:**
   - Visit: https://github.com/ultralytics/assets/releases
   - Download yolov8n.pt
   - Place in project root

### 8.2 Performance Optimization

#### For Low-End Systems
```python
# demo_backend.py adjustments:
MIN_INTERVAL = 2.0        # Slower processing
imgsz=224                 # Smaller images
max_det=3                 # Fewer detections
CONF_THRESHOLD = 0.40     # Higher confidence
```

#### For High-End Systems
```python
# demo_backend.py adjustments:
MIN_INTERVAL = 0.5        # Faster processing
imgsz=640                 # Larger images
max_det=30                # More detections
CONF_THRESHOLD = 0.20     # Lower confidence
```

### 8.3 Logging and Debugging

**Enable detailed logging:**
```python
# demo_backend.py - add after imports:
import logging
logging.basicConfig(level=logging.DEBUG)
```

**Check server logs:**
- Terminal output shows all `print()` statements
- Look for `[YOLO]`, `[CAMERA]`, `[MAP]` prefixes

**Browser console (F12):**
- `[SOCKET]` - Connection status
- `[CAMERA]` - Camera events
- `[GPS]` - Location updates
- `[AIRCRAFT]` - Map data

---

## 9. Deployment Guide

### 9.1 Cloud Deployment (Render)

#### Prerequisites
1. GitHub account
2. Render account (free tier: https://render.com)
3. Project pushed to GitHub

#### Step 1: Push to GitHub
```bash
git init
git add .
git commit -m "AERIES deployment"
git remote add origin https://github.com/YOUR_USERNAME/aeries.git
git push -u origin main
```

#### Step 2: Create Render Web Service
1. Go to https://dashboard.render.com
2. Click "New +" → "Web Service"
3. Connect GitHub repository
4. Configure:
   - **Name:** aeries-detection
   - **Environment:** Python 3
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** (uses Procfile automatically)
   - **Plan:** Free

#### Step 3: Configure Environment
Add environment variables:
- `PYTHON_VERSION`: 3.11.9

#### Step 4: Deploy
- Click "Create Web Service"
- Wait 5-10 minutes for deployment
- Render provides HTTPS URL: https://aeries-detection.onrender.com

#### Step 5: Test
- Visit provided URL
- Camera/GPS require HTTPS (Render provides)

### 9.2 Memory Optimization for Free Tier

Render free tier: 512MB RAM limit

**Current optimizations:**
- Single Gunicorn worker (`-w 1`)
- Small YOLO model (yolov8n.pt)
- Image size: 320px
- Rate limiting: 1 frame/sec
- JPEG quality: 60%

**If OOM errors occur:**
1. Further reduce image size to 224px
2. Increase MIN_INTERVAL to 2.0 seconds
3. Lower max_det to 5

### 9.3 Local Network Access

**Access from other devices on same network:**
1. Find your IP address:
   ```bash
   # Windows:
   ipconfig
   # Look for IPv4 Address (e.g., 192.168.1.100)
   
   # Linux/Mac:
   ifconfig
   ```

2. Access from other devices:
   - http://192.168.1.100:5000

**Note:** Camera/GPS require HTTPS on mobile browsers (use ngrok for testing).

### 9.4 Production Considerations

For serious deployment:

1. **Use proper server:**
   - Replace Werkzeug with Gunicorn (already in Procfile)

2. **Add authentication:**
   - Protect sensitive routes
   - Use Flask-Login

3. **Database logging:**
   - Store detections in PostgreSQL
   - Track aircraft history

4. **SSL Certificate:**
   - Required for camera/GPS on production domain
   - Render provides free SSL

5. **Monitoring:**
   - Use Render metrics
   - Add error tracking (Sentry)

6. **Rate limiting:**
   - Limit API calls to prevent abuse

---

## Appendix A: Keyboard Shortcuts

**Browser:**
- `F12` - Open developer console
- `Ctrl+R` / `F5` - Refresh page
- `Ctrl+Shift+R` - Hard refresh (clear cache)
- `Ctrl+Shift+I` - Inspect element

**Dashboard:**
- Click camera button - Toggle camera
- Click aircraft marker - Show details
- Scroll map - Zoom in/out
- Drag map - Pan

---

## Appendix B: File Size Reference

```
yolov8n.pt              6.5 MB    (YOLO model)
dashboard.html          19 KB     (Main UI)
aeries.html             20 KB     (Obsolete - DELETE)
index.html              20 KB     (Obsolete - DELETE)
demo_backend.py         7.8 KB    (Backend server)
API.py                  6.7 KB    (Aircraft API)
landing.html            5.7 KB    (Entry page)
requirements.txt        286 B     (Dependencies)
```

---

## Appendix C: Technology Credits

**Open Source Libraries:**
- **YOLOv8:** Ultralytics (https://github.com/ultralytics/ultralytics)
- **Flask:** Pallets Projects (https://flask.palletsprojects.com)
- **Leaflet.js:** Vladimir Agafonkin (https://leafletjs.com)
- **OpenCV:** Open Source Computer Vision (https://opencv.org)

**APIs:**
- **ADSB.lol:** Free aircraft tracking API (https://adsb.lol)

---

## Support and Updates

**Project Repository:** GitHub (if available)
**Issues:** Report via GitHub Issues
**Updates:** Check for new releases

**Version:** 2.0 (Ultra Light)
**Last Updated:** December 2025

---

**END OF USER MANUAL**
