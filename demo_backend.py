"""
Demo Backend - Camera Module Logic for Web Interface
Brings Camera module.py features to web browser
"""
import sys

# Critical for Render/Gunicorn with Gevent worker
# Must be done BEFORE other imports on non-Windows
if sys.platform != 'win32':
    from gevent import monkey
    monkey.patch_all()

import os

# Fix for Render/cloud deployment - set writable config directories BEFORE imports
os.environ['YOLO_CONFIG_DIR'] = '/tmp/ultralytics'
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'

import cv2
import numpy as np
import base64
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from datetime import datetime

app = Flask(__name__, static_folder='.', template_folder='.')

# Platform-aware async mode:
# - Windows (local dev): use 'threading' (gevent doesn't work well)
# - Linux (Render/cloud): use 'gevent' (required for WebSocket with Gunicorn)
if sys.platform == 'win32':
    async_mode = 'threading'
else:
    async_mode = 'gevent'
print(f"[SERVER] Platform: {sys.platform}, async_mode: {async_mode}")

socketio = SocketIO(app, cors_allowed_origins="*", async_mode=async_mode, 
                    ping_timeout=60, ping_interval=25, 
                    logger=True, engineio_logger=True)

# Initialize - Same as Camera module.py
MODEL_PATH = "yolov8n.pt"
CONF_THRESHOLD = 0.5
SKY_RATIO_THRESHOLD = 0.15
AIRCRAFT_CLASSES = {"airplane", "aeroplane", "drone", "uav", "aircraft"}

# Lazy loading - load on first request to avoid build timeout
_model = None
_engine = None

def get_model():
    """Lazy load YOLO model"""
    global _model
    if _model is None:
        from ultralytics import YOLO
        _model = YOLO(MODEL_PATH)
        print(f"[OK] YOLO Model loaded: {MODEL_PATH}")
    return _model

def get_engine():
    """Lazy load detection engine"""
    global _engine
    if _engine is None:
        from camera_engine_wrapper import CameraDetectionEngine
        _engine = CameraDetectionEngine(enable_database=True, enable_api=True)
        print(f"[OK] Backend ready - Session: {_engine.session_id}")
    return _engine

# Deduplication
last_detection_time = {}
DETECTION_COOLDOWN = 3.0

def estimate_sky_ratio(frame):
    """Sky detection - from Camera module.py"""
    h, w = frame.shape[:2]
    scale = 320 / w
    small = cv2.resize(frame, (320, int(h * scale)))
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    
    # Blue sky
    lower_blue = np.array([85, 20, 50])
    upper_blue = np.array([140, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # White/gray sky
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 40, 255])
    mask_white = cv2.inRange(hsv, lower_white, upper_white)
    
    mask = cv2.bitwise_or(mask_blue, mask_white)
    sky_pixels = cv2.countNonZero(mask)
    total_pixels = mask.shape[0] * mask.shape[1]
    return sky_pixels / total_pixels


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/health')
def health():
    return {'status': 'ok', 'async_mode': async_mode}

@app.route('/api/status')
def api_status():
    return {
        'status': 'running',
        'async_mode': async_mode,
        'model': MODEL_PATH,
        'message': 'Backend is operational'
    }


@socketio.on('connect')
def handle_connect():
    print("[OK] Client connected")
    try:
        session_id = get_engine().session_id
    except:
        session_id = "demo"
    emit('status', {'message': 'Connected', 'session_id': session_id})


@socketio.on('sensor_data')
def handle_sensor_data(data):
    """Receive GPS and compass data from browser"""
    pass


@socketio.on('get_nearby_aircraft')
def handle_get_nearby_aircraft(data):
    """Fetch nearby aircraft using ADSB.lol API"""
    try:
        from API import AircraftTracker
        
        lat = data.get('latitude')
        lon = data.get('longitude')
        radius = data.get('radius', 50)
        
        if lat is None or lon is None:
            emit('nearby_aircraft', {'error': 'Missing coordinates'})
            return
        
        tracker = AircraftTracker()
        aircraft_data = tracker.get_aircraft(lat, lon, radius)
        
        if aircraft_data and 'ac' in aircraft_data:
            aircraft_list = []
            for ac in aircraft_data['ac']:
                aircraft_list.append({
                    'callsign': str(ac.get('call', 'N/A')).strip(),
                    'registration': ac.get('r', 'N/A'),
                    'type': ac.get('t', 'N/A'),
                    'lat': ac.get('lat'),
                    'lon': ac.get('lon'),
                    'altitude': ac.get('alt_baro', ac.get('alt_geom', 0)),
                    'speed': ac.get('gs', 0),
                    'track': ac.get('track', 0),
                    'hex': ac.get('hex', 'N/A')
                })
            
            emit('nearby_aircraft', {'aircraft': aircraft_list, 'count': len(aircraft_list)})
            print(f"[AIRCRAFT] Found {len(aircraft_list)} aircraft within {radius}nm")
        else:
            emit('nearby_aircraft', {'aircraft': [], 'count': 0})
            print(f"No aircraft found within {radius}nm")
    
    except Exception as e:
        print(f"Error fetching aircraft: {e}")
        emit('nearby_aircraft', {'error': str(e)})


@socketio.on('process_frame')
def handle_process_frame(data):
    """Process frame - Camera module.py logic"""
    try:
        print(f"[FRAME] Received frame from client")
        
        # Decode frame with validation
        if 'frame' not in data:
            print("[ERROR] No frame data received")
            return
            
        frame_parts = data['frame'].split(',')
        if len(frame_parts) < 2:
            print("[ERROR] Invalid frame format")
            return
            
        frame_data = frame_parts[1]
        frame_bytes = base64.b64decode(frame_data)
        nparr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            print("[ERROR] Failed to decode frame")
            return
            
        print(f"[FRAME] Decoded successfully - size: {frame.shape}")
        
        # Sky check (lightweight)
        sky_ratio = estimate_sky_ratio(frame)
        sky_ok = sky_ratio >= SKY_RATIO_THRESHOLD
        
        if not sky_ok:
            emit('sky_warning', {
                'sky_coverage': sky_ratio * 100,
                'message': f'Sky coverage: {sky_ratio*100:.1f}% - Point camera at sky'
            })
        
        detections = []
        annotated_frame = frame.copy()
        
        # Add sky overlay
        cv2.putText(annotated_frame, f"Sky: {sky_ratio*100:.1f}%", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if sky_ok else (0, 0, 255), 2)
        
        # Run YOLO with cloud optimization
        try:
            model = get_model()
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Cloud-optimized: smaller image size, faster inference
            is_cloud = os.environ.get('PORT') is not None
            imgsz = 320 if is_cloud else 640  # Smaller for cloud speed
            
            print(f"[YOLO] Running inference (imgsz={imgsz})...")
            results = model.predict(source=rgb_frame, verbose=False, conf=CONF_THRESHOLD, 
                                  imgsz=imgsz, max_det=5)  # Reduced max_det for speed
            print(f"[YOLO] Inference complete")
            
            if len(results) > 0:
                result = results[0]
                boxes = result.boxes
                
                if boxes is not None:
                    for box in boxes:
                        conf = float(box.conf.cpu().numpy())
                        cls_idx = int(box.cls.cpu().numpy())
                        cls_name = model.names.get(cls_idx, str(cls_idx)).lower()
                        
                        # Filter aircraft only
                        if not any(ac in cls_name for ac in AIRCRAFT_CLASSES):
                            continue
                        
                        # Deduplication
                        detection_key = f"{cls_name}_{int(conf*100)}"
                        current_time = datetime.now()
                        if detection_key in last_detection_time:
                            time_diff = (current_time - last_detection_time[detection_key]).total_seconds()
                            if time_diff < DETECTION_COOLDOWN:
                                continue
                        
                        last_detection_time[detection_key] = current_time
                        
                        xyxy = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = map(int, xyxy)
                        
                        # Process with backend
                        engine = get_engine()
                        enhanced = engine.process_detection({
                            'bbox': xyxy,
                            'conf': conf,
                            'class': cls_name,
                            'timestamp': current_time,
                            'frame_number': data.get('timestamp', 0)
                        })
                        
                        # Color by threat level
                        colors = {
                            'none': (0, 255, 0),
                            'medium': (0, 255, 255),
                            'high': (0, 165, 255),
                            'critical': (0, 0, 255)
                        }
                        color = colors.get(enhanced['threat_level'], (0, 255, 0))
                        
                        # Draw bounding box
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 3)
                        
                        # Label
                        label = f"{cls_name} {conf:.2f}"
                        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                        cv2.rectangle(annotated_frame, (x1, y1 - th - 10), (x1 + tw, y1), color, -1)
                        cv2.putText(annotated_frame, label, (x1, y1 - 6), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                        
                        # Threat badge
                        threat_text = f"THREAT: {enhanced['threat_level'].upper()}"
                        cv2.putText(annotated_frame, threat_text, (x1, y2 + 25),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        
                        detections.append({
                            'object_type': cls_name.upper(),
                            'confidence': conf,
                            'bbox': {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2},
                            'threat_level': enhanced['threat_level'],
                            'verified': enhanced['verified'],
                            'flight_number': enhanced.get('flight_number'),
                            'timestamp': current_time.isoformat()
                        })
                        
                        print(f"[DETECT] {cls_name.upper()} | {conf:.0%} | {enhanced['threat_level'].upper()}")
        
        except Exception as yolo_error:
            print(f"[YOLO ERROR] {yolo_error}")
            # Continue with non-annotated frame
        
        # Send detections
        if detections:
            for det in detections:
                emit('detection_update', det, broadcast=True)
        
        # Send annotated frame with optimized quality
        try:
            # Lower JPEG quality on cloud for speed
            quality = 60 if os.environ.get('PORT') else 75
            _, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
            annotated_b64 = base64.b64encode(buffer).decode('utf-8')
            
            emit('annotated_frame', {
                'frame': f'data:image/jpeg;base64,{annotated_b64}',
                'detections_count': len(detections),
                'sky_ok': sky_ok
            })
            print(f"[FRAME] Sent annotated frame (detections: {len(detections)})")
            
        except Exception as encode_error:
            print(f"[ENCODE ERROR] {encode_error}")
        
    except Exception as e:
        print(f"[ERROR] Frame processing failed: {e}")
        import traceback
        traceback.print_exc()
        # Send error frame indicator
        try:
            emit('frame_error', {'error': str(e)})
        except:
            pass



@socketio.on('disconnect')
def handle_disconnect():
    print("Client disconnected")


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    host = '0.0.0.0'
    print("\n" + "=" * 60)
    print("   AIRCRAFT DETECTION - DEMO MODE")
    print("=" * 60)
    print(f"[OK] Model: {MODEL_PATH}")
    print(f"[OK] Host: {host}")
    print(f"[OK] Port: {port}")
    print(f"[OK] Async Mode: {async_mode}")
    if os.environ.get('PORT'):
        print(f"[OK] Running on Render (Cloud)")
    else:
        print(f"[OK] Running locally - http://localhost:{port}")
    print("=" * 60 + "\n")
    
    socketio.run(app, host='0.0.0.0', port=port, debug=False, allow_unsafe_werkzeug=True)
