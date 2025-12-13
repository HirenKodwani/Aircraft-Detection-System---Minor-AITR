"""
AERIES Backend - Ultra Lightweight Version
Optimized for low RAM usage - processes 1 frame per second
"""
import sys
import time

if sys.platform != 'win32':
    from gevent import monkey
    monkey.patch_all()

import os
os.environ['YOLO_CONFIG_DIR'] = '/tmp/ultralytics'
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'

import cv2
import numpy as np
import base64
from flask import Flask, render_template
from flask_socketio import SocketIO, emit

app = Flask(__name__, static_folder='.', template_folder='.')
async_mode = 'threading' if sys.platform == 'win32' else 'gevent'

socketio = SocketIO(app, cors_allowed_origins="*", async_mode=async_mode,
                    ping_timeout=60, ping_interval=25,
                    logger=False, engineio_logger=False)

# Config
MODEL_PATH = "yolov8n.pt"
CONF_THRESHOLD = 0.25
SKY_RATIO_THRESHOLD = 0.20
PREFERRED_CLASSES = {"airplane", "aeroplane", "drone", "uav"}

# Rate limiting - process only 1 frame per second to prevent crashes
last_process_time = 0
MIN_INTERVAL = 1.0  # 1 second between frames

# Lazy model
_model = None

def get_model():
    global _model
    if _model is None:
        print("[YOLO] Loading model...")
        from ultralytics import YOLO
        _model = YOLO(MODEL_PATH)
        print("[YOLO] Model loaded!")
    return _model


def estimate_sky_ratio(frame):
    h, w = frame.shape[:2]
    scale = 320 / w
    small = cv2.resize(frame, (320, int(h * scale)))
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    
    lower_blue = np.array([85, 20, 50])
    upper_blue = np.array([140, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 40, 255])
    mask_white = cv2.inRange(hsv, lower_white, upper_white)
    
    mask = cv2.bitwise_or(mask_blue, mask_white)
    return cv2.countNonZero(mask) / (mask.shape[0] * mask.shape[1])


def draw_detection(frame, box, conf, cls_name, color=(0, 200, 0)):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    label = f"{cls_name} {conf:.2f}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
    cv2.putText(frame, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)


@app.route('/')
def index():
    return render_template('landing.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/health')
def health():
    return {'status': 'ok'}


@socketio.on('connect')
def on_connect():
    print("[OK] Client connected")
    emit('status', {'message': 'connected'})


@socketio.on('get_nearby_aircraft')
def handle_aircraft(data):
    try:
        from API import AircraftTracker
        lat = data.get('latitude')
        lon = data.get('longitude')
        
        if not lat or not lon:
            emit('nearby_aircraft', {'aircraft': []})
            return
        
        tracker = AircraftTracker()
        result = tracker.get_aircraft(lat, lon, data.get('radius', 50))
        
        if result and 'ac' in result:
            aircraft = []
            for ac in result['ac']:
                flight_info = ac.get('flight', {}) if isinstance(ac.get('flight'), dict) else {}
                aircraft.append({
                    'callsign': str(ac.get('call', 'N/A')).strip(),
                    'registration': ac.get('r', 'N/A'),
                    'type': ac.get('t', 'N/A'),
                    'lat': ac.get('lat'),
                    'lon': ac.get('lon'),
                    'altitude': ac.get('alt_baro', ac.get('alt_geom', 0)),
                    'speed': ac.get('gs', 0),
                    'track': ac.get('track', 0),
                    'origin': flight_info.get('origin', 'N/A'),
                    'destination': flight_info.get('destination', 'N/A')
                })
            emit('nearby_aircraft', {'aircraft': aircraft})
            print(f"[MAP] Sent {len(aircraft)} aircraft")
        else:
            emit('nearby_aircraft', {'aircraft': []})
    except Exception as e:
        print(f"[ERROR] Aircraft: {e}")
        emit('nearby_aircraft', {'aircraft': []})


@socketio.on('process_frame')
def handle_frame(data):
    """Rate-limited frame processing to prevent computer crashes"""
    global last_process_time
    
    # Rate limit - only process 1 frame per second
    current_time = time.time()
    if current_time - last_process_time < MIN_INTERVAL:
        return  # Skip this frame
    last_process_time = current_time
    
    try:
        if 'frame' not in data:
            return
        
        parts = data['frame'].split(',')
        if len(parts) < 2:
            return
        
        frame_bytes = base64.b64decode(parts[1])
        frame = cv2.imdecode(np.frombuffer(frame_bytes, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return
        
        print(f"[CAMERA] Processing frame {frame.shape}")
        
        # Sky check
        sky_ratio = estimate_sky_ratio(frame)
        sky_ok = sky_ratio >= SKY_RATIO_THRESHOLD
        
        if not sky_ok:
            cv2.putText(frame, "Please point the camera to the SKY", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Sky coverage: {sky_ratio*100:.1f}%", (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1, cv2.LINE_AA)
        
        # YOLO detection
        detections = []
        try:
            model = get_model()
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Use smaller image size (320) for faster processing
            results = model.predict(source=rgb, verbose=False, conf=CONF_THRESHOLD, imgsz=320, max_det=10)
            
            if results and len(results) > 0 and results[0].boxes is not None:
                for box in results[0].boxes:
                    conf = float(box.conf.cpu().numpy())
                    cls_idx = int(box.cls.cpu().numpy())
                    cls_name = model.names.get(cls_idx, str(cls_idx)).lower()
                    
                    if not any(p in cls_name for p in PREFERRED_CLASSES):
                        continue
                    
                    xyxy = box.xyxy[0].cpu().numpy()
                    draw_detection(frame, xyxy, conf, cls_name)
                    detections.append({'class': cls_name, 'confidence': conf, 'threat': 'high' if 'drone' in cls_name else 'medium'})
                    print(f"[DETECT] {cls_name} @ {conf:.0%}")
        except Exception as e:
            print(f"[YOLO ERROR] {e}")
        
        # Encode and send
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
        b64 = base64.b64encode(buffer).decode('utf-8')
        
        emit('annotated_frame', {
            'frame': f'data:image/jpeg;base64,{b64}',
            'sky_ok': sky_ok,
            'detections': detections
        })
        print(f"[CAMERA] Sent annotated frame")
        
    except Exception as e:
        print(f"[ERROR] {e}")


@socketio.on('disconnect')
def on_disconnect():
    print("[OK] Client disconnected")


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print("\n" + "="*50)
    print("   AERIES - Ultra Light Backend")
    print("   Processing: 1 frame/sec (prevents crashes)")
    print("="*50)
    print(f"   http://localhost:{port}")
    print("="*50 + "\n")
    socketio.run(app, host='0.0.0.0', port=port, debug=False, allow_unsafe_werkzeug=True)
