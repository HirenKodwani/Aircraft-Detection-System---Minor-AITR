"""
AERIES Backend - Minimal Version
Matches Camera module.py exactly with web serving
"""
import sys

# Gevent for cloud only
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

# Config - EXACTLY as Camera module.py
MODEL_PATH = "yolov8n.pt"
CONF_THRESHOLD = 0.25  # Same as Camera module.py
SKY_RATIO_THRESHOLD = 0.20  # Same as Camera module.py
PREFERRED_CLASSES = {"airplane", "aeroplane", "drone", "uav"}

# Lazy model
_model = None

def get_model():
    global _model
    if _model is None:
        from ultralytics import YOLO
        _model = YOLO(MODEL_PATH)
    return _model


def estimate_sky_ratio(frame):
    """Exact copy from Camera module.py"""
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
    """Exact copy from Camera module.py"""
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    label = f"{cls_name} {conf:.2f}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
    cv2.putText(frame, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)


# Routes
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
    emit('status', {'message': 'connected'})


@socketio.on('get_nearby_aircraft')
def handle_aircraft(data):
    try:
        from API import AircraftTracker
        lat, lon = data.get('latitude'), data.get('longitude')
        if lat is None or lon is None:
            emit('nearby_aircraft', {'aircraft': []})
            return
        
        tracker = AircraftTracker()
        result = tracker.get_aircraft(lat, lon, data.get('radius', 50))
        
        if result and 'ac' in result:
            aircraft = [{
                'callsign': str(ac.get('call', 'N/A')).strip(),
                'registration': ac.get('r', 'N/A'),
                'type': ac.get('t', 'N/A'),
                'lat': ac.get('lat'),
                'lon': ac.get('lon'),
                'altitude': ac.get('alt_baro', 0),
                'speed': ac.get('gs', 0),
                'track': ac.get('track', 0)
            } for ac in result['ac']]
            emit('nearby_aircraft', {'aircraft': aircraft})
        else:
            emit('nearby_aircraft', {'aircraft': []})
    except Exception as e:
        emit('nearby_aircraft', {'aircraft': [], 'error': str(e)})


@socketio.on('process_frame')
def handle_frame(data):
    """Minimal frame processing - matches Camera module.py logic"""
    try:
        if 'frame' not in data:
            return
        
        # Decode
        parts = data['frame'].split(',')
        if len(parts) < 2:
            return
        
        frame_bytes = base64.b64decode(parts[1])
        frame = cv2.imdecode(np.frombuffer(frame_bytes, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return
        
        # Sky check
        sky_ratio = estimate_sky_ratio(frame)
        sky_ok = sky_ratio >= SKY_RATIO_THRESHOLD
        
        # Annotate sky warning (same as Camera module.py)
        if not sky_ok:
            cv2.putText(frame, "Please point the camera to the SKY", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Sky coverage: {sky_ratio*100:.1f}%", (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1, cv2.LINE_AA)
        
        # YOLO detection (exact Camera module.py logic)
        detections = []
        try:
            model = get_model()
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model.predict(source=rgb, verbose=False, conf=CONF_THRESHOLD, imgsz=640, max_det=30)
            
            if results and len(results) > 0:
                boxes = results[0].boxes
                if boxes is not None:
                    for box in boxes:
                        conf = float(box.conf.cpu().numpy())
                        cls_idx = int(box.cls.cpu().numpy())
                        cls_name = model.names.get(cls_idx, str(cls_idx)).lower()
                        
                        # Filter by PREFERRED_CLASSES (exactly as Camera module.py)
                        if not any(p in cls_name for p in PREFERRED_CLASSES):
                            continue
                        
                        xyxy = box.xyxy[0].cpu().numpy()
                        draw_detection(frame, xyxy, conf, cls_name)
                        
                        detections.append({
                            'class': cls_name,
                            'confidence': conf,
                            'threat': 'high' if 'drone' in cls_name else 'medium'
                        })
        except Exception:
            pass  # Silent fail for YOLO errors
        
        # Encode and send
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        b64 = base64.b64encode(buffer).decode('utf-8')
        
        emit('annotated_frame', {
            'frame': f'data:image/jpeg;base64,{b64}',
            'sky_ok': sky_ok,
            'detections': detections
        })
        
    except Exception:
        pass  # Silent fail


@socketio.on('disconnect')
def on_disconnect():
    pass


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"\n=== AERIES Backend ===\nhttp://localhost:{port}\n")
    socketio.run(app, host='0.0.0.0', port=port, debug=False, allow_unsafe_werkzeug=True)
