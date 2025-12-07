# Aircraft Detection System

Real-time aircraft detection using YOLOv8 + Browser camera with FlightRadar-style map.

## Features
- Live camera detection with YOLO bounding boxes
- Interactive map with GPS location
- Real-time aircraft tracking (ADSB.lol API)
- Threat classification & verification
- Detection logging

## Quick Start (Local)
```bash
pip install -r requirements.txt
python demo_backend.py
# Open http://localhost:5000
```

## Deploy to Render (Free)
1. Push to GitHub
2. Go to [render.com](https://render.com)
3. New → Web Service → Connect GitHub
4. Configure:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn -k geventwebsocket.gunicorn.workers.GeventWebSocketWorker -w 1 -b 0.0.0.0:$PORT demo_backend:app`
5. Deploy!

## Project Structure
```
├── index.html          # Frontend
├── demo_backend.py     # Flask server
├── API.py              # Aircraft API
├── yolov8n.pt          # YOLO model
├── requirements.txt    # Dependencies
├── Procfile            # Render/Railway
└── backend/            # Modules
```

## Technologies
- Flask + Flask-SocketIO
- YOLOv8 + OpenCV
- Leaflet.js Maps
- ADSB.lol API
