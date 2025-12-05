# Aircraft Detection System ✈️

Real-time aircraft detection using YOLOv8 + Browser camera with FlightRadar-style map.

## Features
- 🎥 Live camera detection with bounding boxes
- 🗺️ Interactive map with GPS location
- ✈️ Real-time aircraft tracking (ADSB.lol API)
- 🔍 Threat classification & verification
- 📊 Detection logging to database

## Quick Start (Local)
```bash
pip install -r requirements.txt
python demo_backend.py
# Open http://localhost:5000
```

## Deploy to Railway
1. Push to GitHub
2. Go to [railway.app](https://railway.app)
3. New Project → Deploy from GitHub
4. Select this repo
5. Railway auto-detects Python + Procfile
6. Done! 🚀

## Project Structure
```
├── index.html          # Frontend (map + camera)
├── demo_backend.py     # Flask server
├── API.py              # Aircraft tracking API
├── yolov8n.pt          # YOLO model
├── requirements.txt    # Dependencies
├── Procfile            # Railway config
└── backend/            # Modules
```

## Technologies
- Frontend: HTML, Leaflet.js, Socket.IO
- Backend: Flask, Flask-SocketIO
- Detection: YOLOv8, OpenCV
- Aircraft: ADSB.lol API

## License
MIT
