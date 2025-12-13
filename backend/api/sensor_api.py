"""
Sensor API - REST and WebSocket endpoints for mobile sensor data
"""
from flask import Flask, request, jsonify, Response, send_from_directory
from flask_socketio import SocketIO, emit, disconnect
from flask_cors import CORS
from datetime import datetime
import cv2
import base64
import os
from pathlib import Path

from backend.config.config import Config
from backend.modules.sensor_receiver import SensorReceiver
from backend.api.detection_api import detection_bp
from backend.utils.logger import setup_logger

logger = setup_logger('sensor_api')

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = Config.SECRET_KEY
CORS(app)  # Enable CORS for mobile web app and frontend

# Register blueprints
app.register_blueprint(detection_bp)

# Initialize SocketIO
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

# Initialize sensor receiver
sensor_receiver = SensorReceiver()

# Connected clients tracking
connected_clients = {}

# Active detection pipelines
active_pipelines = {}


# REST API Endpoints
@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'connected_clients': len(connected_clients),
        'timestamp': datetime.now().isoformat()
    })


# Serve React frontend
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_frontend(path):
    """Serve React frontend build"""
    frontend_dir = Path(__file__).parent.parent.parent / 'frontend' / 'dist'
    
    # If directory exists (production build)
    if frontend_dir.exists():
        if path and (frontend_dir / path).exists():
            return send_from_directory(str(frontend_dir), path)
        return send_from_directory(str(frontend_dir), 'index.html')
    
    # Development mode - frontend not built yet
    return jsonify({
        'message': 'Frontend not built yet',
        'instructions': [
            '1. cd frontend',
            '2. npm install',
            '3. npm run build',
            '4. Restart backend'
        ],
        'or_use_dev_mode': 'npm run dev (port 3000)'
    })


@app.route('/api/sensors/data', methods=['POST'])
def receive_sensor_data():
    """
    REST endpoint for receiving sensor data
    Use this as fallback if WebSocket is unavailable
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'status': 'error', 'message': 'No data provided'}), 400
        
        # Process sensor data
        result = sensor_receiver.process_sensor_data(data)
        
        if result['status'] == 'error':
            return jsonify(result), 400
        
        # Store in database
        sensor_receiver.store_sensor_data(result['data'])
        
        return jsonify({
            'status': 'success',
            'message': 'Sensor data received',
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Error in REST sensor endpoint: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/devices/active', methods=['GET'])
def get_active_devices():
    """Get list of active devices"""
    devices = sensor_receiver.get_active_devices()
    return jsonify({
        'status': 'success',
        'count': len(devices),
        'devices': devices
    })


# WebSocket Events
@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    client_id = request.sid
    connected_clients[client_id] = {
        'connected_at': datetime.now(),
        'device_id': None
    }
    logger.info(f"Client connected: {client_id}")
    emit('connection_established', {'client_id': client_id, 'timestamp': datetime.now().isoformat()})


@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    client_id = request.sid
    
    if client_id in connected_clients:
        device_id = connected_clients[client_id].get('device_id')
        if device_id:
            sensor_receiver.handle_device_disconnect(device_id)
        
        del connected_clients[client_id]
        logger.info(f"Client disconnected: {client_id}")


@socketio.on('register_device')
def handle_device_registration(data):
    """
    Register a mobile device
    Expected data: {'device_id': 'mobile_xxx', 'device_name': 'My Phone'}
    """
    try:
        client_id = request.sid
        device_id = data.get('device_id')
        device_name = data.get('device_name', 'Unknown Device')
        
        if not device_id:
            emit('error', {'message': 'device_id is required'})
            return
        
        connected_clients[client_id]['device_id'] = device_id
        connected_clients[client_id]['device_name'] = device_name
        
        logger.info(f"Device registered: {device_id} ({device_name})")
        emit('registration_success', {
            'device_id': device_id,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error registering device: {e}")
        emit('error', {'message': str(e)})


@socketio.on('sensor_data')
def handle_sensor_data(data):
    """
    Handle real-time sensor data from mobile device
    Expected data: {
        'device_id': 'mobile_xxx',
        'latitude': 37.7749,
        'longitude': -122.4194,
        'altitude': 15.5,
        'compass_heading': 180.5,
        'accuracy': 10.2,
        'timestamp': '2025-11-28T13:00:00'
    }
    """
    try:
        # Process sensor data
        result = sensor_receiver.process_sensor_data(data)
        
        if result['status'] == 'error':
            emit('error', {'message': result['message']})
            return
        
        # Store in database
        sensor_receiver.store_sensor_data(result['data'])
        
        # Acknowledge receipt
        emit('data_received', {
            'status': 'success',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error processing sensor data: {e}")
        emit('error', {'message': str(e)})


@socketio.on('ping')
def handle_ping():
    """Handle ping for connection keep-alive"""
    emit('pong', {'timestamp': datetime.now().isoformat()})


@socketio.on('request_frame')
def handle_frame_request():
    """Handle request for latest camera frame with detections"""
    try:
        # This will be implemented when frontend is ready
        # For now, acknowledge
        emit('frame_update', {
            'status': 'no_active_session',
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error handling frame request: {e}")
        emit('error', {'message': str(e)})


@socketio.on('threat_alert')
def broadcast_threat_alert(alert_data):
    """Broadcast threat alert to all connected browser clients"""
    try:
        # Broadcast threat alert to all clients
        emit('threat_alert', alert_data, broadcast=True)
        logger.warning(f"Threat alert broadcasted: {alert_data.get('threat_level')}")
    except Exception as e:
        logger.error(f"Error broadcasting threat alert: {e}")


@socketio.on('detection_event')
def broadcast_detection(data):
    """Broadcast detection event to all connected clients"""
    try:
        # Broadcast to all clients
        emit('detection_update', data, broadcast=True)
    except Exception as e:
        logger.error(f"Error broadcasting detection: {e}")


def run_server(host=None, port=None, debug=False):
    """
    Run the sensor API server
    
    Args:
        host: Host address (default from config)
        port: Port number (default from config)
        debug: Debug mode
    """
    host = host or Config.API_HOST
    port = port or Config.API_PORT
    
    logger.info(f"Starting Sensor API server on {host}:{port}")
    socketio.run(app, host=host, port=port, debug=debug)


if __name__ == '__main__':
    run_server(debug=True)
