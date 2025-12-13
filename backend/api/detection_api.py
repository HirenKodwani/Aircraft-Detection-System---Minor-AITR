"""
Detection API - REST endpoints for detection results (for browser access)
"""
from flask import Blueprint, jsonify, request, send_file
from datetime import datetime, timedelta
from pathlib import Path

from backend.database.db_manager import get_db_manager
from backend.utils.logger import setup_logger

logger = setup_logger('detection_api')

# Create Blueprint
detection_bp = Blueprint('detection_api', __name__, url_prefix='/api/detections')


@detection_bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'detection_api',
        'timestamp': datetime.now().isoformat()
    })


@detection_bp.route('/', methods=['GET'])
def get_all_detections():
    """
    Get all detections with pagination
    Query params:
      - limit: Number of results (default 100)
      - offset: Pagination offset (default 0)
      - session_id: Filter by session ID
    """
    try:
        limit = min(int(request.args.get('limit', 100)), 1000)
        offset = int(request.args.get('offset', 0))
        session_id = request.args.get('session_id')
        
        db = get_db_manager()
        detections = db.get_detections(session_id=session_id, limit=limit)
        
        # Apply offset
        detections = detections[offset:offset + limit]
        
        return jsonify({
            'status': 'success',
            'count': len(detections),
            'detections': detections,
            'limit': limit,
            'offset': offset
        })
        
    except Exception as e:
        logger.error(f"Error fetching detections: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@detection_bp.route('/latest', methods=['GET'])
def get_latest_detections():
    """Get latest detections (last 10)"""
    try:
        db = get_db_manager()
        detections = db.get_detections(limit=10)
        
        return jsonify({
            'status': 'success',
            'count': len(detections),
            'detections': detections
        })
        
    except Exception as e:
        logger.error(f"Error fetching latest detections: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@detection_bp.route('/<int:detection_id>', methods=['GET'])
def get_detection(detection_id):
    """Get specific detection by ID"""
    try:
        db = get_db_manager()
        detections = db.get_detections(limit=10000)  # Get all
        
        # Find specific detection
        detection = next((d for d in detections if d['detection_id'] == detection_id), None)
        
        if detection:
            return jsonify({
                'status': 'success',
                'detection': detection
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Detection not found'
            }), 404
            
    except Exception as e:
        logger.error(f"Error fetching detection: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@detection_bp.route('/stats', methods=['GET'])
def get_detection_stats():
    """Get detection statistics"""
    try:
        db = get_db_manager()
        all_detections = db.get_detections(limit=10000)
        
        # Calculate stats
        total_count = len(all_detections)
        
        # Count by type
        type_counts = {}
        for det in all_detections:
            obj_type = det.get('object_type', 'unknown')
            type_counts[obj_type] = type_counts.get(obj_type, 0) + 1
        
        # Recent detections (last hour)
        one_hour_ago = datetime.now() - timedelta(hours=1)
        recent_count = sum(
            1 for d in all_detections 
            if d.get('timestamp') and datetime.fromisoformat(d['timestamp']) > one_hour_ago
        )
        
        # Average confidence
        confidences = [d['confidence'] for d in all_detections if d.get('confidence')]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        return jsonify({
            'status': 'success',
            'stats': {
                'total_detections': total_count,
                'recent_detections_1h': recent_count,
                'by_type': type_counts,
                'average_confidence': round(avg_confidence, 3)
            }
        })
        
    except Exception as e:
        logger.error(f"Error calculating stats: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@detection_bp.route('/image/<int:detection_id>', methods=['GET'])
def get_detection_image(detection_id):
    """Get detection image"""
    try:
        db = get_db_manager()
        detections = db.get_detections(limit=10000)
        detection = next((d for d in detections if d['detection_id'] == detection_id), None)
        
        if not detection or not detection.get('image_path'):
            return jsonify({'status': 'error', 'message': 'Image not found'}), 404
        
        image_path = Path(detection['image_path'])
        if not image_path.exists():
            return jsonify({'status': 'error', 'message': 'Image file not found'}), 404
        
        return send_file(str(image_path), mimetype='image/jpeg')
        
    except Exception as e:
        logger.error(f"Error getting image: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@detection_bp.route('/session/<int:session_id>', methods=['GET'])
def get_session_detections(session_id):
    """Get all detections for a specific session"""
    try:
        db = get_db_manager()
        detections = db.get_detections(session_id=session_id)
        
        return jsonify({
            'status': 'success',
            'session_id': session_id,
            'count': len(detections),
            'detections': detections
        })
        
    except Exception as e:
        logger.error(f"Error fetching session detections: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@detection_bp.route('/export', methods=['GET'])
def export_detections():
    """Export detections as JSON"""
    try:
        session_id = request.args.get('session_id')
        
        db = get_db_manager()
        detections = db.get_detections(session_id=session_id, limit=10000)
        
        return jsonify({
            'status': 'success',
            'export_date': datetime.now().isoformat(),
            'count': len(detections),
            'detections': detections
        })
        
    except Exception as e:
        logger.error(f"Error exporting detections: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500
