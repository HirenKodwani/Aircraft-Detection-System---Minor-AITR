-- Aircraft Detection System - Simplified Database Schema (Phase 1)
-- 4 Core Tables for initial data acquisition and detection

-- 1. Devices Table - Track camera and mobile sensor devices
CREATE TABLE devices (
    device_id SERIAL PRIMARY KEY,
    device_name VARCHAR(100) NOT NULL,
    device_type VARCHAR(50) NOT NULL,  -- 'camera', 'mobile_sensor'
    camera_model VARCHAR(100),
    camera_resolution VARCHAR(30),
    has_gps BOOLEAN DEFAULT FALSE,
    has_compass BOOLEAN DEFAULT FALSE,
    installation_location VARCHAR(200),
    installation_latitude DECIMAL(9,6),
    installation_longitude DECIMAL(9,6),
    status VARCHAR(20) DEFAULT 'active',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 2. Detection Sessions Table - Track detection sessions
CREATE TABLE detection_sessions (
    session_id SERIAL PRIMARY KEY,
    device_id INT REFERENCES devices(device_id),
    start_time TIMESTAMP NOT NULL,
    end_time TIMESTAMP,
    location_latitude DECIMAL(9,6),
    location_longitude DECIMAL(9,6),
    session_status VARCHAR(20) DEFAULT 'active',
    total_detections INT DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 3. Geolocation Data Table - GPS + Compass data from mobile device
CREATE TABLE geolocation_data (
    geo_id SERIAL PRIMARY KEY,
    detection_id INT,
    device_id INT REFERENCES devices(device_id),
    device_latitude DECIMAL(9,6) NOT NULL,
    device_longitude DECIMAL(9,6) NOT NULL,
    device_altitude DECIMAL(8,2),
    compass_heading DECIMAL(5,2),  -- 0-360 degrees
    compass_azimuth DECIMAL(5,2),
    estimated_distance_meters DECIMAL(10,2),
    estimated_altitude_meters DECIMAL(8,2),
    estimated_knots DECIMAL(6,2),
    gps_accuracy DECIMAL(5,2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 4. Detected Objects Table - Aircraft/drone detection results
CREATE TABLE detected_objects (
    detection_id SERIAL PRIMARY KEY,
    session_id INT REFERENCES detection_sessions(session_id),
    detection_timestamp TIMESTAMP NOT NULL,
    object_type VARCHAR(50),  -- 'aircraft', 'drone', 'unknown'
    confidence_score DECIMAL(4,3),
    bounding_box_x INT,
    bounding_box_y INT,
    bounding_box_width INT,
    bounding_box_height INT,
    frame_number INT,
    image_path VARCHAR(255),
    detection_status VARCHAR(30) DEFAULT 'pending',
    is_verified BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for performance
CREATE INDEX idx_sessions_device ON detection_sessions(device_id);
CREATE INDEX idx_sessions_status ON detection_sessions(session_status);
CREATE INDEX idx_geo_device ON geolocation_data(device_id);
CREATE INDEX idx_geo_created ON geolocation_data(created_at);
CREATE INDEX idx_detections_session ON detected_objects(session_id);
CREATE INDEX idx_detections_timestamp ON detected_objects(detection_timestamp);
