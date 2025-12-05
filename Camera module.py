"""
camera_detection.py

Real-time camera module that:
 - captures frames from a webcam
 - does aircraft/UAV detection using YOLO (Ultralytics YOLOv8 recommended)
 - performs a simple sky-detection check and warns if camera is not pointed to sky
 - draws bounding boxes, labels, confidence, and FPS on the feed

Usage:
    python camera_detection.py
Keys:
    q : quit
    s : save current frame to disk (saved in ./captures)
"""

import os
import time
from datetime import datetime
from pathlib import Path

import cv2
import cv2
import numpy as np
import numpy as np

# Try importing YOLO from ultralytics:
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except Exception:
    ULTRALYTICS_AVAILABLE = False

# ----------------------------
# CONFIG
# ----------------------------
CAMERA_INDEX = 0                 # 0 = default webcam; change if you have multiple cameras
MODEL_PATH = "yolov8n.pt"        # recommended small/fast model for prototype
CONF_THRESHOLD = 0.25            # minimum confidence to show detection
CLASS_FILTER = None              # None => show all classes. Optionally set to ['airplane','drone'] if model supports names
SKY_RATIO_THRESHOLD = 0.20       # if proportion of sky pixels < threshold => prompt user to point at sky
SAVE_DIR = Path("captures")
SAVE_DIR.mkdir(exist_ok=True)

# If your YOLO model has different class names you can map them here.
# Ultralytics YOLO models include 'airplane' usually as 'airplane' for aerial datasets,
# but if your model is trained differently, edit accordingly.
PREFERRED_CLASSES = {"airplane", "aeroplane", "drone", "uav"}  # case-insensitive match

# ----------------------------
# SKY DETECTION FUNCTION
# ----------------------------
def estimate_sky_ratio(frame):
    """
    Estimate proportion of frame that looks like sky.
    Heuristic: convert to HSV and count pixels in 'blue-ish' hue range with reasonable brightness.
    Returns ratio between 0 and 1.
    """
    # Resize small for speed
    h, w = frame.shape[:2]
    scale = 320 / w
    small = cv2.resize(frame, (320, int(h * scale)))

    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)

    # Blue-ish color range (tweak if needed)
    lower_blue = np.array([85, 20, 50])   # hue ~85 (blue-cyan)
    upper_blue = np.array([140, 255, 255])# hue ~140 (deep blue)
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    # Also consider high brightness/low saturation (white-ish sky)
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 40, 255])
    mask_white = cv2.inRange(hsv, lower_white, upper_white)

    mask = cv2.bitwise_or(mask_blue, mask_white)
    sky_pixels = cv2.countNonZero(mask)
    total_pixels = mask.shape[0] * mask.shape[1]
    ratio = sky_pixels / total_pixels
    return ratio

# ----------------------------
# UTILS
# ----------------------------
def draw_detection(frame, box, conf, cls_name, color=(0, 255, 0), thickness=2):
    """
    Draw bounding box and label on frame.
    box: [x1, y1, x2, y2]
    """
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    label = f"{cls_name} {conf:.2f}"
    # text background
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
    cv2.putText(frame, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

# ----------------------------
# YOLO WRAPPER
# ----------------------------
class YoloDetector:
    def __init__(self, model_path=MODEL_PATH, conf_thresh=CONF_THRESHOLD):
        if not ULTRALYTICS_AVAILABLE:
            raise RuntimeError("Ultralytics package not available. Install with: pip install ultralytics")
        self.model = YOLO(model_path)
        self.conf_thresh = conf_thresh
        # get names map if available
        self.names = self.model.names if hasattr(self.model, "names") else {}

    def detect(self, frame):
        """
        Run model on a single frame.
        Returns list of detections: each is dict with keys: bbox(x1,y1,x2,y2), conf, cls_name
        """
        # YOLO expects RGB
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # run inference (batch size 1)
        results = self.model.predict(source=img, verbose=False, conf=self.conf_thresh, imgsz=640, max_det=30)
        # results is a list of Results objects (one per image). We used a single image.
        detections = []
        if len(results) == 0:
            return detections
        res = results[0]
        boxes = res.boxes
        if boxes is None:
            return detections
        for box in boxes:
            conf = float(box.conf.cpu().numpy()) if hasattr(box, "conf") else float(box.conf.numpy())
            cls_idx = int(box.cls.cpu().numpy()) if hasattr(box, "cls") else int(box.cls.numpy())
            cls_name = self.names.get(cls_idx, str(cls_idx)).lower()
            xyxy = box.xyxy[0].cpu().numpy() if hasattr(box, "xyxy") else box.xyxy[0].numpy()
            # filter by preferred classes if user specified
            if CLASS_FILTER is not None:
                if cls_name not in CLASS_FILTER:
                    continue
            detections.append({"bbox": xyxy, "conf": conf, "class": cls_name})
        return detections

# ----------------------------
# MAIN
# ----------------------------
def main():
    print("Starting camera module...")
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("ERROR: Unable to open camera index", CAMERA_INDEX)
        return

    # Initialize detector
    if ULTRALYTICS_AVAILABLE:
        print("Loading YOLO model:", MODEL_PATH)
        detector = YoloDetector(MODEL_PATH, CONF_THRESHOLD)
    else:
        print("ERROR: ultralytics not available. Install 'ultralytics' or adapt code to use OpenCV DNN.")
        return

    prev_time = time.time()
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from camera. Exiting.")
            break

        frame_count += 1
        # Optional resize for speed
        display_frame = frame.copy()

        # 1) Sky check (quick)
        sky_ratio = estimate_sky_ratio(display_frame)
        sky_ok = sky_ratio >= SKY_RATIO_THRESHOLD

        if not sky_ok:
            cv2.putText(display_frame, "Please point the camera to the SKY", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(display_frame, f"Sky coverage: {sky_ratio*100:.1f}%", (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1, cv2.LINE_AA)
            # Still proceed with detection, but we warn user

        # 2) Run detection every Nth frame if needed to save compute (optional)
        # For simplicity we run on each frame; for resource-limited devices, run every 2-3 frames.
        detections = detector.detect(display_frame)

        # 3) Process detections: prefer showing airplane/drone classes
        shown_any = False
        for det in detections:
            cls_name = det["class"].lower()
            conf = det["conf"]
            bbox = det["bbox"]  # xyxy
            # Only highlight if class matches preferred aerial classes OR show all
            if PREFERRED_CLASSES:
                if not any(p in cls_name for p in PREFERRED_CLASSES):
                    # skip classes that aren't aircraft/drone (unless you want all)
                    continue
            # At this point, draw
            draw_detection(display_frame, bbox, conf, cls_name, color=(0, 200, 0))
            shown_any = True

        # 4) compute and show FPS
        cur_time = time.time()
        elapsed = cur_time - prev_time
        fps = frame_count / elapsed if elapsed > 0 else 0.0
        cv2.putText(display_frame, f"FPS: {fps:.1f}", (20, display_frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 1, cv2.LINE_AA)

        # 5) Show a small legend for controls
        cv2.putText(display_frame, "Press 'q' to quit | 's' to save frame", (20, display_frame.shape[0] - 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

        # 6) If no sky and no detection, show hint
        if (not sky_ok) and (not shown_any):
            cv2.putText(display_frame, "No aircraft detected - adjust camera", (20, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2, cv2.LINE_AA)

        # Display
        cv2.imshow("Aircraft Detection (YOLO)", display_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            fname = SAVE_DIR / f"capture_{ts}.jpg"
            cv2.imwrite(str(fname), display_frame)
            print("Saved frame to", fname)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
