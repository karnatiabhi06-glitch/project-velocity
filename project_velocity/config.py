import os
import hashlib

# --- SECURITY ---
# Default to "admin" / "admin" by hashing the defaults
# In production, these should be loaded from os.environ
DEFAULT_USER = "admin"
DEFAULT_PASS = "admin"

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Store HASHED credentials, never plain text
ADMIN_USER_HASH = hash_password(os.environ.get("LVISS_USER", DEFAULT_USER))
ADMIN_PASS_HASH = hash_password(os.environ.get("LVISS_PASS", DEFAULT_PASS))

# --- PATHS ---
YOLO_MODEL = 'yolov8n.pt'
LOG_FILE = 'traffic_log.csv'
SNAPSHOT_DIR = 'overspeeding'
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

# --- CAMERA & VIDEO ---
FPS = 30
SKIP_FRAMES = 1

# --- DETECTION ---
CONFIDENCE_THRESHOLD = 0.4
NMS_THRESHOLD = 0.4

# --- SPEED ESTIMATION ---
SPEED_LIMIT = 30

# HOMOGRAPHY POINTS (Calibrated)
# Order: Top-Left, Top-Right, Bottom-Right, Bottom-Left
SOURCE_POINTS = [(373, 160), (889, 161), (1055, 663), (273, 672)]

# Real-world distance (meters)
# Width of road ~ 8 meters, Length of segment ~ 20 meters
# Adjusted to 30 meters to fix "triple digit" speed errors
DEST_POINTS = [(0, 0), (80, 0), (80, 30), (0, 30)]

# --- VISUALIZATION ---
SHOW_ROI = True
