import streamlit as st
import cv2
import numpy as np
import av
import threading
import time
import os
import tempfile
import atexit
import shutil
import queue

from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from aiortc.contrib.media import MediaPlayer

# Import Custom Modules
import config
# --- GLOBAL CONFIG ---
RTC_CONFIG = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

# --- CLEANUP HANDLER ---
from utils.detector_yolo import YOLODetector
from utils.tracker_deepsort import DeepSortTracker
from utils.stabilizer import VideoStabilizer
from utils.speed_estimator import SpeedEstimator, HomographySpeedEstimator
from utils.reporting import Reporter
from utils.visualizer import Visualizer

# --- CLEANUP HANDLER ---
def cleanup_snapshots():
    snapshot_dir = config.SNAPSHOT_DIR
    if os.path.exists(snapshot_dir):
        try:
            for filename in os.listdir(snapshot_dir):
                file_path = os.path.join(snapshot_dir, filename)
                try:
                    if os.path.isfile(file_path): os.unlink(file_path)
                except: pass
            print("🧹 Cleanup: Overspeeding directory cleared.")
        except: pass
atexit.register(cleanup_snapshots)

# --- PAGE CONFIG ---
st.set_page_config(page_title="LVISS Command Center", layout="wide", page_icon="🚔")


# Initialize Session State for Theme (needed early for CSS)
if "is_dark" not in st.session_state:
    st.session_state.is_dark = True # Default to Dark

# Determine theme colors based on state
if st.session_state.is_dark:
    theme_vars = """
    --bg-color: #0e1117;
    --text-color: #e5e7eb;
    --input-bg: #1f2937;
    --input-text: #ffffff;
    --sidebar-bg: #111827;
    --border-color: #374151;
    """
else:
    theme_vars = """
    --bg-color: #f0f2f6;
    --text-color: #111827;
    --input-bg: #ffffff;
    --input-text: #111827;
    --sidebar-bg: #ffffff;
    --border-color: #cbd5e0;
    """

# Inject CSS Variables and Global Overrides
st.markdown(f"""
<style>
:root {{
    {theme_vars}
}}

/* Apply variables to main elements */
html, body, .stApp {{
    background-color: var(--bg-color) !important; 
    color: var(--text-color) !important;
}}

section[data-testid="stSidebar"] {{
    background-color: var(--sidebar-bg) !important;
}}

/* Universal Text Override using Variables */
p, span, label, h1, h2, h3, h4, h5, h6, div, li, a {{
    color: var(--text-color) !important;
}}

/* Input Fields & Text Areas */
input[type="text"], input[type="password"], textarea, select {{
    background-color: var(--input-bg) !important;
    color: var(--input-text) !important;
    border: 1px solid var(--border-color) !important;
    caret-color: var(--text-color) !important;
}}

/* Fix Placeholder Text visibility */
::placeholder {{
    color: var(--text-color) !important;
    opacity: 0.7;
}}

/* Fix Dropdown/Selectbox Menu Items */
div[role="listbox"] div {{
    background-color: var(--input-bg) !important;
    color: var(--text-color) !important;
}}

/* Fix Streamlit Markdown & Labels */
[data-testid="stMarkdownContainer"] p, 
[data-testid="stWidgetLabel"] label,
span[data-testid="stMetricValue"] {{
    color: var(--text-color) !important;
}}

[data-testid="stAppViewContainer"] {{
    background-color: var(--bg-color) !important;
}}
[data-testid="stHeader"] {{
    background-color: transparent !important;
}}
</style>
""", unsafe_allow_html=True)

# Load Unified CSS (we will update style.css to be generic if needed, but the vars above handle most criticals)
# We can stop switching files and just use style.css + vars
try:
    with open("assets/style.css") as f: st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
except: pass





# --- SHARED STATE FOR SIDEBAR UPDATES ---
# Since WebRTC runs in a separate thread, we need a thread-safe way to get the latest vehicle list.
# We will use a dedicated class attribute + method on the processor.

# --- PROCESSOR (UNIFIED) ---
class LVISSProcessor:
    def __init__(self):
        # AI Init (Run once per stream start)
        self.detector = YOLODetector(config.YOLO_MODEL)
        self.tracker = DeepSortTracker(max_age=30)
        
            
        self.stabilizer = VideoStabilizer()
        self.speed_estimator = HomographySpeedEstimator(fps=config.FPS, buffer_size=8, src_points=config.SOURCE_POINTS, dst_points=config.DEST_POINTS)
        self.reporter = Reporter(log_file=config.LOG_FILE, snapshot_dir=config.SNAPSHOT_DIR)
        self.visualizer = Visualizer()
        
        # State shared with UI
        self.latest_sidebar_data = []
        self.lock = threading.Lock()
        
        # Runtime Config
        self.night_mode = False 
        self.show_roi = True
        self.speed_limit = config.SPEED_LIMIT # Initialize with default

    def update_config(self, night_mode, show_roi, speed_limit):
        self.night_mode = night_mode
        self.show_roi = show_roi
        self.speed_limit = speed_limit

    def get_sidebar_data(self):
        with self.lock:
            return self.latest_sidebar_data

    def adjust_gamma(self, image, gamma=1.0):
        # Build a lookup table mapping the pixel values [0, 255] to their adjusted gamma values
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)
    
    def process_frame(self, frame):
        """Core AI logic, extracted to support both WebRTC and Batch Processing."""
        # 0. Optimization: Resize Input
        # We resize for processing AND display speed.
        h, w = frame.shape[:2]
        if w > 640:
            scale = 640 / w
            frame = cv2.resize(frame, (640, int(h * scale)))

        # 0.1 Night Mode Pre-processing
        if self.night_mode:
            # Boost Gamma to reveal details in low light
            frame = self.adjust_gamma(frame, gamma=1.5)

        # 1. Detection
        # Lower threshold at night to catch faint objects
        conf = 0.25 if self.night_mode else 0.45
        detections = self.detector.detect(frame, conf_threshold=conf) 
        
        # 2. Tracking
        objects = self.tracker.update(detections, frame)

        # 3. Logic
        current_sidebar_items = []
        
        # Mapping class names to emojis
        class_emojis = {
            "car": "🚗",
            "motorcycle": "🏍️",
            "bus": "🚌",
            "truck": "🚛",
            "vehicle": "🚗"
        }
        
        for (objectID, track_info) in objects.items():
            centroid, cls_name, bbox = track_info
            speed = self.speed_estimator.estimate_speed(objectID, centroid)
            
            # Determine icon based on speed and class
            base_icon = class_emojis.get(cls_name.lower(), "🚗")
            icon = base_icon
            if speed > self.speed_limit: icon = "🚨"
            
            # Identify vehicle type along with ID and speed
            display_name = cls_name.capitalize()
            current_sidebar_items.append(f"{icon} **{display_name} {objectID}**: `{speed} km/h`")
            
            is_overspeed = speed > self.speed_limit
            color = self.visualizer.COLOR_DANGER if is_overspeed else self.visualizer.COLOR_NORMAL
            
            # Include vehicle type in overlay label
            label = f"{display_name} {objectID} | {speed} km/h"
            
            if is_overspeed:
                self.reporter.log_vehicle(objectID, speed, is_overspeed, frame, bbox, vehicle_type=cls_name)

            frame = self.visualizer.draw_corner_rect(frame, bbox, color, length=15)
            frame = self.visualizer.draw_info_card(frame, bbox, label, color)
        
        if self.show_roi:
            frame = self.visualizer.draw_roi(frame, config.SOURCE_POINTS)

        # Update Sidebar Data (Thread Safe)
        with self.lock:
            self.latest_sidebar_data = current_sidebar_items
        
        return frame

    def recv(self, frame_av):
        try:
            frame = frame_av.to_ndarray(format="bgr24")
            # Call extracted logic
            frame = self.process_frame(frame)
            return av.VideoFrame.from_ndarray(frame, format="bgr24")
        except Exception as e:
            import traceback
            traceback.print_exc()
            return frame_av # Return original if crash happens to keep stream alive


def main():
    # --- AUTHENTICATION (HASHED) ---
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    # Initialize Analysis State
    if "last_uploaded" not in st.session_state:
        st.session_state.last_uploaded = None
    if "processed_video_path" not in st.session_state:
        st.session_state.processed_video_path = None
    if "analysis_active" not in st.session_state:
        st.session_state.analysis_active = False

    if not st.session_state.authenticated:
        # Hide Sidebar by not rendering anything in it
        st.markdown("""
        <style>
        section[data-testid="stSidebar"] { display: none !important; }
        </style>
        """, unsafe_allow_html=True)
        
        st.title("🔒 LVISS Secure Login")
        user = st.text_input("Username")
        pwd = st.text_input("Password", type="password")
        if st.button("Login"):
            # Check Hash instead of Plain Text
            user_hash = config.hash_password(user)
            pwd_hash = config.hash_password(pwd)
            
            if user_hash == config.ADMIN_USER_HASH and pwd_hash == config.ADMIN_PASS_HASH:
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Access Denied.")
        return # Stop execution until authed

    # --- SIDEBAR RENDER (Authenticated Only) ---
    
    # Reverted to Native Toggle for Stability
    st.sidebar.markdown("### Settings")
    
    # Native Toggle
    is_dark = st.sidebar.toggle("🌙 Dark Mode", value=st.session_state.is_dark)
    
    # Update state if changed
    if is_dark != st.session_state.is_dark:
        st.session_state.is_dark = is_dark
        st.rerun()

    st.sidebar.markdown("---")
    
    # Sidebar Controls
    # Determine local variables based on global state
    night_mode = st.session_state.is_dark 
    show_roi = st.sidebar.checkbox("Show ROI (Blue Line)", value=True)
    speed_limit = st.sidebar.slider("Speed Limit (km/h)", min_value=10, max_value=150, value=config.SPEED_LIMIT)
    
    st.sidebar.title("🔧 Control Panel")
    
    st.title("🚔 LVISS COMMAND CENTER")

    c1, c2, c3 = st.columns(3)
    c1.metric("System Status", "ARMED", "Online")
    c2.metric("Speed Limit", f"{speed_limit} km/h")
    c3.metric("Video Engine", "Optimized", "60 FPS")
    
    tab_live, tab_review = st.tabs(["🌍 LIVE OPERATIONS", "📂 EVIDENCE VAULT"])

    with tab_live:
        # --- MODE SWITCH RESET ---
        if "last_mode" not in st.session_state:
            st.session_state.last_mode = None
            
        input_mode = st.sidebar.radio("Video Source", ["Webcam", "Upload Video"])
        
        # Detect mode switch to clear analysis state
        if st.session_state.last_mode != input_mode:
            st.session_state.analysis_active = False
            st.session_state.last_mode = input_mode
            # No st.rerun() here! It causes double-reload slowness.
            # The logic below will automatically pick up the new mode.
            
        st.sidebar.divider()
        sidebar_placeholder = st.sidebar.empty()
        
        ctx = None
        
        # Direct render (No container wrapper causing ghosting issues)
        if input_mode == "Webcam":
            # import pdb; pdb.set_trace()
            st.info("📡 Streaming Signal: LIVE")
            ctx = webrtc_streamer(
                key="cam-stream",
                mode=WebRtcMode.SENDRECV,
                rtc_configuration=RTC_CONFIG,
                video_processor_factory=LVISSProcessor,
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True,
            )
            if ctx.video_processor:
                ctx.video_processor.update_config(night_mode, show_roi, speed_limit)

            # --- LIVE SIDEBAR STATS LOOP FOR WEBCAM ---
            if ctx.video_processor:
                placeholder_text = st.sidebar.empty()
                while ctx.state.playing:
                    # Update config in real-time
                    if ctx.video_processor:
                        ctx.video_processor.update_config(night_mode, show_roi, speed_limit)
                        data = ctx.video_processor.get_sidebar_data()
                        if data:
                            placeholder_text.markdown("### 🚙 Live Vehicles\n" + "\n".join(data))
                        else:
                            placeholder_text.markdown("### 🚙 Live Vehicles\n*Scanning...*")
                    time.sleep(0.1) 
        
        elif input_mode == "Upload Video":
            st.info("🎞️ File Analysis: BATCH PROCESS")
            uploaded_file = st.file_uploader("Upload MP4/AVI", type=['mp4', 'avi', 'mov'])
            
            # Use Session State to manage the processed video file path
            if "processed_video_path" not in st.session_state:
                st.session_state.processed_video_path = None
            
            # Reset if file changes
            if uploaded_file and st.session_state.last_uploaded != uploaded_file.name:
                st.session_state.analysis_active = False
                st.session_state.processed_video_path = None
                st.session_state.last_uploaded = uploaded_file.name

            if uploaded_file is not None:
                # 1. Processing Phase
                if st.session_state.processed_video_path is None:
                    if st.button("🚀 Start Scan & Analysis"):
                        # Live Analysis Preview Container
                        st.markdown("### ⚡ Live Analysis Feed")
                        preview_placeholder = st.empty()
                        
                        progress_bar = st.progress(0, text="Initializing Engine...")
                        
                        # Save uploaded file
                        uploaded_file.seek(0)
                        
                        # Use project directory for temporary files to avoid cross-drive/temp-cleanup issues
                        temp_dir = os.path.join(os.getcwd(), "temp_analysis")
                        os.makedirs(temp_dir, exist_ok=True)
                        
                        # Create a unique filename for this upload
                        import uuid
                        unique_id = str(uuid.uuid4())[:8]
                        input_name = f"input_{unique_id}.mp4"
                        output_name = f"output_{unique_id}.mp4"
                        
                        tfile_path = os.path.join(temp_dir, input_name)
                        output_path = os.path.join(temp_dir, output_name)
                        
                        with open(tfile_path, "wb") as f:
                            f.write(uploaded_file.read())

                        # Open Video
                        cap = cv2.VideoCapture(tfile_path)
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        fps = int(cap.get(cv2.CAP_PROP_FPS))
                        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        
                        # Output Config (Using av / h264 for browser compatibility)
                        # We use a simple MP4V codec as fallback or H264 if available via ffmpeg wrap
                        # For Streamlit, re-encoding to H264 is crucial.
                        # Since OpenCV encoding can be tricky, we'll try 'mp4v'.
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        # Note: We hardcode output size to 640 width because processor resizes frame!
                        # We need to check frame size AFTER processor.
                        
                        # Instantiate Processor
                        processor = LVISSProcessor()
                        processor.update_config(night_mode, show_roi, speed_limit)

                        out = None

                        frame_count = 0
                        last_u_time = 0
                        
                        # Add a sidebar placeholder for live vehicle stats during batch processing
                        st.sidebar.divider()
                        batch_sidebar_placeholder = st.sidebar.empty()
                        
                        while cap.isOpened():
                            ret, frame = cap.read()
                            if not ret: break
                            
                            # Update config on the fly if needed (though batch is usually static settings)
                            processor.update_config(night_mode, show_roi, speed_limit)
                            
                            # Process
                            processed_frame = processor.process_frame(frame)
                            
                            # Live Preview Update (Optimized: Max 5 FPS, Resized)
                            now = time.time()
                            if now - last_u_time > 0.2:
                                # Resize for UI performance (reduce bandwidth)
                                h_p, w_p = processed_frame.shape[:2]
                                scale_p = 480 / w_p
                                preview_small = cv2.resize(processed_frame, (480, int(h_p * scale_p)))
                                preview_placeholder.image(preview_small, channels="BGR", use_container_width=True)
                                
                                # Update Sidebar with live vehicle identification
                                sidebar_data = processor.get_sidebar_data()
                                if sidebar_data:
                                    batch_sidebar_placeholder.markdown("### 🚙 Detected Vehicles\n" + "\n".join(sidebar_data))
                                else:
                                    batch_sidebar_placeholder.markdown("### 🚙 Detected Vehicles\n*Scanning...*")
                                    
                                last_u_time = now

                            # Init Writer once we know size
                            if out is None:
                                h_out, w_out = processed_frame.shape[:2]
                                out = cv2.VideoWriter(output_path, fourcc, fps, (w_out, h_out))
                            
                            out.write(processed_frame)
                            
                            frame_count += 1
                            if total_frames > 0:
                                progress = min(frame_count / total_frames, 1.0)
                                progress_bar.progress(progress, text=f"Scanning Frame {frame_count}/{total_frames}")

                        cap.release()
                        if out: out.release()
                        
                        # Cleanup input temp file but keep output for session
                        try: os.remove(tfile_path)
                        except: pass
                        
                        st.session_state.processed_video_path = output_path
                        st.success("✅ Analysis Complete!")
                        st.rerun()
                
                # 2. Playback Phase (Native Controls)
                if st.session_state.processed_video_path:
                    st.divider()
                    st.markdown("### 🎬 Analysis Results")
                    
                    # Fix: Read file as bytes to avoid MediaFileStorageError with temp paths
                    try:
                        with open(st.session_state.processed_video_path, "rb") as f:
                            video_bytes = f.read()
                        st.video(video_bytes)
                    except Exception as e:
                        st.error(f"Error loading video: {e}")
                    
                    c_reset, _ = st.columns(2)
                    if c_reset.button("🔄 Analyze New Video"):
                        st.session_state.processed_video_path = None
                        st.session_state.last_uploaded = None
                        st.rerun()

    with tab_review:
        st.markdown("### 🚨 Evidence Vault")
        c_ref, c_clear = st.columns([1, 4])
        if c_ref.button("🔄 Refresh Gallery"): st.rerun()
        if c_clear.button("🗑️ Start Fresh (Delete All)"):
            cleanup_snapshots()
            st.rerun()
        
        import glob
        img_dir = config.SNAPSHOT_DIR
        # Only show valid images
        images = [f for f in glob.glob(os.path.join(img_dir, "*.jpg")) if os.path.getsize(f) > 0]
        images.sort(key=os.path.getmtime, reverse=True)
        
        if images:
            cols = st.columns(4)
            for i, img_path in enumerate(images):
                with cols[i % 4]:
                    # Fix Windows File Lock: Read bytes, don't keep handle open!
                    try:
                        with open(img_path, "rb") as f:
                            img_bytes = f.read()
                        st.image(img_bytes, use_container_width=True)
                    except:
                        st.error("Error loading image")
                        continue

                    fname = os.path.basename(img_path)
                    if st.button("🗑️ Delete", key=f"del_{fname}_{i}"):
                        try: 
                            os.remove(img_path)
                            st.toast(f"Deleted {fname}")
                            time.sleep(0.5)
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error: {e}")
        else:
            st.info("No evidence found in vault.")

if __name__ == "__main__":
    main()
