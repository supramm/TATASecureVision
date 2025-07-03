import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from ultralytics import YOLO
import av
import cv2
import numpy as np
import os
import time
from datetime import datetime

# ----------------- UI Setup -----------------
st.set_page_config(page_title="Kadhai Safety Monitor", page_icon="🫕", layout="wide")
st.title(" 🫕 Kadhai Safety Monitoring System")
st.caption("Monitors if a person leaves the kadhai unattended for more than 5 minutes.")

# ----------------- Load Models -----------------
@st.cache_resource
def load_models():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    kadhai_model_path = os.path.join(script_dir, "best.pt")
    person_model_path = os.path.join(script_dir, "yolov8n.pt")

    kadhai_model = YOLO(kadhai_model_path)
    person_model = YOLO(person_model_path)
    return kadhai_model, person_model

kadhai_model, person_model = load_models()

# ----------------- Video Logic -----------------
TIMER_DURATION = 5 * 60  # 5 minutes
output_dir = "flagged_frames"
os.makedirs(output_dir, exist_ok=True)

class KadhaiSafetyTransformer(VideoTransformerBase):
    def __init__(self):
        self.timer_started = False
        self.start_time = None
        self.warning_triggered = False
        self.last_frame = None
        self.show_alert = False
        self.time_remaining = TIMER_DURATION

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        annotated_frame = img.copy()

        # Detect Kadhai
        kadhai_results = kadhai_model.predict(img, conf=0.5, verbose=False)[0]
        kadhai_boxes = [box for box in kadhai_results.boxes.data]

        # Detect Person
        person_results = person_model.predict(img, conf=0.5, classes=[0], verbose=False)[0]
        person_boxes = [box for box in person_results.boxes.data]

        kadhai_detected = len(kadhai_boxes) > 0
        person_detected = len(person_boxes) > 0

        # Draw bounding boxes
        for box in kadhai_boxes:
            x1, y1, x2, y2, conf, cls = box.cpu().numpy()
            cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 165, 0), 2)
            cv2.putText(annotated_frame, f"Kadhai {conf:.2f}", (int(x1), int(y1)-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,165,0), 2)

        for box in person_boxes:
            x1, y1, x2, y2, conf, cls = box.cpu().numpy()
            cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"Person {conf:.2f}", (int(x1), int(y1)-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        # Safety Logic
        if kadhai_detected:
            if not person_detected:
                if not self.timer_started:
                    self.start_time = time.time()
                    self.timer_started = True
                    self.warning_triggered = False
                else:
                    elapsed = time.time() - self.start_time
                    self.time_remaining = max(0, TIMER_DURATION - int(elapsed))
                    if elapsed >= TIMER_DURATION and not self.warning_triggered:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filepath = os.path.join(output_dir, f"flagged_{timestamp}.jpg")
                        cv2.imwrite(filepath, annotated_frame)
                        self.warning_triggered = True
                        self.show_alert = True
            else:
                self.timer_started = False
                self.start_time = None
                self.warning_triggered = False
                self.show_alert = False
                self.time_remaining = TIMER_DURATION
        else:
            self.timer_started = False
            self.start_time = None
            self.warning_triggered = False
            self.show_alert = False
            self.time_remaining = TIMER_DURATION

        self.last_frame = annotated_frame
        return annotated_frame

# ----------------- Streamlit UI -----------------
st.markdown("### 📻 Live Feed")
st.markdown(
    "If a <b>kadhai</b> is detected <b>without a person</b> for over 5 minutes, an alert will appear and a snapshot will be saved.",
    unsafe_allow_html=True
)

ctx = webrtc_streamer(
    key="kadhai-monitor",
    video_processor_factory=KadhaiSafetyTransformer,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True
)

if ctx.video_transformer:
    t = ctx.video_transformer

    if t.timer_started and not t.warning_triggered:
        mins = t.time_remaining // 60
        secs = t.time_remaining % 60
        st.warning(f"⏳ Timer: {mins:02d}:{secs:02d} - Kadhai is unattended!")

    if t.show_alert:
        st.markdown(
            "<h1 style='color:red; text-align:center;'>🚨 ALERT: KADHAI UNATTENDED FOR OVER 5 MINUTES! 🚨</h1>",
            unsafe_allow_html=True
        )
        st.error("Frame has been saved. Please check immediately!")

st.markdown("---")
st.markdown("<center><sub>© 2025 TATASecureVision</sub></center>", unsafe_allow_html=True)
