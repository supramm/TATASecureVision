import streamlit as st
from streamlit_lottie import st_lottie
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from ultralytics import YOLO
import cv2
import requests
import numpy as np
import os
from datetime import datetime

# Page configuration
st.set_page_config(page_title="Harness Detection", page_icon="🧰", layout="wide")

# Lottie animation loader
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Load lottie
lottie_ai = load_lottieurl("https://raw.githubusercontent.com/ybennaceur/streamlit-lottie-animation/main/animations/ai.json")
if lottie_ai:
    st_lottie(lottie_ai, speed=1, height=250, key="ai")

st.title("🧰 Real-Time Harness Compliance Monitoring")
st.caption("Using YOLOv8 + Streamlit for live safety violation detection")

# Load models
@st.cache_resource
def load_models():
    person_model = YOLO("yolo11n.pt")
    harness_model = YOLO("best.pt")
    return person_model, harness_model

person_model, harness_model = load_models()

# Ensure output directory exists
os.makedirs("flagged_frames", exist_ok=True)

# Video processor
class HarnessVideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.frame_idx = 0

    def transform(self, frame):
        self.frame_idx += 1
        img = frame.to_ndarray(format="bgr24")
        annotated_frame = img.copy()
        flagged = False

        # Detect people
        results = person_model(img, classes=[0], conf=0.4)[0]
        person_boxes = results.boxes.xyxy.cpu().numpy().astype(int)

        for box in person_boxes:
            x1, y1, x2, y2 = box
            person_crop = img[y1:y2, x1:x2]
            harness_results = harness_model(person_crop, conf=0.4)[0]
            harness_boxes = harness_results.boxes.xyxy.cpu().numpy().astype(int)

            if len(harness_boxes) == 0:
                flagged = True
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(annotated_frame, "❌ No Harness", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            else:
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated_frame, "✅ Harness", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Save flagged frame
        if flagged:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join("flagged_frames", f"flagged_{timestamp}.jpg")
            cv2.imwrite(path, annotated_frame)

        return annotated_frame

# Webcam stream
st.header("📸 Live Detection Feed")
webrtc_streamer(
    key="harness_stream",
    video_processor_factory=HarnessVideoProcessor,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True
)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #ccc;'>© 2025 | TATAVision Secure | Harness Detection with YOLOv8</div>",
    unsafe_allow_html=True
)
