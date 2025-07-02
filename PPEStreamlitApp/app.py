import streamlit as st
from streamlit_lottie import st_lottie
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import torch
from ultralytics import YOLO
import cv2
import requests
import numpy as np
import json

# Streamlit page config
st.set_page_config(page_title="PPE Detection", page_icon="🦺", layout="wide")

# Load Lottie animation from URL
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_ai = load_lottieurl("https://raw.githubusercontent.com/ybennaceur/streamlit-lottie-animation/main/animations/ai.json")
if lottie_ai:
    st_lottie(lottie_ai, speed=1, height=250, key="ai")

st.title("🦺 Real-Time PPE Detection")
st.caption("Using YOLOv8 + Streamlit for live compliance monitoring")

# Load model
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()
class_names = model.names

REQUIRED_LABELS = {
    "Glasses": "🕶️ Wear safety glasses",
    "Mask": "😷 Wear a mask",
    "Vest": "🦺 Wear a safety vest",
    "Safety Shoes": "🥾 Wear safety shoes",
    "Helmet": "⛑️ Wear a helmet"
}

VIOLATION_LABELS = {
    "Without Glass": "❌ No safety glasses",
    "Without Mask": "❌ No mask",
    "Without Vest": "❌ No safety vest",
    "Without Safety Shoes": "❌ No safety shoes",
    "Without Helmet": "❌ No helmet"
}

CONFIDENCE_THRESHOLD = 0.25

# Custom video processor for webrtc
class PPEVideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.frame_width = 640
        self.frame_height = 480

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_resized = cv2.resize(img, (self.frame_width, self.frame_height))

        results = model(img_resized, conf=CONFIDENCE_THRESHOLD)[0]
        boxes = results.boxes
        detected_classes = [class_names[int(cls)] for cls in boxes.cls]

        annotated = results.plot()  # has bounding boxes

        y = 30
        if "Person" in detected_classes:
            for label, message in REQUIRED_LABELS.items():
                if label not in detected_classes:
                    cv2.putText(annotated, message, (10, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    y += 30
            for label, message in VIOLATION_LABELS.items():
                if label in detected_classes:
                    cv2.putText(annotated, message, (10, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    y += 30

        return annotated

# Stream video from webcam
st.header("📸 Live Detection Feed")
webrtc_streamer(
    key="ppe_stream",
    video_processor_factory=PPEVideoProcessor,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True
)

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #ccc;'>© 2025 | TATAVision Secure | Powered by YOLOv8</div>",
    unsafe_allow_html=True
)
