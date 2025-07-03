import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from ultralytics import YOLO
import cv2
import numpy as np
import os
import time
from datetime import datetime

# -------------------- Streamlit UI --------------------
st.set_page_config(page_title="Harness Detection", page_icon="🧰", layout="wide")
st.title("🧰 Real-Time Harness Compliance Monitoring")
st.caption("Detects workers without safety harness using YOLOv8")

# -------------------- Load Models --------------------
@st.cache_resource
def load_models():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    person_model_path = os.path.join(script_dir, "yolo11n.pt")
    harness_model_path = os.path.join(script_dir, "best.pt")

    person_model = YOLO(person_model_path)
    harness_model = YOLO(harness_model_path)
    return person_model, harness_model

person_model, harness_model = load_models()

# -------------------- Save Dir --------------------
output_dir = "flagged_frames"
os.makedirs(output_dir, exist_ok=True)

# -------------------- Video Transformer --------------------
class HarnessComplianceTransformer(VideoTransformerBase):
    def __init__(self):
        self.frame_idx = 0

    def transform(self, frame):
        self.frame_idx += 1
        img = frame.to_ndarray(format="bgr24")
        flagged = False
        annotated_frame = img.copy()

        # === Detect Persons ===
        results = person_model(img, classes=[0], conf=0.4)[0]
        person_boxes = results.boxes.xyxy.cpu().numpy().astype(int)

        for box in person_boxes:
            x1, y1, x2, y2 = box
            person_crop = img[y1:y2, x1:x2]

            # === Harness Detection in Person Crop ===
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

        # === Save Frame If Flagged ===
        if flagged:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(output_dir, f"flagged_{timestamp}.jpg")
            cv2.imwrite(filepath, annotated_frame)
            print(f"[!] Flagged: {filepath}")

        return annotated_frame

# -------------------- Webcam Stream --------------------
st.header("📸 Live Webcam Feed")
ctx = webrtc_streamer(
    key="harness-detection",
    video_processor_factory=HarnessComplianceTransformer,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True
)

st.markdown("---")
st.markdown(
    "<center><sub>© 2025 TATAVision Secure | Harness Monitoring with YOLOv8</sub></center>",
    unsafe_allow_html=True
)
