import streamlit as st
import cv2
import numpy as np
from shapely.geometry import Point, Polygon
from ultralytics import YOLO
from PIL import Image
import os
from datetime import datetime

# Load your model
model_path = os.path.join(os.path.dirname(__file__), "yolo11n.pt")
model = YOLO(model_path)  # 🔁 Replace with your custom model if needed

# Directories
os.makedirs("output/violations", exist_ok=True)

# Page setup
st.set_page_config(layout="wide")
st.title("🚧 Safety Zone Monitor with YOLO & HSV")

# ------------------------------
# Green Zone Detection Function
# ------------------------------
def detect_green_zone(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest = max(contours, key=cv2.contourArea)
        approx = cv2.approxPolyDP(largest, 5, True)
        pts = [tuple(pt[0]) for pt in approx]
        return Polygon(pts), np.array(pts)
    return None, None

# ------------------------------
# Person Detection & Violation Check
# ------------------------------
def process_frame(frame):
    orig_frame = frame.copy()
    zone_polygon, zone_pts = detect_green_zone(frame)

    if zone_pts is not None:
        cv2.polylines(frame, [zone_pts], isClosed=True, color=(0, 255, 0), thickness=2)

    results = model.predict(orig_frame, verbose=False)[0]
    boxes = results.boxes

    violation = False
    for box in boxes:
        cls = int(box.cls[0])
        if cls != 0:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx, cy = (x1 + x2) // 2, y2

        if zone_polygon and not zone_polygon.contains(Point(cx, cy)):
            color = (0, 0, 255)
            violation = True
        else:
            color = (0, 255, 0)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.circle(frame, (cx, cy), 5, color, -1)

    if violation:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        cv2.imwrite(f"output/violations/frame_{timestamp}.jpg", frame)

    return frame

# ===================================
# REGION 1: Main Video Upload & Run
# ===================================
st.header("📹 Main Video Processing")

uploaded_file = st.file_uploader("Upload factory floor video", type=["mp4", "mov", "avi"])
if uploaded_file:
    tfile = open("input.mp4", 'wb')
    tfile.write(uploaded_file.read())

    cap = cv2.VideoCapture("input.mp4")
    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        result = process_frame(frame)
        stframe.image(result, channels="BGR", use_container_width=True)

    cap.release()
    st.success("✅ Video Processing Completed.")

# ===================================
# REGION 1.5: Explanation Section
# ===================================
st.markdown("---")
st.subheader("🦺 What We Do")
st.markdown("""
Our system ensures industrial safety through intelligent video processing:

- 📸 **Capture Frames**: The webcam or video feed is read frame-by-frame in real-time.
- 🟩 **Detect Safety Zone**: Green-colored safety regions are identified using HSV color thresholding.
- 🧍 **Detect Workers**: A custom-trained YOLO model detects factory workers in each frame.
- 📍 **Track Position**: Each worker’s bounding box center is compared against the safety zone.
- ⚠️ **Flag Violations**: If a worker is found outside the safety area, a violation is recorded and the annotated frames saved for compliance checks.
""")

# ===================================
# REGION 2: Demo (Image Only)
# ===================================
st.markdown("---")
st.header("🧪 Demo Image Region (Try it!)")

col1, col2 = st.columns(2)

# LEFT COLUMN: User selects or uses default image
with col1:
    st.subheader("🔍 Input Image")
    demo_file = st.file_uploader("Use your own demo image (optional)", type=["jpg", "jpeg", "png"], key="demo_img")

    if demo_file:
        demo_image = Image.open(demo_file)
    else:
        # ⬇️ Replace this with your default image path (must exist on local or host)
        img_path = os.path.join(os.path.dirname(__file__), "demo.png")
        demo_image = Image.open(img_path)
        st.caption("Default image loaded (replace 'demo.png')")

    st.image(demo_image, caption="Original Demo Image", use_container_width=True)

# RIGHT COLUMN: Processed image output
with col2:
    st.subheader("⚙️ Processed Result")
    if st.button("Run Demo", key="run_demo_btn"):
        if demo_image.mode != "RGB":
            demo_image = demo_image.convert("RGB")
        img_np = np.array(demo_image)
        frame = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        result = process_frame(frame)
        st.image(result, channels="BGR", caption="Processed Demo Image", use_container_width=True)
