import streamlit as st
import cv2
import tempfile
import pandas as pd
from ultralytics import YOLO
import os

st.set_page_config(page_title="Demo Analysis", page_icon="🎥", layout="wide")

st.title("🎥 PPE Detection Demo + Analytics")

# Load model (reuse same best.pt)
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "..", "best.pt")
    return YOLO(model_path)

model = load_model()
class_names = model.names

CONFIDENCE_THRESHOLD = 0.25

REQUIRED_LABELS = {
    "Glasses": "Missing Glasses",
    "Mask": "Missing Mask",
    "Vest": "Missing Vest",
    "Safety Shoes": "Missing Shoes",
    "Helmet": "Missing Helmet"
}

VIOLATION_LABELS = {
    "Without Glass": "No Glasses",
    "Without Mask": "No Mask",
    "Without Vest": "No Vest",
    "Without Safety Shoes": "No Shoes",
    "Without Helmet": "No Helmet"
}

# 🎯 OPTION SELECTOR
option = st.radio("Select Video Source", ["📤 Upload Video", "🎬 Use Demo Video"])

video_path = None

if option == "📤 Upload Video":
    uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        video_path = tfile.name

else:
    video_path = os.path.join(os.path.dirname(__file__), "..", "demo_videos", "construction_site.mp4")
    st.info(f"Using demo video: {video_path}")

# 🚀 PROCESS VIDEO
if video_path and st.button("▶️ Run Detection"):

    cap = cv2.VideoCapture(video_path)

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name

    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (frame_width, frame_height)
    )

    stframe = st.empty()
    progress_bar = st.progress(0)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    log_data = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        results = model(frame, conf=CONFIDENCE_THRESHOLD)[0]
        boxes = results.boxes

        detected_classes = []
        class_summary = {}
        violations = []

        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = class_names[cls_id]

            detected_classes.append(label)
            class_summary[label] = class_summary.get(label, 0) + 1

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            text = f"{label} ({conf:.2f})"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 🚨 Safety logic
        if "Person" in detected_classes:
            for label, msg in REQUIRED_LABELS.items():
                if label not in detected_classes:
                    violations.append(msg)

            for label, msg in VIOLATION_LABELS.items():
                if label in detected_classes:
                    violations.append(msg)

        # Overlay summary
        y = 20
        for label, count in class_summary.items():
            cv2.putText(frame, f"{label}: {count}", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            y += 20

        for v in violations:
            cv2.putText(frame, v, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            y += 25

        # Log
        log_data.append({
            "frame": frame_count,
            "violations": len(violations),
            "labels": ", ".join(detected_classes)
        })

        out.write(frame)
        stframe.image(frame, channels="BGR")

        progress_bar.progress(frame_count / total_frames)

    cap.release()
    out.release()

    st.success("✅ Processing Complete")

    df = pd.DataFrame(log_data)

    # 📊 Analytics
    st.subheader("📊 Violations Over Time")
    st.line_chart(df.set_index("frame")["violations"])

    st.subheader("📋 Frame-wise Log")
    st.dataframe(df)

    # Downloads
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download CSV", csv, "log.csv", "text/csv")

    with open(output_path, "rb") as f:
        st.download_button("⬇️ Download Video", f, "output.mp4", "video/mp4")
