import streamlit as st
import cv2
import os
import tempfile
from ultralytics import YOLO
from collections import Counter

st.set_page_config(page_title="PPE Demo", page_icon="🎬", layout="wide")

st.title("🎬 PPE Demo Analyzer")
st.caption("Runs the model on your video and gives you a fully annotated playback")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "best.pt")
DEMO_DIR   = os.path.join(BASE_DIR, "demo_videos")

DEMO_VIDEOS = [
    {"title": "🏗️ Construction Site", "file": "construction_site.mp4"},
    {"title": "🏭 Factory Floor",      "file": "factory_floor.mp4"},
    {"title": "🧪 Lab Environment",    "file": "lab_env.mp4"},
    {"title": "🚧 Road Works",         "file": "road_works.mp4"},
]

CONFIDENCE_THRESHOLD = 0.25

# ── Model ──────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

model       = load_model()
class_names = model.names

# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.header("Controls")
source = st.sidebar.radio("Video source", ["Upload a video", "Use a demo video"])

video_path = None

if source == "Upload a video":
    uploaded = st.sidebar.file_uploader("Upload video", type=["mp4", "avi", "mov", "mkv"])
    if uploaded:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded.name)[-1])
        tmp.write(uploaded.read())
        tmp.close()
        video_path = tmp.name
else:
    options = {d["title"]: d["file"] for d in DEMO_VIDEOS}
    choice  = st.sidebar.radio("Pick a clip", list(options.keys()))
    fpath   = os.path.join(DEMO_DIR, options[choice])
    if os.path.exists(fpath):
        video_path = fpath
    else:
        st.sidebar.warning(f"File not found: {options[choice]}")

conf = st.sidebar.slider("Confidence", 0.10, 0.90, CONFIDENCE_THRESHOLD, 0.05)

run = st.sidebar.button("▶ Run Analysis", disabled=video_path is None, use_container_width=True)

# ── Run ────────────────────────────────────────────────────────────────────────
if run and video_path:

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Could not open video.")
        st.stop()

    total      = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps        = cap.get(cv2.CAP_PROP_FPS) or 25
    width      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # output file
    out_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    out_path = out_file.name
    out_file.close()

    writer = cv2.VideoWriter(
        out_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    counts  = Counter()
    idx     = 0

    progress_bar  = st.progress(0, text="Processing…")
    preview_slot  = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        idx += 1
        results  = model(frame, conf=conf, verbose=False)[0]
        detected = [class_names[int(c)] for c in results.boxes.cls]
        counts.update(detected)

        annotated = results.plot()
        writer.write(annotated)

        # show every 10th frame as a live preview while processing
        if idx % 10 == 0:
            preview_slot.image(
                cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                use_container_width=True,
                caption=f"Processing… frame {idx}/{total}"
            )

        pct = min(idx / max(total, 1), 1.0)
        progress_bar.progress(pct, text=f"Processing frame {idx}/{total}")

    cap.release()
    writer.release()
    progress_bar.empty()
    preview_slot.empty()

    # ── Show results ──────────────────────────────────────────────────────────
    st.success("✅ Done! Here's your annotated video:")

    # Streamlit needs an H264-encoded mp4 to play in browser
    # Re-encode with ffmpeg if available, else serve as-is
    h264_path = out_path.replace(".mp4", "_h264.mp4")
    ffmpeg_ok = os.system(f"ffmpeg -y -i {out_path} -vcodec libx264 -acodec aac {h264_path} -loglevel quiet") == 0

    playable = h264_path if ffmpeg_ok and os.path.exists(h264_path) else out_path

    with open(playable, "rb") as f:
        st.video(f.read())

    # ── Detection summary ─────────────────────────────────────────────────────
    st.subheader("📊 Detection Summary")
    cols = st.columns(min(len(counts), 4))
    for i, (label, count) in enumerate(counts.most_common()):
        cols[i % len(cols)].metric(label, count)

else:
    st.info("Select a video source from the sidebar and press ▶ Run Analysis")

st.markdown("---")
st.caption("© 2025 | TATAVision Secure | Powered by YOLOv8")
