import streamlit as st
import cv2
import os
import tempfile
import time
from ultralytics import YOLO
from collections import Counter

st.set_page_config(page_title="PPE Demo", page_icon="🎬", layout="wide")

st.title("🎬 PPE Demo Analyzer")
st.caption("Run the model on a demo video or upload your own")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "best.pt")
DEMO_DIR   = os.path.join(BASE_DIR, "demo_videos")

# ── Model ──────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

model       = load_model()
class_names = model.names

# ── Demo videos ────────────────────────────────────────────────────────────────
DEMO_VIDEOS = [
    {"title": "🏗️ Construction Site", "file": "construction_site.mp4"},
    {"title": "🏭 Factory Floor",      "file": "factory_floor.mp4"},
    {"title": "🧪 Lab Environment",    "file": "lab_env.mp4"},
    {"title": "🚧 Road Works",         "file": "road_works.mp4"},
]

# ── Sidebar controls ───────────────────────────────────────────────────────────
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

conf       = st.sidebar.slider("Confidence", 0.10, 0.90, 0.25, 0.05)
frame_skip = st.sidebar.slider("Process every N frames", 1, 6, 2)

run  = st.sidebar.button("▶ Run", disabled=video_path is None, use_container_width=True)
stop = st.sidebar.button("■ Stop", use_container_width=True)

if stop:
    st.session_state.running = False
if run:
    st.session_state.running = True

# ── Main layout ────────────────────────────────────────────────────────────────
vid_col, info_col = st.columns([2, 1])

with vid_col:
    frame_slot = st.empty()

with info_col:
    st.subheader("Live Detections")
    det_slot    = st.empty()
    st.subheader("Stats")
    stats_slot  = st.empty()
    prog_slot   = st.empty()

# ── Idle screen ────────────────────────────────────────────────────────────────
if not st.session_state.get("running"):
    frame_slot.info("Select a video source and press ▶ Run")

# ── Inference loop ─────────────────────────────────────────────────────────────
if st.session_state.get("running") and video_path:

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Could not open video.")
        st.session_state.running = False
        st.stop()

    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 25
    idx    = 0
    counts = Counter()

    while cap.isOpened() and st.session_state.get("running"):
        ret, frame = cap.read()
        if not ret:
            break

        idx += 1
        if idx % frame_skip != 0:
            continue

        # run model
        results  = model(frame, conf=conf, verbose=False)[0]
        detected = [class_names[int(c)] for c in results.boxes.cls]
        counts.update(detected)

        # annotated frame
        annotated = cv2.cvtColor(results.plot(), cv2.COLOR_BGR2RGB)
        frame_slot.image(annotated, use_container_width=True)

        # live detection list
        if detected:
            det_slot.markdown("\n".join(f"- **{c}**" for c in detected))
        else:
            det_slot.markdown("_Nothing detected_")

        # cumulative stats
        stats_md = "\n".join(f"**{k}** — {v}" for k, v in counts.most_common())
        stats_slot.markdown(stats_md)

        # progress
        pct = min(idx / max(total, 1), 1.0)
        ts  = f"{int(idx/fps//60):02d}:{int(idx/fps%60):02d}"
        prog_slot.progress(pct, text=f"{ts}  |  frame {idx}/{total}")

        time.sleep(0.01)

    cap.release()
    st.session_state.running = False
    st.success("✅ Done!")

st.markdown("---")
st.caption("© 2025 | TATAVision Secure | Powered by YOLOv8")
