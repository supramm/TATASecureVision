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

DEMO_VIDEOS = [
    {"title": "🏗️ Construction Site", "file": "construction_site.mp4"},
    {"title": "🏭 Factory Floor",      "file": "factory_floor.mp4"},
    {"title": "🧪 Lab Environment",    "file": "lab_env.mp4"},
    {"title": "🚧 Road Works",         "file": "road_works.mp4"},
]

# ── Session state defaults ─────────────────────────────────────────────────────
if "running"      not in st.session_state: st.session_state.running      = False
if "tmp_vid_path" not in st.session_state: st.session_state.tmp_vid_path = None

# ── Model ──────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

model       = load_model()
class_names = model.names  # dict {int: str}

# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.header("Controls")
source = st.sidebar.radio("Video source", ["Upload a video", "Use a demo video"])

video_path = None

if source == "Upload a video":
    uploaded = st.sidebar.file_uploader("Upload video", type=["mp4", "avi", "mov", "mkv"])
    if uploaded:
        # Only write to disk once per upload, reuse on reruns
        if st.session_state.tmp_vid_path is None or \
           not os.path.exists(st.session_state.tmp_vid_path):
            suffix = os.path.splitext(uploaded.name)[-1] or ".mp4"
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tmp.write(uploaded.read())
            tmp.close()
            st.session_state.tmp_vid_path = tmp.name
        video_path = st.session_state.tmp_vid_path
else:
    # Clear any stale upload temp path when switching to demo mode
    st.session_state.tmp_vid_path = None
    options = {d["title"]: d["file"] for d in DEMO_VIDEOS}
    choice  = st.sidebar.radio("Pick a clip", list(options.keys()))
    fpath   = os.path.join(DEMO_DIR, options[choice])
    if os.path.exists(fpath):
        video_path = fpath
    else:
        st.sidebar.warning(f"File not found: {options[choice]}")

conf       = st.sidebar.slider("Confidence", 0.10, 0.90, 0.25, 0.05)
frame_skip = st.sidebar.slider("Process every N frames", 1, 6, 2)

st.sidebar.markdown("---")
mode = st.sidebar.radio(
    "Mode",
    ["▶ Live Preview", "💾 Export Annotated Video"],
    help="Live Preview streams frames as they process. Export saves the full annotated video for download."
)

run  = st.sidebar.button("Run",  disabled=video_path is None, use_container_width=True)
stop = st.sidebar.button("■ Stop", disabled=mode == "💾 Export Annotated Video",
                          use_container_width=True)

if stop: st.session_state.running = False
if run:  st.session_state.running = True

# ── Helpers ────────────────────────────────────────────────────────────────────
def render_peak_html(peak_counts):
    rows = []
    for cls, cnt in peak_counts.most_common():
        is_bad = "no-" in cls.lower() or "without" in cls.lower()
        colour = "#ff4b4b" if is_bad else "#21c354"
        rows.append(
            f"<div style='display:flex;justify-content:space-between;"
            f"padding:5px 8px;margin-bottom:4px;border-radius:5px;"
            f"background:rgba(255,255,255,0.05);'>"
            f"<span style='color:{colour}'>{cls}</span>"
            f"<span style='font-weight:700;color:{colour}'>{cnt}</span>"
            f"</div>"
        )
    return "".join(rows)

def render_summary(peak_counts):
    st.markdown("---")
    st.subheader("✅ Analysis Complete — Final Summary")
    violations = {k: v for k, v in peak_counts.items()
                  if "no-" in k.lower() or "without" in k.lower()}
    compliant  = {k: v for k, v in peak_counts.items()
                  if "no-" not in k.lower() and "without" not in k.lower()}
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**✅ PPE Detected**")
        for k, v in sorted(compliant.items(), key=lambda x: -x[1]):
            st.metric(k, v)
    with c2:
        st.markdown("**❌ Violations**")
        if violations:
            for k, v in sorted(violations.items(), key=lambda x: -x[1]):
                st.metric(k, v, delta=f"peak {v} in one frame", delta_color="inverse")
        else:
            st.success("No violations detected!")

# ── Layout ─────────────────────────────────────────────────────────────────────
vid_col, info_col = st.columns([2, 1])

with vid_col:
    frame_slot = st.empty()

with info_col:
    st.markdown("#### 🔍 This Frame")
    det_slot  = st.empty()
    st.markdown("---")
    st.markdown("#### 📊 Peak Counts")
    st.caption("Max of each class seen in any single frame")
    peak_slot = st.empty()
    st.markdown("---")
    prog_slot = st.empty()

if not st.session_state.running:
    frame_slot.info("Select a video source and press Run")

# ══════════════════════════════════════════════════════════════════════════════
# MODE A — Live Preview
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.running and video_path and mode == "▶ Live Preview":

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Could not open video — try re-uploading or check the demo file path.")
        st.session_state.running = False
        st.stop()

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    fps   = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
    idx   = 0
    peak_counts = Counter()

    try:
        while cap.isOpened() and st.session_state.running:
            ret, frame = cap.read()
            if not ret:
                break

            idx += 1
            if idx % frame_skip != 0:
                continue

            results  = model(frame, conf=conf, verbose=False)[0]
            boxes    = results.boxes
            detected = [class_names[int(c)] for c in boxes.cls] if len(boxes) > 0 else []

            frame_counts = Counter(detected)
            for cls, cnt in frame_counts.items():
                if cnt > peak_counts[cls]:
                    peak_counts[cls] = cnt

            annotated = results.plot(line_width=2)
            h, w      = annotated.shape[:2]
            if w > 800:
                annotated = cv2.resize(annotated, (800, int(h * 800 / w)))

            if idx % 5 == 0:
                frame_slot.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                                 use_container_width=True)

            if detected:
                lines = []
                for cls, cnt in sorted(frame_counts.items(), key=lambda x: -x[1]):
                    icon = "🔴" if ("no-" in cls.lower() or "without" in cls.lower()) else "🟢"
                    lines.append(f"{icon} **{cls}** × {cnt}")
                det_slot.markdown("\n".join(lines))
            else:
                det_slot.markdown("_Nothing detected_")

            if peak_counts:
                peak_slot.markdown(render_peak_html(peak_counts), unsafe_allow_html=True)

            pct = min(idx / total, 1.0)
            ts  = f"{int(idx/fps//60):02d}:{int(idx/fps%60):02d}"
            prog_slot.progress(pct, text=f"{ts}  |  frame {idx}/{total}")

            time.sleep(0.01)

    finally:
        cap.release()
        st.session_state.running = False

    if peak_counts:
        render_summary(peak_counts)

# ══════════════════════════════════════════════════════════════════════════════
# MODE B — Export Annotated Video
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.running and video_path and mode == "💾 Export Annotated Video":

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Could not open video — try re-uploading or check the demo file path.")
        st.session_state.running = False
        st.stop()

    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    fps    = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    idx    = 0
    peak_counts = Counter()

    # temp output file
    out_tmp  = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    out_path = out_tmp.name
    out_tmp.close()

    writer = cv2.VideoWriter(
        out_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    prog_slot.info("⏳ Processing all frames and writing annotated video…")

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            idx += 1
            results  = model(frame, conf=conf, verbose=False)[0]
            boxes    = results.boxes
            detected = [class_names[int(c)] for c in boxes.cls] if len(boxes) > 0 else []

            frame_counts = Counter(detected)
            for cls, cnt in frame_counts.items():
                if cnt > peak_counts[cls]:
                    peak_counts[cls] = cnt

            # write every frame (no skip — export should be complete)
            annotated = results.plot(line_width=2)
            writer.write(annotated)

            # preview every 15th frame so user sees progress
            if idx % 15 == 0:
                h, w = annotated.shape[:2]
                preview = cv2.resize(annotated, (800, int(h * 800 / w))) if w > 800 else annotated
                frame_slot.image(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB),
                                 use_container_width=True,
                                 caption=f"Processing frame {idx}/{total}…")

            if peak_counts:
                peak_slot.markdown(render_peak_html(peak_counts), unsafe_allow_html=True)

            pct = min(idx / total, 1.0)
            ts  = f"{int(idx/fps//60):02d}:{int(idx/fps%60):02d}"
            prog_slot.progress(pct, text=f"{ts}  |  frame {idx}/{total}")

    finally:
        cap.release()
        writer.release()
        st.session_state.running = False

    # ── Offer download ──────────────────────────────────────────────────────
    frame_slot.empty()
    prog_slot.empty()

    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        with open(out_path, "rb") as f:
            video_bytes = f.read()
        os.unlink(out_path)   # clean up temp file after reading

        st.success("✅ Done! Your annotated video is ready.")
        st.download_button(
            label="⬇️ Download Annotated Video",
            data=video_bytes,
            file_name="ppe_annotated.mp4",
            mime="video/mp4",
            use_container_width=True,
        )
    else:
        st.error("Something went wrong writing the video file.")

    if peak_counts:
        render_summary(peak_counts)

st.markdown("---")
st.caption("© 2025 | TATAVision Secure | Powered by YOLOv8")
