import streamlit as st
import cv2
import numpy as np
import os
import tempfile
import time
from ultralytics import YOLO
from collections import defaultdict

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PPE Demo | TATAVision Secure",
    page_icon="🎬",
    layout="wide",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;600;700&family=Share+Tech+Mono&family=Exo+2:wght@300;400;600&display=swap');

/* ── Root theme ── */
:root {
    --bg:        #0a0e17;
    --surface:   #111827;
    --border:    #1e2d45;
    --accent:    #00c8ff;
    --accent2:   #ff4560;
    --ok:        #00e396;
    --warn:      #feb019;
    --text:      #e2e8f0;
    --muted:     #64748b;
    --font-head: 'Rajdhani', sans-serif;
    --font-mono: 'Share Tech Mono', monospace;
    --font-body: 'Exo 2', sans-serif;
}

html, body, [class*="css"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: var(--font-body) !important;
}

/* ── Header bar ── */
.ppe-header {
    background: linear-gradient(135deg, #0d1b2a 0%, #0a1628 60%, #0d2137 100%);
    border: 1px solid var(--border);
    border-left: 4px solid var(--accent);
    border-radius: 8px;
    padding: 24px 32px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}
.ppe-header::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 200px; height: 200px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(0,200,255,0.08) 0%, transparent 70%);
    pointer-events: none;
}
.ppe-header h1 {
    font-family: var(--font-head) !important;
    font-size: 2.2rem !important;
    font-weight: 700 !important;
    color: var(--accent) !important;
    letter-spacing: 2px !important;
    margin: 0 0 4px 0 !important;
    text-transform: uppercase;
}
.ppe-header p {
    color: var(--muted) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.78rem !important;
    margin: 0 !important;
    letter-spacing: 1px;
}

/* ── Section label ── */
.section-label {
    font-family: var(--font-mono) !important;
    font-size: 0.7rem !important;
    letter-spacing: 3px !important;
    color: var(--accent) !important;
    text-transform: uppercase !important;
    margin-bottom: 10px !important;
    opacity: 0.8;
}

/* ── Demo video cards ── */
.demo-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin-bottom: 20px;
}
.demo-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 14px 16px;
    cursor: pointer;
    transition: all 0.2s ease;
    position: relative;
}
.demo-card:hover { border-color: var(--accent); }
.demo-card.active {
    border-color: var(--accent);
    background: rgba(0,200,255,0.07);
}
.demo-card .icon { font-size: 1.6rem; margin-bottom: 6px; }
.demo-card .title {
    font-family: var(--font-head);
    font-size: 0.95rem;
    font-weight: 600;
    color: var(--text);
    letter-spacing: 0.5px;
}
.demo-card .desc {
    font-size: 0.72rem;
    color: var(--muted);
    margin-top: 3px;
    font-family: var(--font-mono);
}

/* ── Metric tiles ── */
.metric-row {
    display: flex;
    gap: 10px;
    margin-bottom: 16px;
    flex-wrap: wrap;
}
.metric-tile {
    flex: 1;
    min-width: 100px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 14px 16px;
    text-align: center;
}
.metric-tile .val {
    font-family: var(--font-mono);
    font-size: 1.6rem;
    font-weight: 700;
    display: block;
}
.metric-tile .lbl {
    font-size: 0.65rem;
    letter-spacing: 2px;
    color: var(--muted);
    text-transform: uppercase;
    margin-top: 2px;
}
.val-ok    { color: var(--ok); }
.val-warn  { color: var(--warn); }
.val-err   { color: var(--accent2); }
.val-blue  { color: var(--accent); }

/* ── Violation log ── */
.vlog-entry {
    background: rgba(255,69,96,0.07);
    border-left: 3px solid var(--accent2);
    border-radius: 0 6px 6px 0;
    padding: 7px 12px;
    margin-bottom: 6px;
    font-family: var(--font-mono);
    font-size: 0.75rem;
    color: #f87171;
}
.vlog-entry span { color: var(--muted); margin-right: 8px; }

/* ── Compliant log ── */
.clog-entry {
    background: rgba(0,227,150,0.07);
    border-left: 3px solid var(--ok);
    border-radius: 0 6px 6px 0;
    padding: 7px 12px;
    margin-bottom: 6px;
    font-family: var(--font-mono);
    font-size: 0.75rem;
    color: #6ee7b7;
}
.clog-entry span { color: var(--muted); margin-right: 8px; }

/* ── Upload zone ── */
.upload-zone {
    border: 2px dashed var(--border);
    border-radius: 10px;
    padding: 40px 20px;
    text-align: center;
    background: var(--surface);
    margin-bottom: 20px;
    transition: border-color 0.2s;
}
.upload-zone:hover { border-color: var(--accent); }

/* ── Progress bar override ── */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, var(--accent), var(--ok)) !important;
}

/* ── Streamlit chrome cleanup ── */
header[data-testid="stHeader"] { background: var(--bg) !important; }
.stSidebar { background: #0d1520 !important; }
[data-testid="stFileUploaderDropzone"] {
    background: var(--surface) !important;
    border-color: var(--border) !important;
}
div[data-testid="stRadio"] label { color: var(--text) !important; }
.stButton > button {
    background: var(--accent) !important;
    color: #000 !important;
    border: none !important;
    font-family: var(--font-head) !important;
    font-weight: 600 !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
    border-radius: 6px !important;
    padding: 10px 28px !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    background: #00a8d8 !important;
    transform: translateY(-1px);
}
.stop-btn > button {
    background: var(--accent2) !important;
    color: #fff !important;
}
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="ppe-header">
    <h1>🎬 PPE Demo Analyzer</h1>
    <p>// OFFLINE VIDEO ANALYSIS · YOLOV8 INFERENCE ENGINE · TATAVIISION SECURE</p>
</div>
""", unsafe_allow_html=True)

# ── Load shared model ──────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "..", "best.pt")
    return YOLO(model_path)

model = load_model()
class_names = model.names

REQUIRED_PPE = {
    "Glasses":       ("🕶️", "Safety Glasses"),
    "Mask":          ("😷", "Face Mask"),
    "Vest":          ("🦺", "Safety Vest"),
    "Safety Shoes":  ("🥾", "Safety Shoes"),
    "Helmet":        ("⛑️", "Helmet"),
}
VIOLATION_LABELS = {
    "Without Glass":        "No safety glasses",
    "Without Mask":         "No mask",
    "Without Vest":         "No safety vest",
    "Without Safety Shoes": "No safety shoes",
    "Without Helmet":       "No helmet",
}
CONFIDENCE_THRESHOLD = 0.25

# ── Demo video registry ────────────────────────────────────────────────────────
DEMO_DIR = os.path.join(os.path.dirname(__file__), "..", "demo_videos")

DEMO_VIDEOS = [
    {
        "id": "demo1",
        "icon": "🏗️",
        "title": "Construction Site",
        "desc": "Outdoor scaffold zone",
        "file": "construction_site.mp4",
    },
    {
        "id": "demo2",
        "icon": "🏭",
        "title": "Factory Floor",
        "desc": "Heavy machinery area",
        "file": "factory_floor.mp4",
    },
    {
        "id": "demo3",
        "icon": "🧪",
        "title": "Lab Environment",
        "desc": "Chemical handling zone",
        "file": "lab_env.mp4",
    },
    {
        "id": "demo4",
        "icon": "🚧",
        "title": "Road Works",
        "desc": "Traffic-adjacent site",
        "file": "road_works.mp4",
    },
]

# ── Session state ──────────────────────────────────────────────────────────────
if "running" not in st.session_state:
    st.session_state.running = False
if "selected_demo" not in st.session_state:
    st.session_state.selected_demo = None

# ═══════════════════════════════════════════════════════════════════════════════
# LAYOUT — left panel (controls) | right panel (output + analytics)
# ═══════════════════════════════════════════════════════════════════════════════
left, right = st.columns([1, 2], gap="large")

# ── LEFT PANEL ─────────────────────────────────────────────────────────────────
with left:
    # ── Source selector ──
    st.markdown('<p class="section-label">▸ Select Source</p>', unsafe_allow_html=True)
    source_mode = st.radio(
        "Source",
        ["📁 Upload a video", "🎬 Use a demo video"],
        label_visibility="collapsed",
    )

    uploaded_file = None
    demo_video_path = None

    # ── Upload mode ──
    if source_mode == "📁 Upload a video":
        st.markdown('<p class="section-label">▸ Upload Video</p>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Drop your video here",
            type=["mp4", "avi", "mov", "mkv"],
            label_visibility="collapsed",
        )
        if uploaded_file:
            st.success(f"✅ Loaded: `{uploaded_file.name}`")

    # ── Demo mode ──
    else:
        st.markdown('<p class="section-label">▸ Choose Demo Clip</p>', unsafe_allow_html=True)
        for dv in DEMO_VIDEOS:
            fpath = os.path.join(DEMO_DIR, dv["file"])
            available = os.path.exists(fpath)
            is_sel = st.session_state.selected_demo == dv["id"]
            tag = "✔ " if is_sel else ""
            disabled_note = "" if available else " *(file missing)*"
            if st.button(
                f"{dv['icon']} {tag}{dv['title']}{disabled_note} — {dv['desc']}",
                key=f"btn_{dv['id']}",
                disabled=not available,
                use_container_width=True,
            ):
                st.session_state.selected_demo = dv["id"]
                st.rerun()

        if st.session_state.selected_demo:
            sel = next(d for d in DEMO_VIDEOS if d["id"] == st.session_state.selected_demo)
            demo_video_path = os.path.join(DEMO_DIR, sel["file"])
            st.info(f"Selected: **{sel['icon']} {sel['title']}**")

    # ── Settings ──
    st.markdown("---")
    st.markdown('<p class="section-label">▸ Inference Settings</p>', unsafe_allow_html=True)
    conf_thresh = st.slider("Confidence Threshold", 0.10, 0.90, CONFIDENCE_THRESHOLD, 0.05)
    frame_skip = st.slider("Process every N-th frame", 1, 6, 2,
                            help="Higher = faster but less detailed")
    show_boxes = st.checkbox("Show bounding boxes", value=True)
    show_labels = st.checkbox("Show class labels", value=True)

    st.markdown("---")
    # ── Run / Stop buttons ──
    can_run = (uploaded_file is not None) or (demo_video_path is not None)

    run_col, stop_col = st.columns(2)
    with run_col:
        run_btn = st.button("▶ Run Analysis", disabled=not can_run or st.session_state.running,
                             use_container_width=True)
    with stop_col:
        st.markdown('<div class="stop-btn">', unsafe_allow_html=True)
        stop_btn = st.button("■ Stop", disabled=not st.session_state.running,
                              use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    if run_btn:
        st.session_state.running = True
    if stop_btn:
        st.session_state.running = False

# ── RIGHT PANEL ────────────────────────────────────────────────────────────────
with right:
    st.markdown('<p class="section-label">▸ Live Output</p>', unsafe_allow_html=True)

    # Placeholders
    video_placeholder = st.empty()
    progress_bar      = st.empty()

    st.markdown('<p class="section-label">▸ Session Analytics</p>', unsafe_allow_html=True)
    metrics_placeholder = st.empty()

    col_v, col_c = st.columns(2)
    with col_v:
        st.markdown('<p class="section-label">▸ Violations Log</p>', unsafe_allow_html=True)
        violations_placeholder = st.empty()
    with col_c:
        st.markdown('<p class="section-label">▸ Compliance Log</p>', unsafe_allow_html=True)
        compliance_placeholder = st.empty()

    # ── Idle state display ──
    if not st.session_state.running:
        video_placeholder.markdown("""
        <div style="background:#111827;border:1px solid #1e2d45;border-radius:10px;
                    height:360px;display:flex;flex-direction:column;
                    align-items:center;justify-content:center;gap:12px;">
            <div style="font-size:3rem;">🎬</div>
            <div style="font-family:'Rajdhani',sans-serif;font-size:1.1rem;
                        color:#64748b;letter-spacing:2px;text-transform:uppercase;">
                Awaiting video source
            </div>
            <div style="font-family:'Share Tech Mono',monospace;font-size:0.7rem;
                        color:#334155;letter-spacing:1px;">
                SELECT A VIDEO AND PRESS RUN
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # INFERENCE LOOP
    # ══════════════════════════════════════════════════════════════════════════
    if st.session_state.running:

        # ── Resolve video path ──
        tmp_path = None
        if uploaded_file is not None:
            suffix = os.path.splitext(uploaded_file.name)[-1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
                f.write(uploaded_file.read())
                tmp_path = f.name
            video_source = tmp_path
        elif demo_video_path:
            video_source = demo_video_path
        else:
            st.error("No video source found.")
            st.session_state.running = False
            st.stop()

        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            st.error("❌ Could not open video. Check the file path or re-upload.")
            st.session_state.running = False
            if tmp_path:
                os.unlink(tmp_path)
            st.stop()

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps          = cap.get(cv2.CAP_PROP_FPS) or 25

        # ── Aggregate stats ──
        frame_idx       = 0
        processed       = 0
        total_persons   = 0
        total_violations= 0
        total_compliant = 0
        class_counter   = defaultdict(int)
        violation_log   = []   # list of (frame_ts, violation_str)
        compliance_log  = []   # list of (frame_ts, label)

        # ── Frame loop ──
        while cap.isOpened() and st.session_state.running:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1

            # Skip frames for speed
            if frame_idx % frame_skip != 0:
                continue

            processed += 1
            ts = f"{int(frame_idx / fps // 60):02d}:{int(frame_idx / fps % 60):02d}"

            # ── YOLO inference ──
            results = model(frame, conf=conf_thresh, verbose=False)[0]
            boxes   = results.boxes
            detected = [class_names[int(c)] for c in boxes.cls]

            for cls in detected:
                class_counter[cls] += 1

            # Annotate
            if show_boxes:
                annotated = results.plot(labels=show_labels)
            else:
                annotated = frame.copy()

            # Overlay compliance warnings
            if "Person" in detected:
                total_persons += 1
                frame_violations = []
                frame_ok = []

                for label, (_, nice) in REQUIRED_PPE.items():
                    if label not in detected:
                        frame_violations.append(nice)
                for label, nice in VIOLATION_LABELS.items():
                    if label in detected:
                        frame_violations.append(nice)
                for label, (_, nice) in REQUIRED_PPE.items():
                    if label in detected:
                        frame_ok.append(nice)

                if frame_violations:
                    total_violations += 1
                    for v in frame_violations:
                        cv2.putText(annotated, f"⚠ {v}", (10, 30 + frame_violations.index(v)*30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (60, 60, 255), 2)
                        if len(violation_log) < 50:
                            violation_log.append((ts, v))
                else:
                    total_compliant += 1
                    if frame_ok and len(compliance_log) < 50:
                        compliance_log.append((ts, "All PPE compliant"))

            # ── Resize for display ──
            h, w = annotated.shape[:2]
            max_w = 760
            if w > max_w:
                scale = max_w / w
                annotated = cv2.resize(annotated, (max_w, int(h * scale)))

            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

            # ── Update video frame ──
            video_placeholder.image(annotated_rgb, channels="RGB", use_container_width=True)

            # ── Update progress ──
            pct = min(frame_idx / max(total_frames, 1), 1.0)
            progress_bar.progress(pct, text=f"Frame {frame_idx}/{total_frames}  |  {ts}")

            # ── Update metrics ──
            compliance_rate = (
                round(total_compliant / total_persons * 100, 1) if total_persons else 0
            )
            c_color = "val-ok" if compliance_rate >= 80 else "val-warn" if compliance_rate >= 50 else "val-err"

            metrics_placeholder.markdown(f"""
            <div class="metric-row">
                <div class="metric-tile">
                    <span class="val val-blue">{processed}</span>
                    <div class="lbl">Frames Analysed</div>
                </div>
                <div class="metric-tile">
                    <span class="val val-blue">{total_persons}</span>
                    <div class="lbl">Person Detections</div>
                </div>
                <div class="metric-tile">
                    <span class="val val-err">{total_violations}</span>
                    <div class="lbl">Violation Frames</div>
                </div>
                <div class="metric-tile">
                    <span class="val val-ok">{total_compliant}</span>
                    <div class="lbl">Compliant Frames</div>
                </div>
                <div class="metric-tile">
                    <span class="val {c_color}">{compliance_rate}%</span>
                    <div class="lbl">Compliance Rate</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Update logs (last 8 entries) ──
            vlog_html = "".join(
                f'<div class="vlog-entry"><span>{ts}</span>{v}</div>'
                for ts, v in violation_log[-8:][::-1]
            ) or '<div style="color:#334155;font-family:\'Share Tech Mono\',monospace;font-size:0.75rem;">No violations yet</div>'

            clog_html = "".join(
                f'<div class="clog-entry"><span>{ts}</span>{v}</div>'
                for ts, v in compliance_log[-8:][::-1]
            ) or '<div style="color:#334155;font-family:\'Share Tech Mono\',monospace;font-size:0.75rem;">No compliant frames yet</div>'

            violations_placeholder.markdown(vlog_html, unsafe_allow_html=True)
            compliance_placeholder.markdown(clog_html, unsafe_allow_html=True)

            # Small sleep to keep UI responsive
            time.sleep(0.01)

        cap.release()
        if tmp_path:
            os.unlink(tmp_path)

        st.session_state.running = False

        # ── Final summary ──
        if processed > 0:
            compliance_rate = round(total_compliant / max(total_persons, 1) * 100, 1)
            st.markdown("---")
            st.markdown('<p class="section-label">▸ Analysis Complete — Summary</p>', unsafe_allow_html=True)

            top_detections = sorted(class_counter.items(), key=lambda x: -x[1])[:6]
            det_html = " &nbsp;·&nbsp; ".join(
                f'<span style="color:var(--accent);font-family:var(--font-mono)">{k}</span>'
                f'<span style="color:var(--muted)"> ×{v}</span>'
                for k, v in top_detections
            )

            verdict_color = "#00e396" if compliance_rate >= 80 else "#feb019" if compliance_rate >= 50 else "#ff4560"
            verdict_text  = "✔ SITE COMPLIANT" if compliance_rate >= 80 else "⚠ REVIEW REQUIRED" if compliance_rate >= 50 else "✖ NON-COMPLIANT"

            st.markdown(f"""
            <div style="background:#111827;border:1px solid #1e2d45;border-radius:10px;padding:24px 28px;margin-top:8px;">
                <div style="font-family:'Rajdhani',sans-serif;font-size:1.8rem;font-weight:700;
                            color:{verdict_color};letter-spacing:3px;margin-bottom:14px;">
                    {verdict_text}
                </div>
                <div style="font-family:'Share Tech Mono',monospace;font-size:0.8rem;
                            color:#64748b;margin-bottom:12px;line-height:1.9;">
                    FRAMES PROCESSED &nbsp;·&nbsp; <span style="color:#e2e8f0">{processed}</span><br>
                    PERSON DETECTIONS &nbsp;·&nbsp; <span style="color:#e2e8f0">{total_persons}</span><br>
                    VIOLATION FRAMES &nbsp;&nbsp;·&nbsp; <span style="color:#ff4560">{total_violations}</span><br>
                    COMPLIANT FRAMES &nbsp;&nbsp;·&nbsp; <span style="color:#00e396">{total_compliant}</span><br>
                    COMPLIANCE RATE &nbsp;&nbsp;&nbsp;·&nbsp; <span style="color:{verdict_color}">{compliance_rate}%</span>
                </div>
                <div style="font-family:'Share Tech Mono',monospace;font-size:0.7rem;
                            color:#334155;margin-top:10px;border-top:1px solid #1e2d45;padding-top:10px;">
                    TOP DETECTIONS &nbsp; {det_html}
                </div>
            </div>
            """, unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#334155;font-family:\"Share Tech Mono\",monospace;"
    "font-size:0.7rem;letter-spacing:2px;'>© 2025 | TATAVision Secure | Powered by YOLOv8</div>",
    unsafe_allow_html=True,
)