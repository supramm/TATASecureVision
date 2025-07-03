# 🧰 Real-Time Harness Compliance Detection

This project is a real-time harness detection system using [YOLOv8](https://github.com/ultralytics/ultralytics), [Streamlit](https://github.com/streamlit/streamlit), and [streamlit-webrtc](https://github.com/whitphx/streamlit-webrtc). It flags workers not wearing a safety harness by analyzing webcam feed and saves evidence frames for review.

---

## 🚀 Demo

![Demo Screenshot](screenshot.png) <!-- Add your own screenshot -->

---

## 📦 Features

* ✅ Real-time detection of persons without harness
* 📸 Saves flagged frames with timestamp
* ⚡ Powered by YOLOv8 object detection
* 🖥️ Live webcam feed using `streamlit-webrtc`
* 🧠 Model loading with `@st.cache_resource`

---

## 🧠 Tech Stack

| Tool / Library                                                  | Usage            |
| --------------------------------------------------------------- | ---------------- |
| [YOLOv8](https://github.com/ultralytics/ultralytics)            | Object detection |
| [Streamlit](https://github.com/streamlit/streamlit)             | Web UI           |
| [streamlit-webrtc](https://github.com/whitphx/streamlit-webrtc) | Webcam stream    |
| [OpenCV](https://github.com/opencv/opencv-python)               | Frame processing |
| [Python](https://www.python.org/)                               | Backend logic    |

---

## 🖼️ How It Works

1. **YOLOv8 person model** detects people in the webcam feed.
2. **Harness model** (trained on custom data) checks if the person region contains a harness.
3. If **no harness** is detected, it:

   * Draws a red bounding box
   * Shows “❌ No Harness” label
   * Saves the frame to the `flagged_frames/` folder
4. If harness **is present**, it:

   * Draws a green bounding box
   * Shows “✅ Harness” label
