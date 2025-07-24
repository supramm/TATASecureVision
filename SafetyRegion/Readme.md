Our system ensures industrial safety through intelligent video processing:

- 📸 **Capture Frames**: The webcam or video feed is read frame-by-frame in real-time.
- 🟩 **Detect Safety Zone**: Green-colored safety regions are identified using HSV color thresholding.
- 🧍 **Detect Workers**: A custom-trained YOLO model detects factory workers in each frame.
- 📍 **Track Position**: Each worker’s bounding box center is compared against the safety zone.
- ⚠️ **Flag Violations**: If a worker is found outside the safety area, a violation is recorded and the annotated frames saved for compliance checks.
""")
