# 🔒 TataSecureVision

**Smart vision safeguarding Tata Motors’ workforce.**

TataSecureVision is a computer vision-based safety monitoring system developed during my internship at Tata Technologies. The system ensures industrial safety compliance by detecting personal protective equipment (PPE), monitoring designated safety zones, and enforcing kitchen safety protocols — all powered by real-time YOLOv11-based object detection.

---

## 🚀 Deployable Project

🔗 **Try the live demo:**
[https://tatasecurevision-ppe3.streamlit.app/](https://tatasecurevision-ppe3.streamlit.app/)

---

## 🧠 Modules

### 1. 🦺 PPE Compliance Detection

* Detects whether factory workers are wearing mandatory safety equipment like **harnesses**.
* Model trained on custom-labeled datasets with class balancing and YOLOv11 optimizations.
* Webcam/live input support via Streamlit.

### 2. 📍 Safety Region Monitoring (Optional Zone Compliance)

* Verifies if personnel are within designated safety zones.
* Uses **Shapely** to calculate whether a person is inside the defined polygonal area.

### 3. 🍳 Kitchen Safety Enforcement *(Planned/Prototype)*

* Ensures proper protective measures like gloves and masks in kitchen areas.
* Designed for expansion into food or health safety zones.

---

## 🛠️ Tech Stack

* **YOLOv11** – Object detection
* **Python**, **OpenCV**, **Streamlit** – App development
* **Shapely** – Zone compliance (geospatial)
* **Label Studio** – Dataset annotation
* **Google Colab** – Training
* **Streamlit Cloud** – Hosting

---

## 📁 Project Structure

```bash
TATASecureVision/
│
├── PPEStreamlitApp/        # Streamlit UI Code
├── runs/                   # YOLOv11 output
├── datasets/               # Annotated images
├── best.pt                 # Trained YOLOv11 model weights
├── utils/                  # Helper functions
└── README.md               # This file
```
