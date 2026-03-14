# 🛡️ TATASecure Vision 🚧  
### Smart vision safeguarding Tata Motors’ workforce.

Welcome to **TATASecure Vision** – a real-time, multi-model safety surveillance suite designed to **prevent accidents, ensure compliance**, and **safeguard lives** in industrial environments using the power of **Computer Vision + AI** 💡.

---
![Python](https://img.shields.io/badge/Python-3.10-blue)
![YOLO](https://img.shields.io/badge/YOLOv11-ObjectDetection-red)
![Streamlit](https://img.shields.io/badge/Streamlit-App-orange)


## 🔍 Overview

Industrial safety is non-negotiable. TATASecure Vision is an **end-to-end AI-powered system** that deploys **multiple custom-trained object detection models** on **live factory camera feeds** to ensure:

- **PPE Compliance**
- **Harness Safety at Heights**
- **Walking Within Safe Zones**
- **Kitchen/Factory Hazard Alerts**

These models are deployed on the edge and cloud via **Streamlit WebApps** to provide **real-time feedback** and decision support for on-ground supervisors and safety officers.

---

## 🚀 Deployed Apps

🌐 Full System Dashboard:Access all modules and live demos at🔗 https://tata-secure-vision.vercel.app/

| Module | Description | Accuracy | Live App |
|--------|-------------|----------|----------|
| 🦺 **PPE Detection** | Detects **Safety Helmet**, **Vest**, **Glasses**, **Gloves**, and **Boots** at factory entrances. | **93%** | [Open App 🔗](https://tatasecurevision-ppe3.streamlit.app/) |
| 🧗‍♂️ **Harness Detection** | Monitors usage of **Safety Harnesses** on elevated workstations. | **98%** | [Open App 🔗](https://tatasecurevision-harness.streamlit.app/) |
| 🚶‍♂️ **Safety Region Compliance** | Ensures workers walk only in **safe green-lined factory zones**. Uses **Shapely** for polygon-based checks. | **95%** | [Open App 🔗](https://tatasecurevision-safetyregion.streamlit.app/) |
| 🍳 **Kitchen Safety Detection** | Custom model to detect **Kadhai** and **Coco**; flags if **hot oil vessels are left unattended > 5 mins**. | Custom Accuracy | [Open App 🔗](https://tatasecurevision-kitchensafe.streamlit.app/) |

---

## 📦 Model Training Details

All models were trained using **Roboflow datasets** and annotated using **CVAT** and **Label Studio**. Here's a quick overview:

### 🦺 PPE Detection
- **Classes**: Safety Helmet, Vest, Glasses, Gloves, Boots  
- **Accuracy**: **93%**  
- **Dataset**: [Roboflow PPE Dataset](https://app.roboflow.com/supram/safety-pyazl-khx3r/)  
- **Deployment**: Cameras at **entry checkpoints**

### 🧗‍♂️ Safety Harness Detection
- **Classes**: Harness On / Harness Off  
- **Accuracy**: **98%**  
- **Dataset**: [Harness Detection Dataset](https://app.roboflow.com/supram/harness-knfmk-9ozbf/1)  
- **Deployment**: **Overhead cameras** in high-altitude workspaces

### 🚶‍♂️ Region Compliance Detection
- **Approach**: Trained on **live factory video feed**, and used **Shapely** for polygon detection  
- **Use Case**: Ensures workers stay within **green-lined safe zones**

### 🍳 Kitchen Safety Model
- **Goal**: Detect if **Kadhai with hot oil** is left **unattended > 5 mins**  
- **Custom Annotated Dataset**: [Kitchen Safety Dataset](https://app.roboflow.com/supram/kitchensafety/1)  
- **Annotation Tools**: **CVAT**, **Label Studio**  
- **Deployment**: **Factory canteen kitchens**

---

## 🔧 Tech Stack

- **YOLOv8 / YOLOv11** (Ultralytics)
- **Streamlit** for frontend apps
- **Roboflow** for dataset management
- **Shapely** for zone detection
- **CVAT** and **Label Studio** for custom annotations
- **Google Colab + PyTorch** for model training
- **Hosted on Streamlit Cloud**

---

## 📸 Screenshots:

<img width="1918" height="1079" alt="VercelSite" src="https://github.com/user-attachments/assets/2c03c6b4-f945-45a4-b3d7-ba465ed2835b" />
<img width="1919" height="1079" alt="VercelSite2" src="https://github.com/user-attachments/assets/e6f30bf0-451a-4807-84b8-7d6856f8a332" />
<img width="1901" height="1080" alt="VercelSite3" src="https://github.com/user-attachments/assets/18b30be2-3965-46e4-8908-6651b0d41e79" />
<img width="1919" height="1079" alt="VercelSite4" src="https://github.com/user-attachments/assets/e5d8ca90-0871-493e-bd8b-5c98e70c5e5b" />

---


## 🧠 Key Highlights

✅ Multiple purpose-built CV models trained for factory-specific use-cases  
✅ Seamless deployment with live inference on video streams  
✅ Combines **geometric logic** (Shapely) and **temporal logic** (unattended duration)  
✅ High-accuracy models with real-world validation  
✅ Custom annotation and dataset creation to tailor models for critical safety scenarios

---

## 👨‍💼 About the Author

👋 Hi, I’m **Supram Kumar**, a passionate AI & ML engineer currently working with **Tata Technologies**. This project was conceptualized, developed, and deployed as part of my **industrial internship** experience to solve **real safety problems** in industrial zones.

Feel free to check out my other work or connect on [LinkedIn](https://www.linkedin.com/in/supramkumar) or [GitHub](https://github.com/supramm) 💼



---

## 🧩 Future Work

- 📦 Docker-based edge deployment for offline cameras  
- 🔄 Integration with SMS/Email/WhatsApp alert system  
- 📊 Admin dashboard with analytics & alert logs  
- 🤖 Auto-retraining pipeline for continuously improving performance  

---

## 💡 Installation & Local Development

Coming soon...

> All deployed apps are ready-to-use, no setup required.

---

## ⭐ Give It a Star!

If you find this project helpful or interesting, don’t forget to ⭐ star this repo and share it!

---

## 📬 Feedback / Contributions

Have an idea or want to contribute? Open an issue or drop a PR! Collaboration is welcome 🙌
