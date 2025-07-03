# 🧑‍🍳 StreamlitKitchenSafe: Kadhai Safety Monitoring App

This is a real-time safety monitoring web app built with **YOLOv8** and **Streamlit**. It detects if a **kadhai (wok)** is left unattended for more than 5 minutes and flags the situation by capturing a snapshot.

---

## 🔍 What It Does

* Detects **kadhai** using your custom-trained YOLO model (`best.pt`)
* Detects **person** presence using pretrained YOLOv8n
* If a kadhai is detected **but no person is nearby**, it starts a **5-minute timer**
* If the timer completes without a person returning, a snapshot of the frame is **saved as evidence**
* Works directly in the **browser using your webcam**

---

## 🎥 Live Demo (if deployed)

👉 [Click here to try it](https://your-app-name.streamlit.app) *(replace with your deployed URL)*

---

## 🧠 Models Used

* `best.pt` → Custom YOLOv8 model trained to detect a kadhai
* `yolov8n.pt` → Pretrained YOLOv8n model for detecting persons 
