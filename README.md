# 🖐️ Finger Drawing Using OpenCV

A real-time **virtual drawing application** that lets you draw on the screen using your **finger movements** captured through a webcam.  
Built using **Python, OpenCV, and MediaPipe**.

---

## 📌 Features
- Real-time webcam hand tracking
- Draw using **index finger**
- Pause drawing using **index + middle finger**
- Press **`c`** to clear the drawing board
- Press **`q`** to quit the application
- Smooth and responsive drawing

---

## 🧠 How It Works
1. Webcam captures live video feed  
2. MediaPipe detects hand landmarks  
3. Index finger tip position is tracked  
4. OpenCV draws lines based on finger movement  

---

## 📸 Demo Preview

<img width="787" height="640" alt="Screenshot 2025-12-28 010310" src="https://github.com/user-attachments/assets/5bb010c8-8d10-4137-9267-2e4595b2a8a0" />

---

## 🛠️ Tech Stack
- Python
- OpenCV
- MediaPipe
- NumPy

---

## 📦 Installation

### Clone the repository
git clone https://github.com/your-username/finger-drawing-opencv.git
cd finger-drawing-opencv
Install dependencies
pip install opencv-python mediapipe numpy

## ▶️ Run the Project
python finger_drawing.py


### ⚠️ Make sure your webcam is connected.

## 🎮 Controls
Action	Control
Draw	Index finger ☝
Stop drawing	Index + Middle finger ✌
Clear board	c
Quit	q
## 📁 Project Structure
finger-drawing-opencv/
│
├── finger_drawing.py
├── README.md
└── requirements.txt

## 🚀 Future Enhancements

🎨 Color selection using gestures

🧽 Eraser mode

✏️ Brush thickness control

💾 Save drawings as images

## 👤 Author

Daksh
Python | OpenCV | Computer Vision

## ⭐ Support

If you find this project useful, please star ⭐ the repository.
