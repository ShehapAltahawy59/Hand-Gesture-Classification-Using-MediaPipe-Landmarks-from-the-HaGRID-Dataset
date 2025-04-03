```markdown
# ✋ Hand Gesture Recognition System

![Gesture Recognition Demo](demo.gif)

## 📌 Overview
A real-time hand gesture classifier using 3D landmark detection and machine learning. Processes webcam input to identify gestures with **98.7% accuracy**.

## 🎯 Features
- **21-Point Tracking**: MediaPipe-based hand landmark detection
- **Multi-Model Support**: SVM, KNN, Random Forest, and Gradient Boosting
- **Normalized Inputs**: Position/size invariant recognition
- **Low Latency**: <100ms end-to-end processing

## 🧠 Model Performance
| Model          | Accuracy | F1-Score |
|----------------|----------|----------|
| SVM            | 98.69%   | 98.69%   |
| KNN            | 97.90%   | 97.90%   |
| Random Forest  | 97.74%   | 97.74%   |


## 🛠️ Installation
```bash
# Clone repository
git clone https://github.com/yourusername/hand-gesture-recognition.git
cd hand-gesture-recognition

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/MacOS
.\venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

## 📋 Requirements
```text
mediapipe==0.8.9.1
scikit-learn==1.0.2
opencv-python==4.5.5.64
Flask==2.0.1
numpy==1.22.0
```

## 🚀 Usage
### Real-Time Recognition
```bash
python app.py
```
Access `http://localhost:5000` in your browser


