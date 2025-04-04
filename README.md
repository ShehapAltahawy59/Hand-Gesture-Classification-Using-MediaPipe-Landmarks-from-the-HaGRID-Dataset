# Hand Gesture Recognition System


![Alt](https://github.com/ShehapAltahawy59/Hand-Gesture-Classification-Using-MediaPipe-Landmarks-from-the-HaGRID-Dataset/raw/main/demo.gif?v=3) 


*Real-time gesture classification demo*

## Table of Contents
1. [Project Overview](#project-overview)
2. [System Architecture](#system-architecture)
3. [Model Development](#model-development)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Directory Structure](#directory-structure)
7. [License](#license)

## Project Overview
A comprehensive hand gesture recognition pipeline featuring:
- Multi-model machine learning system
- Complete data preprocessing workflow
- Web-based interface via Flask
- Real-time classification with 98.7% accuracy

## System Architecture

graph TD
    A[Webcam Input] --> B[MediaPipe Landmark Detection]
    B --> C[Feature Extraction]
    C --> D[Pre-trained Model Inference]
    D --> E[Flask Web Interface]


## Model Development
### Dataset Preparation
- Located in `dataset/` directory
- Includes labeled gesture samples

### Preprocessing Pipeline (notebooks/)
1. Data normalization
2. Feature engineering
3. Train-test splitting

### Model Training Workflow
1. Initial evaluation of multiple algorithms
2. Selection of top 3 performers (SVM, KNN, Random Forest)
3. Hyperparameter tuning via GridSearchCV
4. Final model selection based on:
   - Accuracy (98.7% for SVM)
   - Inference speed
   - Memory footprint

## Installation

### Prerequisites
- Python 3.8+
- Webcam-enabled system
- 4GB RAM minimum

```bash
# Clone repository
git clone https://github.com/ShehapAltahawy59/Hand-Gesture-Classification-Using-MediaPipe-Landmarks-from-the-HaGRID-Dataset.git
cd hand-gesture-recognition

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/MacOS
.\venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Running the Web Interface
```bash
python app.py
```
Access the interface at `http://localhost:5000`

### Running the script
```bash
python script.py
```


## Directory Structure
```
.
├── models/                # Pretrained model files
│   ├── svm_model.pkl
├── dataset/               # Training datasets
├── notebooks.ipynb            # Jupyter notebooks
├── app.py                 # Flask application
├── requirements.txt       # Dependencies
└── demo.gif               # System demonstration
```

