# 🎭 Face AI System

<p align="center">
  <a href="https://github.com/Rajat-Pandya/face_ai_project/actions/workflows/ci.yml"><img src="https://github.com/Rajat-Pandya/face_ai_project/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python" alt="Python">
  <img src="https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow" alt="TensorFlow">
  <img src="https://img.shields.io/badge/OpenCV-4.x-green?style=for-the-badge&logo=opencv" alt="OpenCV">
  <img src="https://img.shields.io/badge/YOLOv8-Ultralytics-purple?style=for-the-badge" alt="YOLO">
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge" alt="License">
</p>

<p align="center">
  <strong>Real-time Computer Vision Application for Face Detection, Mask Detection, Emotion Recognition, Hand Gesture Recognition, and Object Detection</strong>
</p>

---

## 📋 Table of Contents

- [Features](#-features)
- [Quick Start](#-quick-start)
- [Interpreter Configuration](#-interpreter-configuration)
- [Models](#-models)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Training](#-training-your-own-models)
- [Testing](#-testing)
- [Known Limitations & Roadmap](#-known-limitations--roadmap)
- [Credits](#-credits)
- [License](#-license)

---

## 🌟 Features

| Feature | Description | Status |
|---------|-------------|--------|
| **Face Detection** | Real-time face detection using Haar Cascade classifier | ✅ |
| **Mask Detection** | Binary classification: MASK / NO MASK with temporal confidence smoothing | ✅ |
| **Emotion Recognition** | 7-class emotion detection (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral) | ✅ |
| **Hand Gesture Recognition** | MediaPipe-based hand tracking with gesture recognition and auto-capture | ✅ |
| **Object Detection** | YOLOv8-powered real-time object detection (80 COCO classes) | ✅ |
| **Control Panels** | Tkinter-based control panel launcher and HUD statistics | ✅ |

<!--
### Demo Screenshots
<!-- TODO: Add actual screenshot images or demo GIF once available -->
<!-- ![Mask Detection](assets/screenshots/mask_detection_demo.jpg) -->
<!-- ![Emotion Detection](assets/screenshots/emotion_happy.jpg) -->
<!-- ![GUI Control Panel](assets/screenshots/gui_main.jpg) -->
-->

---

## 🚀 Quick Start

Dependencies are split by feature so you only install what you need for your use case:

### 1. Core Vision (Mask, Emotion, Combined Detection)

Clone the repository and install core dependencies:

```bash
git clone https://github.com/Rajat-Pandya/face_ai_project.git
cd face_ai_project

# Create and activate a virtual environment
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install core requirements
pip install -r requirements.txt
```

Launch the application:
```bash
python main.py
```

### 2. Hand Gesture Recognition (Optional)

To enable MediaPipe hand tracking and gesture detection:

```bash
pip install -r requirements-hand.txt
```

Launch hand gesture mode:
```bash
python main.py hand
```

### 3. Object Detection with YOLOv8 (Optional)

To enable YOLOv8 real-time object detection:

```bash
pip install -r requirements-yolo.txt
```

Launch object detection:
```bash
python main.py yolo
```

---

## ⚙️ Interpreter Configuration

The primary launcher (`main.py`) and the control panel launcher (`gui/app.py`) spawn detection tasks as independent subprocesses.

By default, every subprocess uses the current Python interpreter (`sys.executable`).

If you maintain separate virtual environments for different deep learning toolkits (e.g., to isolate MediaPipe or PyTorch from TensorFlow), you can point the launchers to specific interpreters using environment variables:

| Environment Variable | Target Feature | Default |
|----------------------|----------------|---------|
| `FACE_AI_PYTHON` | Mask detection, emotion recognition, and GUI panel | Current Python interpreter (`sys.executable`) |
| `HAND_AI_PYTHON` | Hand gesture tracking (`src/hand_detection.py`) | Current Python interpreter (`sys.executable`) |
| `YOLO_AI_PYTHON` | Object detection (`src/object_detection.py`) | Current Python interpreter (`sys.executable`) |

### Example: Custom Environment Setup

On Windows (PowerShell):
```powershell
$env:FACE_AI_PYTHON = "C:\path\to\envs\face_env\Scripts\python.exe"
$env:HAND_AI_PYTHON = "C:\path\to\envs\hand_env\Scripts\python.exe"
$env:YOLO_AI_PYTHON = "C:\path\to\envs\yolo_env\Scripts\python.exe"
python main.py
```

On Linux/macOS:
```bash
export FACE_AI_PYTHON="/path/to/envs/face_env/bin/python"
export HAND_AI_PYTHON="/path/to/envs/hand_env/bin/python"
export YOLO_AI_PYTHON="/path/to/envs/yolo_env/bin/python"
python main.py
```

---

## 🧠 Models

- **Pre-trained Keras Models**:
  - `models/mask_model.h5`: MobileNetV2 transfer-learned model (224x224 RGB input) for binary mask classification. Tracked directly in this repository.
  - `models/emotion_model.h5`: 3-block CNN (48x48 grayscale input) for 7-class emotion recognition. Tracked directly in this repository.
- **YOLOv8 Weights**:
  - `yolov8s.pt` is **not** tracked in this repository.
  - When running YOLO object detection for the first time, Ultralytics automatically downloads `yolov8s.pt` into the project root directory.
  - **License Notice**: Ultralytics YOLOv8 is released under the **AGPL-3.0 License**. Users who incorporate or distribute YOLOv8 must comply with its AGPL-3.0 licensing terms.

---

## 💻 Usage

`main.py` is the primary application launcher.

### Subprocess Launcher (CLI)

```bash
# Launch Tkinter GUI launcher panel
python main.py

# Launch individual modes directly
python main.py mask      # Real-time mask detection
python main.py emotion   # Real-time emotion detection
python main.py hand      # Hand gesture detection and auto-capture
python main.py yolo      # YOLOv8 object detection
```

### Standalone Script Execution

You can also execute each module directly:

```bash
python src/face_detection.py       # Basic Haar face detection
python src/mask_detection.py       # Mask detection
python src/emotion_detection.py   # Emotion detection
python src/real_time_detection.py combined  # Combined mask + emotion HUD
python src/hand_detection.py       # Hand tracking
python src/object_detection.py     # YOLO detection
```

### Interactive Video Controls

- **ESC key (`27`)**: Close active OpenCV stream window and exit.
- **Window Close button**: Safely terminates the stream loop.

---

## 📁 Project Structure

```
face_ai_project/
├── .github/workflows/ci.yml       # Continuous integration pipeline
├── assets/
│   └── screenshots/              # Auto-captured screenshots (ignored from git)
├── dataset/
│   └── README.md                 # Dataset folder structure documentation
├── gui/
│   ├── app.py                    # Multi-process control panel launcher
│   └── interface.py              # Full experimental dashboard with HUD stats
├── models/
│   ├── emotion_model.h5          # Pre-trained emotion model
│   └── mask_model.h5             # Pre-trained mask model
├── src/
│   ├── __init__.py               # Package initialization & interpreter config
│   ├── camera.py                 # Cross-platform camera backend helper
│   ├── emotion_detection.py      # Emotion classification module
│   ├── face_detection.py         # Haar cascade face detection & cropping
│   ├── hand_detection.py         # MediaPipe hand gesture tracker
│   ├── mask_detection.py         # Mask detection & prediction buffer
│   ├── model_loader.py           # Singleton lazy model loader
│   ├── object_detection.py       # YOLOv8 object detector
│   ├── real_time_detection.py    # Combined real-time detection pipeline
│   ├── stats.py                  # Thread-safe telemetry statistics manager
│   ├── train_emotion_model.py    # Emotion model training script
│   └── train_mask_model.py       # Mask model training script
├── tests/
│   ├── conftest.py               # Test stubs for optional ML libraries
│   ├── test_emotion_labels.py    # Emotion labels invariant tests
│   ├── test_face_detection.py    # Face detection & padding boundary tests
│   ├── test_gap002.py            # Mask input size regression tests
│   ├── test_gesture.py           # Gesture classification tests
│   ├── test_mask_predictor.py    # Prediction smoothing and decision rule tests
│   └── test_stats.py             # StatsManager concurrency & contract tests
├── main.py                       # Primary application launcher
├── pytest.ini                    # Pytest configuration
├── requirements.txt              # Core dependencies (OpenCV, TF, etc.)
├── requirements-dev.txt          # Development & test dependencies
├── requirements-hand.txt         # MediaPipe dependency
└── requirements-yolo.txt         # Ultralytics dependency
```

---

## 🎓 Training Your Own Models

Training datasets are **not** bundled with this repository (GAP-003).

To retrain the models:
1. Refer to [dataset/README.md](dataset/README.md) for expected folder layout and class folder names.
2. Populate the training directories with your images.
3. Run the training scripts:
   ```bash
   python src/train_mask_model.py
   python src/train_emotion_model.py
   ```

<!-- TODO: dataset names, sources and licenses -->

---

## 🧪 Testing

Automated test suite using pytest:

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run test suite
pytest
```

> **Note on Test Scope**: Continuous Integration (CI) runs on Ubuntu with Python 3.10, 3.11, and 3.12 without installing heavy GPU/ML frameworks (TensorFlow, MediaPipe, or Ultralytics). The test suite verifies pure application logic, prediction smoothing algorithms, coordinate clipping, statistics telemetry contracts, and label order invariants. Live webcam hardware capture, active neural model inference, MediaPipe hand tracking, and YOLO runtime are not covered by automated CI tests.

---

## ⚠️ Known Limitations & Roadmap

- **Datasets Not Included (GAP-003)**: The training datasets for mask detection and emotion recognition are not distributed with this repository.
- **Primary Launcher**: `main.py` is the primary and recommended launcher. The dashboard in `gui/interface.py` runs OpenCV display loops in background threads and is experimental (GAP-007, GAP-008).
- **Subprocess Telemetry (GAP-009)**: Subprocesses launched from `gui/app.py` run in independent processes and do not stream telemetry back to the GUI stats panel.
- **Combined Annotation Overwriting**: In `combined` mode in `src/real_time_detection.py`, emotion annotations may visually overlay mask annotations for the same bounding box.
- **Haar Cascade Face Detection**: Face localization uses OpenCV Haar Cascades, which may struggle with profile angles, occlusion, or extreme lighting conditions compared to modern deep face detectors.
- **Logging & Configuration**: System messages and runtime telemetry use stdout/stderr prints; structured file logging and external configuration files (`.env` or YAML) are planned.
- **Accessibility**: Graphical interfaces in `gui/` currently lack screen reader metadata, scalable DPI layouts, and keyboard focus hooks.

---

## 🙏 Credits

- **[OpenCV](https://opencv.org/)**: Real-time computer vision library and Haar Cascade classifiers.
- **[TensorFlow / Keras](https://www.tensorflow.org/)**: Deep learning framework and MobileNetV2 ImageNet pre-trained base model weights.
- **[MediaPipe](https://developers.google.com/mediapipe)**: Real-time hand landmark and perception pipeline.
- **[Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)**: Object detection model architecture (licensed under AGPL-3.0).

<!-- TODO: dataset names, sources and licenses -->

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. Note that third-party components (such as Ultralytics YOLOv8) are subject to their respective licenses (e.g., AGPL-3.0).
