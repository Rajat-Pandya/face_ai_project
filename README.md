# ?? Face AI System

<p align="center">
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

## ?? Table of Contents

- [Features](#-features)
- [Demo](#-demo)
- [Technologies](#-technologies-used)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Training](#-training-your-own-models)
- [API Reference](#-api-reference)
- [Performance](#-performance)
- [Troubleshooting](#-troubleshooting)
- [Future Improvements](#-future-improvements)
- [Contributing](#-contributing)
- [License](#-license)
- [Author](#-author)

---

## ? Features

| Feature | Description | Status |
|---------|-------------|--------|
| **Face Detection** | Real-time face detection using Haar Cascade classifier | ? |
| **Mask Detection** | Binary classification: MASK / NO MASK with confidence score | ? |
| **Emotion Recognition** | 7-class emotion detection (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral) | ? |
| **Hand Gesture Recognition** | MediaPipe-based hand tracking with gesture recognition | ? |
| **Object Detection** | YOLOv8-powered real-time object detection (80 COCO classes) | ? |
| **Modern UI** | Enhanced real-time display with FPS counter and confidence percentages | ? |
| **Error Handling** | Comprehensive error handling for webcam, models, and datasets | ? |

### Key Features

- ?? **Real-time Processing**: Optimized for 15-30 FPS depending on mode
- ?? **Modern Visualizations**: Corner-style bounding boxes with confidence display
- ?? **Live Statistics**: FPS counter, face count, detection status
- ?? **Frame Skipping**: Configurable frame skipping for performance optimization
- ??? **Robust Error Handling**: Graceful degradation when models/camera unavailable
- ??? **GUI Control Panel**: Tkinter-based interface for easy control

---

## ?? Demo

### Real-time Mask Detection
```
+-----------------------------------------+
¦         Mask Detection                  ¦
¦                                         ¦
¦    +-----+                              ¦
¦    ¦ ??  ¦  MASK (98.5%)                ¦
¦    +-----+                              ¦
¦                                         ¦
¦  +-----------------+                    ¦
¦  ¦ FPS: 24         ¦                    ¦
¦  ¦ Faces: 1        ¦                    ¦
¦  ¦ Emotion: N/A    ¦                    ¦
¦  ¦ Mask: MASK      ¦                    ¦
¦  +-----------------+                    ¦
¦                                         ¦
+-----------------------------------------+
```

### Real-time Emotion Detection
```
+-----------------------------------------+
¦       Emotion Detection                 ¦
¦                                         ¦
¦    +-----+                              ¦
¦    ¦ ??  ¦  Happy (94.2%)                ¦
¦    +-----+                              ¦
¦                                         ¦
¦  +-----------------+                    ¦
¦  ¦ FPS: 22         ¦                    ¦
¦  ¦ Faces: 1        ¦                    ¦
¦  ¦ Emotion: Happy  ¦                    ¦
¦  ¦ Mask: N/A       ¦                    ¦
¦  +-----------------+                    ¦
¦                                         ¦
+-----------------------------------------+
```

---

## ?? Technologies Used

- **Python 3.10+** - Core programming language
- **TensorFlow/Keras 2.x** - Deep learning framework for mask and emotion models
- **OpenCV 4.x** - Computer vision and image processing
- **MediaPipe** - Hand tracking and gesture recognition
- **Ultralytics YOLOv8** - State-of-the-art object detection
- **NumPy** - Numerical computing
- **Tkinter** - GUI framework

### Model Architectures

| Model | Architecture | Input Size | Classes |
|-------|-------------|------------|---------|
| Mask Detection | MobileNetV2 (Transfer Learning) | 224x224 | 2 (With/Without Mask) |
| Emotion Detection | Custom CNN | 48x48 | 7 (Emotions) |
| Object Detection | YOLOv8s | 640x640 | 80 (COCO) |

---

## ?? Installation

### Prerequisites

- Python 3.10 or higher
- Webcam (for real-time detection)
- 4GB+ RAM recommended
- CUDA-compatible GPU (optional, for faster inference)

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/face-ai-project.git
cd face-ai-project
```

### 2. Create Virtual Environment

Using conda:
```bash
conda create -n face_ai python=3.10
conda activate face_ai
```

Using venv:
```bash
python -m venv face_ai
# Windows:
face_ai\Scripts\activate
# Linux/Mac:
source face_ai/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download YOLO Weights (Optional)

```python
from ultralytics import YOLO
# This will automatically download yolov8s.pt on first run
model = YOLO("yolov8s.pt")
```

---

## ?? Usage

### Run GUI Application

```bash
python main.py
```

The GUI provides:
- **Start Detection**: Launch real-time detection windows
- **Mode Selection**: Choose between Mask, Emotion, Hand, or Object detection
- **Live Statistics**: Real-time face count, emotion, and mask status
- **Model Status**: Shows which models are loaded

### Command Line Usage

```bash
# Run GUI (default)
python main.py

# Run specific detection modes
python main.py mask      # Mask detection
python main.py emotion   # Emotion detection
python main.py hand      # Hand gesture recognition
python main.py yolo      # Object detection
```

### Run Individual Modules

```bash
# Face detection only
python src/face_detection.py

# Mask detection
python src/mask_detection.py

# Emotion detection
python src/emotion_detection.py

# Hand gesture recognition
python src/hand_detection.py

# Object detection (requires yolov8s.pt)
python src/object_detection.py
```

### Keyboard Controls

| Key | Action |
|-----|--------|
| `ESC` | Exit detection window |
| Close Window | Stop detection |

---

## ?? Project Structure

```
face_ai_project/
¦
+-- ?? assets/                    # Project assets
¦   +-- screenshots/              # Demo screenshots
¦
+-- ?? dataset/                   # Training datasets
¦   +-- mask_dataset/            # Mask detection dataset
¦   ¦   +-- with_mask/
¦   ¦   +-- without_mask/
¦   +-- emotion_dataset/         # Emotion recognition dataset
¦       +-- angry/
¦       +-- disgust/
¦       +-- fear/
¦       +-- happy/
¦       +-- neutral/
¦       +-- sad/
¦       +-- surprise/
¦
+-- ?? gui/                       # GUI components
¦   +-- app.py                   # Simple GUI launcher
¦   +-- interface.py             # Full-featured Tkinter interface
¦
+-- ?? models/                    # Trained models
¦   +-- mask_model.h5           # Mask detection model
¦   +-- emotion_model.h5        # Emotion recognition model
¦
+-- ?? src/                       # Source code
¦   +-- __init__.py             # Package initialization
¦   +-- face_detection.py       # Haar Cascade face detector
¦   +-- mask_detection.py        # Mask classification module
¦   +-- emotion_detection.py    # Emotion recognition module
¦   +-- hand_detection.py       # MediaPipe hand gesture recognition
¦   +-- object_detection.py     # YOLOv8 object detection
¦   +-- real_time_detection.py  # Enhanced unified real-time detector
¦   +-- model_loader.py         # Singleton model loader with caching
¦   +-- stats.py                # Statistics manager for GUI
¦   +-- train_mask_model.py     # Mask model training script
¦   +-- train_emotion_model.py  # Emotion model training script
¦
+-- .gitignore                   # Git ignore rules
+-- main.py                      # Application entry point
+-- requirements.txt             # Python dependencies
+-- README.md                    # Project documentation
+-- RUNTIME_AUDIT.md            # Runtime verification report
```

---

## ?? Training Your Own Models

### Mask Detection Model

1. **Prepare Dataset**
   ```
   dataset/mask_dataset/
   +-- with_mask/        # Place images with masks here
   +-- without_mask/   # Place images without masks here
   ```

2. **Train Model**
   ```bash
   python src/train_mask_model.py
   ```

3. **Model Details**
   - Architecture: MobileNetV2 with transfer learning
   - Input: 224x224 RGB images
   - Output: 2 classes (With/Without Mask)
   - Training: 15 epochs with data augmentation

### Emotion Detection Model

1. **Prepare Dataset**
   ```
   dataset/emotion_dataset/
   +-- angry/
   +-- disgust/
   +-- fear/
   +-- happy/
   +-- neutral/
   +-- sad/
   +-- surprise/
   ```

2. **Train Model**
   ```bash
   python src/train_emotion_model.py
   ```

3. **Model Details**
   - Architecture: Custom CNN (3 convolutional blocks)
   - Input: 48x48 grayscale images
   - Output: 7 emotion classes
   - Training: 25 epochs with dropout regularization

---

## ?? API Reference

### Model Loader (Singleton Pattern)

```python
from src.model_loader import get_model_loader

loader = get_model_loader()
mask_model = loader.mask_model      # Lazy loaded
emotion_model = loader.emotion_model # Lazy loaded
```

### Real-time Detection

```python
from src.real_time_detection import run_detection

# Run mask detection
run_detection(mode="mask", camera_index=0, skip_frames=2)

# Run emotion detection
run_detection(mode="emotion", camera_index=0, skip_frames=2)
```

### Statistics Manager

```python
from src.stats import get_stats_manager

stats = get_stats_manager()
stats.update_faces(1)
stats.update_emotion("Happy")
stats.update_mask("MASK")
current_stats = stats.get_stats()
```

---

## ? Performance

| Module | Approximate FPS | Notes |
|--------|----------------|-------|
| Face Detection | 25-30 FPS | Multi-scale Haar Cascade |
| Mask Detection | 18-24 FPS | With frame skipping (skip_frames=2) |
| Emotion Detection | 18-24 FPS | With frame skipping (skip_frames=2) |
| Hand Gesture | 20-25 FPS | MediaPipe optimization |
| Object Detection (YOLO) | 15-20 FPS | YOLOv8s model |

*Performance measured on Intel i5/RTX 3060. Results may vary based on hardware.*

### Optimization Techniques

1. **Frame Skipping**: Process every Nth frame, reuse predictions for intermediate frames
2. **Multi-scale Detection**: Start with downscaled image, fallback to full resolution
3. **Prediction Buffering**: Average predictions over 3 frames for stability
4. **Singleton Model Loading**: Models loaded once, reused throughout session

---

## ?? Troubleshooting

### Camera Not Available

```python
# Try different camera index
cap = cv2.VideoCapture(1)  # Second camera
```

**Common Causes:**
- Camera is being used by another application
- Camera drivers not installed
- Wrong camera index

### Model Not Found

```
Error: Mask model not found at models/mask_model.h5
```

**Solutions:**
1. Train the model: `python src/train_mask_model.py`
2. Check models are in the `models/` directory
3. Verify file permissions

### Import Errors

```bash
# Ensure all dependencies are installed
pip install -r requirements.txt

# If TensorFlow issues:
pip install --upgrade tensorflow
```

### Low FPS

**Solutions:**
- Increase `skip_frames` parameter (try 3-5)
- Close other applications using GPU
- Reduce camera resolution in code
- Use CPU-only TensorFlow if GPU memory insufficient

---

## ?? Future Improvements

- [ ] Replace Haar Cascade with MTCNN or RetinaFace for better accuracy
- [ ] Add face recognition/authentication capabilities
- [ ] Implement age and gender detection
- [ ] Multi-face tracking across frames
- [ ] Edge device optimization (Raspberry Pi, Jetson Nano)
- [ ] Web-based interface using Flask or Streamlit
- [ ] Mobile app version using TensorFlow Lite
- [ ] Add pose detection with MediaPipe
- [ ] Docker containerization for easy deployment
- [ ] CI/CD pipeline for automated testing

---

## ?? Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m '"'"'Add some AmazingFeature'"'"'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

Please ensure your code follows the existing style and includes appropriate tests.

---

## ?? License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## ????? Author

**Your Name**
- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [Your Profile](https://linkedin.com/in/yourprofile)
- Email: your.email@example.com

---

## ?? Acknowledgments

- [OpenCV](https://opencv.org/) for computer vision tools
- [TensorFlow](https://tensorflow.org/) and [Keras](https://keras.io/) for deep learning
- [MediaPipe](https://mediapipe.dev/) for hand tracking
- [Ultralytics](https://ultralytics.com/) for YOLO models
- OpenCV Haar Cascade classifiers
- FER-2013 dataset for emotion recognition research

---

<p align="center">
  ? Star this repository if you find it helpful!
</p>

<p align="center">
  Made with ?? and Python
</p>
