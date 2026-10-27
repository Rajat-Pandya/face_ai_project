import os
import sys

FACE_AI_PYTHON = os.environ.get("FACE_AI_PYTHON", sys.executable)
HAND_AI_PYTHON = os.environ.get("HAND_AI_PYTHON", sys.executable)
YOLO_AI_PYTHON = os.environ.get("YOLO_AI_PYTHON", sys.executable)

# Only import non-MediaPipe modules here
from .face_detection import detect_faces
from .mask_detection import run_mask_detection
from .emotion_detection import run_emotion_detection

# Don't import hand_detection here - mediapipe compatibility issue
# from .hand_detection import run_hand_gesture
# from .object_detection import run_yolo_detection
