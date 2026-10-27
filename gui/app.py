import os
import subprocess
import sys
import tkinter as tk

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src import FACE_AI_PYTHON, HAND_AI_PYTHON, YOLO_AI_PYTHON

MASK_SCRIPT = os.path.join(PROJECT_ROOT, "src", "mask_detection.py")
EMOTION_SCRIPT = os.path.join(PROJECT_ROOT, "src", "emotion_detection.py")
HAND_SCRIPT = os.path.join(PROJECT_ROOT, "src", "hand_detection.py")
YOLO_SCRIPT = os.path.join(PROJECT_ROOT, "src", "object_detection.py")


def run_mask():
    subprocess.Popen([FACE_AI_PYTHON, MASK_SCRIPT])


def run_emotion():
    subprocess.Popen([FACE_AI_PYTHON, EMOTION_SCRIPT])


def run_hand():
    subprocess.Popen([HAND_AI_PYTHON, HAND_SCRIPT])


def run_yolo():
    subprocess.Popen([YOLO_AI_PYTHON, YOLO_SCRIPT])


root = tk.Tk()
root.title("Face AI System")
root.geometry("400x400")
root.configure(bg="#1e1e1e")

title = tk.Label(
    root,
    text="Face AI Control Panel",
    font=("Arial", 16, "bold"),
    fg="white",
    bg="#1e1e1e"
)
title.pack(pady=20)

btn_style = {
    "width": 25,
    "height": 2,
    "bg": "#2d89ef",
    "fg": "white",
    "font": ("Arial", 11, "bold"),
    "bd": 0
}

tk.Button(root, text="Mask Detection", command=run_mask, **btn_style).pack(pady=8)
tk.Button(root, text="Emotion Detection", command=run_emotion, **btn_style).pack(pady=8)
tk.Button(root, text="Hand Gesture", command=run_hand, **btn_style).pack(pady=8)
tk.Button(root, text="Object Detection (YOLO)", command=run_yolo, **btn_style).pack(pady=8)

tk.Button(
    root,
    text="Exit",
    command=root.destroy,
    bg="#e81123",
    fg="white",
    font=("Arial", 11, "bold"),
    width=25,
    height=2,
    bd=0
).pack(pady=20)

root.mainloop()
