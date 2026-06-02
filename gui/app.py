import tkinter as tk
import subprocess

FACE_ENV_PYTHON = r"C:\Users\rajat\anaconda3\envs\final_env\python.exe"
HAND_ENV_PYTHON = r"C:\Users\rajat\anaconda3\envs\hand_env_clean\python.exe"
YOLO_ENV_PYTHON = r"C:\Users\rajat\anaconda3\envs\yolo_env\python.exe"

MASK_SCRIPT = r"src\mask_detection.py"
EMOTION_SCRIPT = r"src\emotion_detection.py"
HAND_SCRIPT = r"src\hand_detection.py"
YOLO_SCRIPT = r"src\object_detection.py"


def run_mask():
    subprocess.Popen([FACE_ENV_PYTHON, MASK_SCRIPT])


def run_emotion():
    subprocess.Popen([FACE_ENV_PYTHON, EMOTION_SCRIPT])


def run_hand():
    subprocess.Popen([HAND_ENV_PYTHON, HAND_SCRIPT])


def run_yolo():
    subprocess.Popen([YOLO_ENV_PYTHON, YOLO_SCRIPT])


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