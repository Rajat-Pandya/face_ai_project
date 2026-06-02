import sys
import os
import subprocess

FACE_ENV_PYTHON = r"C:\Users\rajat\anaconda3\envs\final_env\python.exe"
HAND_ENV_PYTHON = r"C:\Users\rajat\anaconda3\envs\hand_env_clean\python.exe"
YOLO_ENV_PYTHON = r"C:\Users\rajat\anaconda3\envs\yolo_env\python.exe"

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def main():
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        
        if mode == "mask":
            subprocess.Popen([FACE_ENV_PYTHON, os.path.join(PROJECT_ROOT, "src", "mask_detection.py")])
        elif mode == "emotion":
            subprocess.Popen([FACE_ENV_PYTHON, os.path.join(PROJECT_ROOT, "src", "emotion_detection.py")])
        elif mode == "hand":
            subprocess.Popen([HAND_ENV_PYTHON, os.path.join(PROJECT_ROOT, "src", "hand_detection.py")])
        elif mode == "yolo":
            subprocess.Popen([YOLO_ENV_PYTHON, os.path.join(PROJECT_ROOT, "src", "object_detection.py")])
        else:
            print(f"Unknown mode: {mode}")
            print("Usage: python main.py [mask|emotion|hand|yolo]")
    else:
        # Run GUI in face env (has tkinter)
        subprocess.Popen([FACE_ENV_PYTHON, os.path.join(PROJECT_ROOT, "gui", "app.py")])


if __name__ == "__main__":
    main()