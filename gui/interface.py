import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
os.chdir(project_root)

import tkinter as tk
from tkinter import messagebox
import threading
import cv2 # type: ignore

from src.model_loader import get_model_loader
from src.stats import get_stats_manager
from src.real_time_detection import run_detection
from src.hand_detection import run_hand_gesture
from src.object_detection import run_yolo_detection


class AIInterface:
    BG_COLOR = "#1e1e1e"
    BTN_COLOR = "#2d89ef"
    BTN_HOVER = "#3a9cf5"
    EXIT_COLOR = "#e81123"
    SUCCESS_COLOR = "#107c10"
    WARNING_COLOR = "#ff8c00"

    def __init__(self, root):
        self.root = root
        self.root.title("Face AI System")
        self.root.geometry("420x620")
        self.root.configure(bg=self.BG_COLOR)
        self.root.resizable(False, False)

        self.active_detection = None
        self.stats = get_stats_manager()
        self.poll_stats = None

        self._setup_ui()
        self._start_stats_poll()

    def _setup_ui(self):
        title = tk.Label(
            self.root,
            text="Face AI Control Panel",
            font=("Arial", 18, "bold"),
            fg="white",
            bg=self.BG_COLOR
        )
        title.pack(pady=(20, 10))

        tk.Frame(self.root, bg="#3a3a3a", height=2).pack(fill=tk.X, padx=30, pady=(0, 15))

        self._create_detection_buttons()
        self._create_control_buttons()
        self._create_stats_panel()
        self._create_exit_button()

        self._check_models()

    def _create_detection_buttons(self):
        btn_config = {
            "width": 22,
            "height": 2,
            "bg": self.BTN_COLOR,
            "fg": "white",
            "font": ("Arial", 11, "bold"),
            "bd": 0,
            "cursor": "hand2",
            "activebackground": self.BTN_HOVER,
            "activeforeground": "white"
        }

        self.mask_btn = tk.Button(
            self.root,
            text="Start Mask Detection",
            command=self.start_mask,
            **btn_config
        )
        self.mask_btn.pack(pady=5)

        self.emotion_btn = tk.Button(
            self.root,
            text="Start Emotion Detection",
            command=self.start_emotion,
            **btn_config
        )
        self.emotion_btn.pack(pady=5)

        self.hand_btn = tk.Button(
            self.root,
            text="Start Hand Gesture",
            command=self.start_hand,
            **btn_config
        )
        self.hand_btn.pack(pady=5)

        self.object_btn = tk.Button(
            self.root,
            text="Start Object Detection",
            command=self.start_object,
            **btn_config
        )
        self.object_btn.pack(pady=5)

        tk.Frame(self.root, bg=self.BG_COLOR, height=10).pack(pady=0)

    def _create_control_buttons(self):
        control_config = {
            "width": 10,
            "height": 1,
            "font": ("Arial", 10, "bold"),
            "bd": 0,
            "cursor": "hand2"
        }

        btn_frame = tk.Frame(self.root, bg=self.BG_COLOR)
        btn_frame.pack(pady=8)

        tk.Button(
            btn_frame,
            text="START",
            bg=self.SUCCESS_COLOR,
            fg="white",
            command=self.start_selected,
            **control_config
        ).pack(side=tk.LEFT, padx=5)

        tk.Button(
            btn_frame,
            text="STOP",
            bg=self.WARNING_COLOR,
            fg="white",
            command=self.stop_detection,
            **control_config
        ).pack(side=tk.LEFT, padx=5)

    def _create_stats_panel(self):
        stats_frame = tk.Frame(self.root, bg="#2a2a2a", padx=20, pady=12)
        stats_frame.pack(fill=tk.X, padx=25, pady=10)

        tk.Label(
            stats_frame,
            text="Statistics",
            font=("Arial", 11, "bold"),
            fg="white",
            bg="#2a2a2a"
        ).pack(anchor=tk.W)

        stats_container = tk.Frame(stats_frame, bg="#2a2a2a")
        stats_container.pack(fill=tk.X, pady=(8, 0))

        self.lbl_faces = tk.Label(
            stats_container,
            text="Faces: 0",
            font=("Arial", 10),
            fg="#00ff00",
            bg="#2a2a2a",
            anchor=tk.W,
            width=18
        )
        self.lbl_faces.grid(row=0, column=0, sticky=tk.W)

        self.lbl_emotion = tk.Label(
            stats_container,
            text="Emotion: N/A",
            font=("Arial", 10),
            fg="#888888",
            bg="#2a2a2a",
            anchor=tk.W,
            width=18
        )
        self.lbl_emotion.grid(row=1, column=0, sticky=tk.W, pady=3)

        self.lbl_mask = tk.Label(
            stats_container,
            text="Mask: N/A",
            font=("Arial", 10),
            fg="#888888",
            bg="#2a2a2a",
            anchor=tk.W,
            width=18
        )
        self.lbl_mask.grid(row=2, column=0, sticky=tk.W, pady=3)

        status_frame = tk.Frame(stats_frame, bg="#2a2a2a")
        status_frame.pack(fill=tk.X, pady=(10, 0))

        self.lbl_status = tk.Label(
            status_frame,
            text="Status: Ready",
            font=("Arial", 10, "bold"),
            fg="#00ff00",
            bg="#2a2a2a"
        )
        self.lbl_status.pack(side=tk.LEFT)

        self.lbl_models = tk.Label(
            status_frame,
            text="Models: Checking...",
            font=("Arial", 9),
            fg="#888888",
            bg="#2a2a2a"
        )
        self.lbl_models.pack(side=tk.RIGHT)

    def _create_exit_button(self):
        tk.Button(
            self.root,
            text="Exit Application",
            command=self.exit_app,
            bg=self.EXIT_COLOR,
            fg="white",
            font=("Arial", 11, "bold"),
            width=22,
            height=2,
            bd=0,
            cursor="hand2",
            activebackground="#c50f1f",
            activeforeground="white"
        ).pack(pady=15)

    def _start_stats_poll(self):
        def poll():
            while self.poll_stats is not False:
                stats = self.stats.get_stats()
                self.root.after(0, lambda s=stats: self._update_stats_display(s))
                self.root.after(200, lambda: None)
                if self.poll_stats is None:
                    break
                import time
                time.sleep(0.5)
        
        self.poll_stats = True
        thread = threading.Thread(target=poll, daemon=True)
        thread.start()

    def _update_stats_display(self, stats):
        if self.active_detection is None:
            return
        
        self.lbl_faces.config(text=f"Faces: {stats['faces']}")
        
        emotion = stats['emotion']
        if emotion != "N/A":
            self.lbl_emotion.config(text=f"Emotion: {emotion}", fg="#ff6600")
        else:
            self.lbl_emotion.config(text="Emotion: N/A", fg="#888888")
        
        mask = stats['mask']
        if mask == "MASK":
            self.lbl_mask.config(text=f"Mask: {mask}", fg="#00ff00")
        elif mask == "NO MASK":
            self.lbl_mask.config(text=f"Mask: {mask}", fg="#ff0000")
        else:
            self.lbl_mask.config(text="Mask: N/A", fg="#888888")

    def _check_models(self):
        loader = get_model_loader()
        mask = "Loaded" if loader.mask_model else "Missing"
        emotion = "Loaded" if loader.emotion_model else "Missing"
        self.lbl_models.config(text=f"Mask:{mask} | Emo:{emotion}")

    def _set_buttons_state(self, state):
        state_map = {"normal": tk.NORMAL, "disabled": tk.DISABLED}
        for btn in [self.mask_btn, self.emotion_btn, self.hand_btn, self.object_btn]:
            btn.config(state=state_map.get(state, tk.NORMAL))

    def update_status(self, text, color="#00ff00"):
        self.lbl_status.config(text=f"Status: {text}", fg=color)

    def start_mask(self):
        self.active_detection = "mask"
        self._set_buttons_state("disabled")
        self.update_status("Mask Active")
        
        def run():
            run_detection(mode="mask")
            self.root.after(0, self._detection_ended)
        
        thread = threading.Thread(target=run, daemon=True)
        thread.start()

    def start_emotion(self):
        self.active_detection = "emotion"
        self._set_buttons_state("disabled")
        self.update_status("Emotion Active")
        
        def run():
            run_detection(mode="emotion")
            self.root.after(0, self._detection_ended)
        
        thread = threading.Thread(target=run, daemon=True)
        thread.start()

    def start_hand(self):
        self.active_detection = "hand"
        self._set_buttons_state("disabled")
        self.update_status("Hand Active")
        
        def run():
            run_hand_gesture()
            self.root.after(0, self._detection_ended)
        
        thread = threading.Thread(target=run, daemon=True)
        thread.start()

    def start_object(self):
        self.active_detection = "object"
        self._set_buttons_state("disabled")
        self.update_status("Object Active")
        
        def run():
            run_yolo_detection()
            self.root.after(0, self._detection_ended)
        
        thread = threading.Thread(target=run, daemon=True)
        thread.start()

    def start_selected(self):
        mode = messagebox.askstring(
            "Select Mode",
            "Enter: mask, emotion, hand, or object"
        )
        methods = {
            "mask": self.start_mask,
            "emotion": self.start_emotion,
            "hand": self.start_hand,
            "object": self.start_object
        }
        if mode and mode.lower() in methods:
            methods[mode.lower()]()

    def _detection_ended(self):
        self.active_detection = None
        self.update_status("Ready")
        self._set_buttons_state("normal")

    def stop_detection(self):
        cv2.destroyAllWindows()
        self.update_status("Stopped", self.WARNING_COLOR)
        self.active_detection = None
        self._set_buttons_state("normal")

    def exit_app(self):
        if messagebox.askyesno("Exit", "Exit application?"):
            self.poll_stats = False
            cv2.destroyAllWindows()
            self.root.destroy()


def main():
    root = tk.Tk()
    app = AIInterface(root)
    root.mainloop()


if __name__ == "__main__":
    main()