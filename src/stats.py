import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
os.chdir(project_root)

import threading


class StatsManager:
    def __init__(self):
        self.faces_detected = 0
        self.emotion = "N/A"
        self.mask_status = "N/A"
        self._lock = threading.Lock()
    
    def update_faces(self, count):
        with self._lock:
            self.faces_detected = count
    
    def update_emotion(self, emotion):
        with self._lock:
            self.emotion = emotion
    
    def update_mask(self, status):
        with self._lock:
            self.mask_status = status
    
    def get_stats(self):
        with self._lock:
            return {
                "faces": self.faces_detected,
                "emotion": self.emotion,
                "mask": self.mask_status
            }
    
    def reset(self):
        with self._lock:
            self.faces_detected = 0
            self.emotion = "N/A"
            self.mask_status = "N/A"


stats_manager = StatsManager()


def get_stats_manager():
    return stats_manager
