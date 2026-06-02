"""
Real-time Face Detection with Mask and Emotion Recognition
Enhanced version with FPS counter, confidence display, and error handling
"""
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

import cv2
import numpy as np
from collections import deque
import time
from typing import Optional, Tuple, List
from src.model_loader import get_model_loader
from src.stats import get_stats_manager


class DetectionError(Exception):
    """Custom exception for detection errors"""
    pass


class RealtimeDetector:
    """
    Real-time detector supporting mask detection and emotion recognition.
    Features: FPS counter, confidence display, smooth overlays, error handling
    """
    EMOTION_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
    IMG_SIZE = 160

    # Color scheme
    COLOR_MASK = (0, 255, 0)          # Green for mask
    COLOR_NO_MASK = (0, 0, 255)       # Red for no mask
    COLOR_EMOTION = (255, 165, 0)     # Orange for emotion
    COLOR_BOX_DEFAULT = (0, 255, 255) # Cyan for default
    COLOR_TEXT_BG = (30, 30, 30)      # Dark gray for text background
    COLOR_FPS = (0, 255, 255)         # Cyan for FPS
    COLOR_INFO = (220, 220, 220)      # Light gray for info text

    def __init__(self, skip_frames: int = 2, buffer_size: int = 3):
        """
        Initialize the real-time detector.
        
        Args:
            skip_frames: Number of frames to skip between processing
            buffer_size: Size of prediction buffer for smoothing
        """
        self.face_cascade = self._load_face_cascade()
        self.loader = get_model_loader()
        self.stats = get_stats_manager()
        
        self.mask_buffer = deque(maxlen=buffer_size)
        self.emotion_buffer = deque(maxlen=buffer_size)
        self.skip_frames = skip_frames
        self.frame_count = 0
        
        self.last_faces: List = []
        self.last_predictions: dict = {}
        
        # FPS calculation
        self.fps_start_time = time.time()
        self.fps_frame_count = 0
        self.current_fps = 0

    def _load_face_cascade(self):
        """Load Haar Cascade face detector with error handling"""
        try:
            cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            if cascade.empty():
                raise DetectionError("Failed to load Haar Cascade classifier")
            return cascade
        except Exception as e:
            raise DetectionError(f"Face cascade loading error: {e}")

    @property
    def mask_model(self):
        """Lazy load mask model"""
        return self.loader.mask_model

    @property
    def emotion_model(self):
        """Lazy load emotion model"""
        return self.loader.emotion_model

    def detect_faces(self, frame):
        """
        Detect faces in frame using multi-scale approach.
        
        Args:
            frame: Input BGR image
            
        Returns:
            Array of face bounding boxes (x, y, w, h)
        """
        if frame is None or frame.size == 0:
            return np.array([])
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # First try on downscaled image for performance
        small = cv2.resize(gray, (320, 240))
        
        faces = self.face_cascade.detectMultiScale(
            small,
            scaleFactor=1.15,
            minNeighbors=5,
            minSize=(40, 40)
        )
        
        # Fallback to full resolution if no faces found
        if len(faces) == 0:
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.25,
                minNeighbors=3,
                minSize=(30, 30)
            )
        
        # Scale coordinates back to original frame size
        if len(faces) > 0:
            scale_x = frame.shape[1] / 320
            scale_y = frame.shape[0] / 240
            faces = faces * np.array([scale_x, scale_y, scale_x, scale_y])
            faces = faces.astype(int)
        
        return faces

    def predict_mask(self, face_roi):
        """
        Predict mask status with confidence.
        
        Args:
            face_roi: Face region of interest
            
        Returns:
            Tuple of (label, confidence)
        """
        if self.mask_model is None or face_roi is None or face_roi.size == 0:
            return None, 0.0
        
        try:
            face_resized = cv2.resize(face_roi, (self.IMG_SIZE, self.IMG_SIZE))
            face_array = np.array(face_resized, dtype=np.float32) / 255.0
            face_array = face_array.reshape(1, self.IMG_SIZE, self.IMG_SIZE, 3)
            
            pred = self.mask_model.predict(face_array, verbose=0)
            
            self.mask_buffer.append(pred[0])
            avg_pred = np.mean(self.mask_buffer, axis=0)
            
            label = "MASK" if avg_pred[0] > avg_pred[1] else "NO MASK"
            confidence = float(np.max(avg_pred)) * 100
            
            return label, confidence
        except Exception as e:
            print(f"Mask prediction error: {e}")
            return None, 0.0

    def predict_emotion(self, face_roi):
        """
        Predict emotion with confidence.
        
        Args:
            face_roi: Face region of interest
            
        Returns:
            Tuple of (emotion, confidence)
        """
        if self.emotion_model is None or face_roi is None or face_roi.size == 0:
            return None, 0.0
        
        try:
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            gray_resized = cv2.resize(gray, (48, 48))
            gray_array = np.array(gray_resized, dtype=np.float32) / 255.0
            gray_array = gray_array.reshape(1, 48, 48, 1)
            
            pred = self.emotion_model.predict(gray_array, verbose=0)
            
            self.emotion_buffer.append(pred[0])
            avg_pred = np.mean(self.emotion_buffer, axis=0)
            
            emotion_idx = np.argmax(avg_pred)
            emotion = self.EMOTION_LABELS[emotion_idx]
            confidence = float(np.max(avg_pred)) * 100
            
            return emotion, confidence
        except Exception as e:
            print(f"Emotion prediction error: {e}")
            return None, 0.0

    def update_fps(self):
        """Update and return current FPS"""
        self.fps_frame_count += 1
        elapsed = time.time() - self.fps_start_time
        
        if elapsed >= 1.0:
            self.current_fps = self.fps_frame_count
            self.fps_frame_count = 0
            self.fps_start_time = time.time()
        
        return self.current_fps

    @staticmethod
    def add_padding(bbox, frame_shape, padding=15):
        """Add padding to bounding box with bounds checking"""
        x, y, w, h = bbox
        y1 = max(0, y - padding)
        y2 = min(frame_shape[0], y + h + padding)
        x1 = max(0, x - padding)
        x2 = min(frame_shape[1], x + w + padding)
        return x1, y1, x2, y2

    def draw_modern_bbox(self, frame, x1, y1, x2, y2, color, thickness=2, corner_len=20):
        """Draw modern corner-style bounding box"""
        # Draw corners
        cv2.line(frame, (x1, y1), (x1 + corner_len, y1), color, thickness)
        cv2.line(frame, (x1, y1), (x1, y1 + corner_len), color, thickness)
        cv2.line(frame, (x2, y1), (x2 - corner_len, y1), color, thickness)
        cv2.line(frame, (x2, y1), (x2, y1 + corner_len), color, thickness)
        cv2.line(frame, (x1, y2), (x1 + corner_len, y2), color, thickness)
        cv2.line(frame, (x1, y2), (x1, y2 - corner_len), color, thickness)
        cv2.line(frame, (x2, y2), (x2 - corner_len, y2), color, thickness)
        cv2.line(frame, (x2, y2), (x2, y2 - corner_len), color, thickness)

    def draw_label_with_confidence(self, frame, label, confidence, x, y, color):
        """Draw label with confidence percentage using modern styling"""
        text = f"{label}"
        conf_text = f"{confidence:.1f}%"
        
        # Calculate text sizes
        (label_w, label_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        (conf_w, conf_h), _ = cv2.getTextSize(conf_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        
        total_w = max(label_w, conf_w) + 15
        total_h = label_h + conf_h + 15
        
        # Draw background with rounded corners effect
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y - total_h - 5), (x + total_w, y), color, -1)
        cv2.addWeighted(overlay, 0.9, frame, 0.1, 0, frame)
        
        # Draw label
        cv2.putText(frame, text, (x + 5, y - total_h + label_h), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        
        # Draw confidence
        cv2.putText(frame, conf_text, (x + 5, y - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)

    def draw_stats_panel(self, frame, faces_count, emotion, mask_status, fps):
        """Draw modern stats panel overlay"""
        panel_w = 200
        panel_h = 140
        margin = 10
        
        # Panel background
        overlay = frame.copy()
        cv2.rectangle(overlay, 
                     (frame.shape[1] - panel_w - margin, margin), 
                     (frame.shape[1] - margin, margin + panel_h), 
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Border
        cv2.rectangle(frame, 
                     (frame.shape[1] - panel_w - margin, margin), 
                     (frame.shape[1] - margin, margin + panel_h), 
                     (100, 100, 100), 1)
        
        y_pos = margin + 25
        line_height = 28
        x_pos = frame.shape[1] - panel_w - margin + 10
        
        # FPS with color coding
        fps_color = self.COLOR_FPS if fps >= 20 else (0, 165, 255) if fps >= 10 else (0, 0, 255)
        cv2.putText(frame, f"FPS: {fps}", (x_pos, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, fps_color, 2)
        
        y_pos += line_height
        cv2.putText(frame, f"Faces: {faces_count}", (x_pos, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.COLOR_INFO, 1)
        
        y_pos += line_height
        emotion_color = self.COLOR_EMOTION if emotion != "N/A" else (100, 100, 100)
        cv2.putText(frame, f"Emotion: {emotion}", (x_pos, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.55, emotion_color, 1)
        
        y_pos += line_height
        if mask_status == "MASK":
            mask_color = self.COLOR_MASK
        elif mask_status == "NO MASK":
            mask_color = self.COLOR_NO_MASK
        else:
            mask_color = (100, 100, 100)
        cv2.putText(frame, f"Mask: {mask_status}", (x_pos, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.55, mask_color, 1)

    def draw_instructions(self, frame):
        """Draw keyboard instructions at bottom"""
        instructions = "ESC: Exit | Window Close: Stop"
        (text_w, text_h), _ = cv2.getTextSize(instructions, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        
        x = (frame.shape[1] - text_w) // 2
        y = frame.shape[0] - 15
        
        cv2.putText(frame, instructions, (x, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    def should_process(self):
        """Check if this frame should be processed"""
        self.frame_count += 1
        return self.frame_count % (self.skip_frames + 1) == 0


def check_webcam_available(camera_index=0):
    """Check if webcam is available"""
    try:
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW if os.name == "nt" else 0)
        if cap.isOpened():
            cap.release()
            return True
        return False
    except Exception:
        return False


def check_model_available(model_name):
    """Check if a model file exists"""
    model_paths = {
        "mask": "models/mask_model.h5",
        "emotion": "models/emotion_model.h5"
    }
    return os.path.exists(model_paths.get(model_name, ""))


def run_detection(mode="mask", camera_index=0, skip_frames=2):
    """
    Run real-time detection with comprehensive error handling.
    
    Args:
        mode: Detection mode - "mask", "emotion", or "combined"
        camera_index: Camera device index
        skip_frames: Frames to skip for performance
    """
    # Pre-flight checks
    if not check_webcam_available(camera_index):
        print("ERROR: Webcam not available. Please check:")
        print("  - Camera is connected")
        print("  - No other application is using the camera")
        print("  - Camera drivers are installed")
        return

    if mode == "mask" and not check_model_available("mask"):
        print("ERROR: Mask model not found at models/mask_model.h5")
        print("  Run: python src/train_mask_model.py")
        return

    if mode == "emotion" and not check_model_available("emotion"):
        print("ERROR: Emotion model not found at models/emotion_model.h5")
        print("  Run: python src/train_emotion_model.py")
        return

    try:
        detector = RealtimeDetector(skip_frames=skip_frames)
    except DetectionError as e:
        print(f"ERROR: Failed to initialize detector: {e}")
        return
    except Exception as e:
        print(f"ERROR: Unexpected error during initialization: {e}")
        return

    stats = get_stats_manager()
    stats.reset()
    
    # Model availability check
    if mode == "mask" and detector.mask_model is None:
        print("ERROR: Failed to load mask model")
        return
    if mode == "emotion" and detector.emotion_model is None:
        print("ERROR: Failed to load emotion model")
        return

    # Initialize camera
    try:
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW if os.name == "nt" else 0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        if not cap.isOpened():
            print("ERROR: Could not open camera")
            return
    except Exception as e:
        print(f"ERROR: Camera initialization failed: {e}")
        return

    # Window configuration
    mode_config = {
        "mask": ("Mask Detection", detector.COLOR_MASK),
        "emotion": ("Emotion Detection", detector.COLOR_EMOTION),
        "combined": ("Face AI - Combined", detector.COLOR_BOX_DEFAULT)
    }
    window_name, label_color = mode_config.get(mode, ("Detection", detector.COLOR_BOX_DEFAULT))

    # State variables
    current_fps = 0
    last_emotion = "N/A"
    last_mask = "N/A"
    last_confidence = 0.0

    print(f"Starting {window_name}...")
    print("Press ESC to exit")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("WARNING: Failed to capture frame")
                continue

            # Update FPS
            current_fps = detector.update_fps()

            # Detect faces
            if detector.should_process():
                faces = detector.detect_faces(frame)
                detector.last_faces = faces
            else:
                faces = detector.last_faces

            faces_count = len(faces)

            # Process each face
            for i, (x, y, w, h) in enumerate(faces):
                x1, y1, x2, y2 = detector.add_padding((x, y, w, h), frame.shape)
                face_roi = frame[y1:y2, x1:x2]

                if face_roi is None or face_roi.size == 0:
                    continue

                # Run predictions
                if detector.should_process():
                    if mode in ["mask", "combined"] and detector.mask_model is not None:
                        label, conf = detector.predict_mask(face_roi)
                        if label:
                            last_mask = label
                            last_confidence = conf
                            
                            box_color = detector.COLOR_MASK if label == "MASK" else detector.COLOR_NO_MASK
                            detector.draw_modern_bbox(frame, x1, y1, x2, y2, box_color, 2)
                            detector.draw_label_with_confidence(frame, label, conf, x1, y1, box_color)
                    
                    if mode in ["emotion", "combined"] and detector.emotion_model is not None:
                        emotion, conf = detector.predict_emotion(face_roi)
                        if emotion:
                            last_emotion = emotion
                            last_confidence = conf
                            
                            detector.draw_modern_bbox(frame, x1, y1, x2, y2, detector.COLOR_EMOTION, 2)
                            detector.draw_label_with_confidence(frame, emotion, conf, x1, y1, detector.COLOR_EMOTION)
                else:
                    # Draw cached predictions
                    if mode in ["mask", "combined"] and last_mask != "N/A":
                        box_color = detector.COLOR_MASK if last_mask == "MASK" else detector.COLOR_NO_MASK
                        detector.draw_modern_bbox(frame, x1, y1, x2, y2, box_color, 2)
                        detector.draw_label_with_confidence(frame, last_mask, last_confidence, x1, y1, box_color)
                    
                    if mode in ["emotion", "combined"] and last_emotion != "N/A":
                        detector.draw_modern_bbox(frame, x1, y1, x2, y2, detector.COLOR_EMOTION, 2)
                        detector.draw_label_with_confidence(frame, last_emotion, last_confidence, x1, y1, detector.COLOR_EMOTION)

            # Update stats
            stats.update_faces(faces_count)
            stats.update_mask(last_mask)
            stats.update_emotion(last_emotion)

            # Draw overlays
            detector.draw_stats_panel(frame, faces_count, last_emotion, last_mask, current_fps)
            detector.draw_instructions(frame)

            # Show frame
            cv2.imshow(window_name, frame)

            # Check for exit
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
            if cv2.waitKey(1) & 0xFF == 27:
                break

    except Exception as e:
        print(f"ERROR: Runtime error: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"{window_name} stopped")


if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "mask"
    run_detection(mode)
