import cv2 # type: ignore
import numpy as np # type: ignore
from tensorflow.keras.models import load_model # type: ignore
from face_detection import detect_faces

EMOTION_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

class EmotionPredictor:
    IMG_SIZE = 48
    TARGET_SIZE = (48, 48)

    def __init__(self, model):
        self.model = model
        if model is None:
            raise ValueError("Emotion model is required")

    def preprocess(self, faceROI):
        if faceROI is None or faceROI.size == 0:
            return None
        
        gray = cv2.cvtColor(faceROI, cv2.COLOR_BGR2GRAY)
        
        if gray.shape[:2] != self.TARGET_SIZE:
            gray = cv2.resize(gray, self.TARGET_SIZE)
        
        normalized = gray.astype(np.float32) / 255.0
        
        return normalized.reshape(1, self.IMG_SIZE, self.IMG_SIZE, 1)

    def predict(self, faceROI):
        processed = self.preprocess(faceROI)
        if processed is None:
            return None, 0.0

        try:
            pred = self.model.predict(processed, verbose=0)
            
            emotion_idx = np.argmax(pred[0])
            emotion = EMOTION_LABELS[emotion_idx]
            confidence = float(np.max(pred[0])) * 100
            
            return emotion, confidence
        except Exception as e:
            print(f"Prediction error: {e}")
            return None, 0.0


def run_emotion_detection(camera_index=0):
    emotion_model = load_model("models/emotion_model.h5")
    predictor = EmotionPredictor(emotion_model)
    
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        print("Camera not available")
        return

    window_name = "Emotion Detection"

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = detect_faces(frame)

        for (x, y, w, h) in faces:
            padding = 15
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(frame.shape[1], x + w + padding)
            y2 = min(frame.shape[0], y + h + padding)

            face_roi = frame[y1:y2, x1:x2]

            label, confidence = predictor.predict(face_roi)
            
            if label:
                text = f"{label} ({confidence:.1f}%)"
                color = (255, 0, 0)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.rectangle(frame, (x1, y1 - 28), (x2, y1), color, -1)
                cv2.putText(frame, text, (x1 + 5, y1 - 8),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        cv2.imshow(window_name, frame)

        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_emotion_detection()