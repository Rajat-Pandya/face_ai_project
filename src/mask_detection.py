import cv2 # type: ignore
import numpy as np # type: ignore
from collections import deque
from tensorflow.keras.models import load_model # type: ignore
from face_detection import detect_faces


class MaskPredictor:
    IMG_SIZE = 224
    
    def __init__(self, model, buffer_size=3):
        self.model = model
        self.buffer = deque(maxlen=buffer_size)
        
        if model is None:
            raise ValueError("Mask model is required")

    def preprocess(self, faceROI):
        if faceROI is None or faceROI.size == 0:
            return None
        
        resized = cv2.resize(faceROI, (self.IMG_SIZE, self.IMG_SIZE))
        normalized = resized.astype(np.float32) / 255.0
        
        return normalized.reshape(1, self.IMG_SIZE, self.IMG_SIZE, 3)

    def predict(self, faceROI):
        processed = self.preprocess(faceROI)
        if processed is None:
            return None, 0.0

        try:
            pred = self.model.predict(processed, verbose=0)
            
            self.buffer.append(pred[0])
            avg_pred = np.mean(self.buffer, axis=0)
            
            if avg_pred[0] > avg_pred[1]:
                label = "MASK"
            else:
                label = "NO MASK"
            
            confidence = float(np.max(avg_pred)) * 100
            
            return label, confidence
        except Exception as e:
            print(f"Prediction error: {e}")
            return None, 0.0


def run_mask_detection(camera_index=0):
    mask_model = load_model("models/mask_model.h5")
    predictor = MaskPredictor(mask_model)
    
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        print("Camera not available")
        return

    window_name = "Mask Detection"

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
                if label == "MASK":
                    color = (0, 255, 0)
                else:
                    color = (0, 0, 255)
                
                text = f"{label} ({confidence:.1f}%)"
                
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
    run_mask_detection()