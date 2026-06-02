import cv2 # type: ignore
import numpy as np # type: ignore


cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)


def detect_faces(frame, scaleFactor=1.2, minNeighbors=6, minSize=(60, 60)):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=scaleFactor,
        minNeighbors=minNeighbors,
        minSize=minSize
    )
    return faces


def crop_face_with_padding(frame, bbox, padding=20):
    x, y, w, h = bbox
    y1 = max(0, y - padding)
    y2 = min(frame.shape[0], y + h + padding)
    x1 = max(0, x - padding)
    x2 = min(frame.shape[1], x + w + padding)
    return frame[y1:y2, x1:x2]


if __name__ == "__main__":
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("Camera not available")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = detect_faces(frame)

        for (x, y, w, h) in faces:
            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

        cv2.imshow("Face Detection", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()