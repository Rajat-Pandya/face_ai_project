import numpy as np
from src.face_detection import crop_face_with_padding, detect_faces


def test_crop_face_with_padding_bounds():
    # Frame of size 100x100
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    # Bbox near top-left: x=5, y=5, w=20, h=20, padding=20
    # Negative coordinates must be clipped to 0
    bbox = (5, 5, 20, 20)
    cropped = crop_face_with_padding(frame, bbox, padding=20)
    assert cropped.shape == (45, 45, 3)

    # Bbox near bottom-right: x=80, y=80, w=30, h=30, padding=20
    # Out of range coordinates must be clipped to 100
    bbox_br = (80, 80, 30, 30)
    cropped_br = crop_face_with_padding(frame, bbox_br, padding=20)
    assert cropped_br.shape == (40, 40, 3)


def test_detect_faces_blank_frame_returns_no_faces():
    blank = np.zeros((300, 300, 3), dtype=np.uint8)
    faces = detect_faces(blank)
    assert len(faces) == 0
