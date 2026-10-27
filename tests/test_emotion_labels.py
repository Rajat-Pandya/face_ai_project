from src.emotion_detection import EMOTION_LABELS
from src.real_time_detection import RealtimeDetector


def test_emotion_labels_order():
    expected_order = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
    assert EMOTION_LABELS == expected_order
    assert RealtimeDetector.EMOTION_LABELS == expected_order
