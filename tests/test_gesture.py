import pytest

try:
    from src.hand_detection import get_gesture
except Exception as e:
    get_gesture = None
    _import_err = e


def test_gesture_classification():
    if get_gesture is None:
        pytest.skip(
            f"Gesture classification cannot be tested without MediaPipe: {_import_err}"
        )

    assert get_gesture([0, 1, 1, 0, 0]) == "Peace"
    assert get_gesture([1, 0, 0, 0, 0]) == "Thumbs Up"
    assert get_gesture([1, 1, 1, 1, 1]) == "Open Palm"
    assert get_gesture([0, 0, 0, 0, 0]) == "Fist"
    assert get_gesture([1, 1, 0, 0, 0]) == "Unknown"
