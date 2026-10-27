import sys
from unittest.mock import MagicMock


def _register_stub(module_name):
    mock = MagicMock()
    sys.modules[module_name] = mock
    return mock


# Register stubs ONLY when the respective library is not importable
try:
    import tensorflow  # noqa: F401
except ImportError:
    tf = _register_stub("tensorflow")
    _register_stub("tensorflow.keras")
    _register_stub("tensorflow.keras.models")
    _register_stub("tensorflow.keras.preprocessing")
    _register_stub("tensorflow.keras.preprocessing.image")
    _register_stub("tensorflow.keras.applications")
    _register_stub("tensorflow.keras.layers")

try:
    import mediapipe  # noqa: F401
except ImportError:
    _register_stub("mediapipe")
    _register_stub("mediapipe.solutions")
    _register_stub("mediapipe.solutions.hands")
    _register_stub("mediapipe.solutions.drawing_utils")

try:
    import ultralytics  # noqa: F401
except ImportError:
    _register_stub("ultralytics")
