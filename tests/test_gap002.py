from src.real_time_detection import RealtimeDetector
from src.mask_detection import MaskPredictor


def test_realtime_detector_mask_input_size_matches_mask_model():
    """GAP-002: RealtimeDetector IMG_SIZE must equal the mask model's trained input size (224)."""
    assert RealtimeDetector.IMG_SIZE == 224, (
        f"RealtimeDetector.IMG_SIZE is {RealtimeDetector.IMG_SIZE}, expected 224"
    )
    assert RealtimeDetector.IMG_SIZE == MaskPredictor.IMG_SIZE
