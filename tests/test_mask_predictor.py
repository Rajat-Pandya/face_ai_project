import numpy as np
from src.mask_detection import MaskPredictor


class DummyModel:
    def __init__(self, pred_queue):
        self.pred_queue = list(pred_queue)

    def predict(self, face_array, verbose=0):
        if self.pred_queue:
            return np.array([self.pred_queue.pop(0)], dtype=np.float32)
        return np.array([[0.5, 0.5]], dtype=np.float32)


def test_mask_predictor_decision_rule_and_smoothing():
    dummy_model = DummyModel([
        [0.9, 0.1],
        [0.2, 0.8],
        [0.1, 0.9],
        [0.1, 0.9],
    ])

    predictor = MaskPredictor(dummy_model, buffer_size=3)
    face_roi = np.zeros((100, 100, 3), dtype=np.uint8)

    label1, conf1 = predictor.predict(face_roi)
    assert label1 == "MASK"
    assert round(conf1, 1) == 90.0

    label2, conf2 = predictor.predict(face_roi)
    assert label2 == "MASK"
    assert round(conf2, 1) == 55.0

    label3, conf3 = predictor.predict(face_roi)
    assert label3 == "NO MASK"
    assert round(conf3, 1) == 60.0

    label4, conf4 = predictor.predict(face_roi)
    assert label4 == "NO MASK"
    assert round(conf4, 1) == 86.7
