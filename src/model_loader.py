import os
from typing import Optional

import numpy as np # type: ignore
from tensorflow.keras.models import load_model # type: ignore


class ModelLoader:
    _instance = None
    _mask_model = None
    _emotion_model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @property
    def mask_model(self):
        if self._mask_model is None:
            self._mask_model = self._load_mask_model()
        return self._mask_model

    @property
    def emotion_model(self):
        if self._emotion_model is None:
            self._emotion_model = self._load_emotion_model()
        return self._emotion_model

    def _load_mask_model(self, model_path: str = "models/mask_model.h5"):
        if not os.path.exists(model_path):
            print(f"Warning: Mask model not found at {model_path}")
            return None
        try:
            model = load_model(model_path)
            print("Mask model loaded successfully")
            return model
        except Exception as e:
            print(f"Error loading mask model: {e}")
            return None

    def _load_emotion_model(self, model_path: str = "models/emotion_model.h5"):
        if not os.path.exists(model_path):
            print(f"Warning: Emotion model not found at {model_path}")
            return None
        try:
            model = load_model(model_path)
            print("Emotion model loaded successfully")
            return model
        except Exception as e:
            print(f"Error loading emotion model: {e}")
            return None

    def load_all_models(self):
        self.mask_model
        self.emotion_model
        return self._mask_model is not None or self._emotion_model is not None

    def unload_all(self):
        if self._mask_model is not None:
            del self._mask_model
            self._mask_model = None
        if self._emotion_model is not None:
            del self._emotion_model
            self._emotion_model = None
        print("All models unloaded")


_loader = ModelLoader()


def load_mask_model(model_path: str = "models/mask_model.h5"):
    if _loader.mask_model is None:
        _loader._mask_model = _loader._load_mask_model(model_path)
    return _loader.mask_model


def load_emotion_model(model_path: str = "models/emotion_model.h5"):
    if _loader.emotion_model is None:
        _loader._emotion_model = _loader._load_emotion_model(model_path)
    return _loader.emotion_model


def get_model_loader() -> ModelLoader:
    return _loader