# Dataset Directory Layout

<!-- TODO: dataset names, sources and licenses -->

This directory stores training datasets for the mask detection and facial emotion recognition models. Datasets are not included in this repository.

## Mask Dataset Layout

Expected by `src/train_mask_model.py`:

```
dataset/mask_dataset/
├── with_mask/
└── without_mask/
```

- **Class folders**:
  1. `with_mask` (index 0)
  2. `without_mask` (index 1)
- **Image format**: RGB images resized to 224x224 during training.

## Emotion Dataset Layout

Expected by `src/train_emotion_model.py`:

```
dataset/emotion_dataset/
├── angry/
├── disgust/
├── fear/
├── happy/
├── neutral/
├── sad/
└── surprise/
```

- **Class folders**: 7 emotion categories corresponding to the 7-class emotion model.
- **Image format**: Grayscale images resized to 48x48 during training.
