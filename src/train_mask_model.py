import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import tensorflow as tf  # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # type: ignore
from tensorflow.keras.applications import MobileNetV2  # type: ignore
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout  # type: ignore
from tensorflow.keras.models import Model  # type: ignore
from src.mask_detection import MASK_INPUT_SIZE


train_dir = "dataset/mask_dataset"

# ---------------- DATA ----------------
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

train_data = datagen.flow_from_directory(
    train_dir,
    classes=['with_mask', 'without_mask'],
    target_size=(MASK_INPUT_SIZE, MASK_INPUT_SIZE),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

val_data = datagen.flow_from_directory(
    train_dir,
    classes=['with_mask', 'without_mask'],
    target_size=(MASK_INPUT_SIZE, MASK_INPUT_SIZE),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

print("Class indices:", train_data.class_indices)

# ---------------- MODEL ----------------
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(MASK_INPUT_SIZE, MASK_INPUT_SIZE, 3)
)

base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(2, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ---------------- TRAIN ----------------
model.fit(
    train_data,
    validation_data=val_data,
    epochs=15
)

# ---------------- SAVE ----------------
os.makedirs("models", exist_ok=True)
model.save("models/mask_model.h5")

print("✅ Model retrained successfully!")
