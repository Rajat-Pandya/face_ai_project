import os
import tensorflow as tf # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization # type: ignore
from tensorflow.keras import Input # type: ignore

# ---------------- PATH ----------------
train_dir = "dataset/emotion_dataset"

# ---------------- SETTINGS ----------------
IMG_SIZE = 48
BATCH_SIZE = 32
EPOCHS = 25

# ---------------- DATA GENERATOR ----------------
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=15,
    zoom_range=0.1,
    horizontal_flip=True
)

train_data = datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)

val_data = datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation"
)

# ---------------- MODEL ----------------
model = Sequential([

    Input(shape=(48,48,1)),

    # Block 1
    Conv2D(32,(3,3),activation='relu'),
    BatchNormalization(),
    MaxPooling2D(),

    # Block 2
    Conv2D(64,(3,3),activation='relu'),
    BatchNormalization(),
    MaxPooling2D(),

    # Block 3
    Conv2D(128,(3,3),activation='relu'),
    BatchNormalization(),
    MaxPooling2D(),

    # Dense
    Flatten(),
    Dense(128,activation='relu'),
    Dropout(0.5),   # 🔥 important
    Dense(7,activation='softmax')
])

# ---------------- COMPILE ----------------
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ---------------- TRAIN ----------------
model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS
)

# ---------------- SAVE ----------------
os.makedirs("models", exist_ok=True)
model.save("models/emotion_model.h5")

print("✅ Emotion model saved!")
print("Class indices:", train_data.class_indices)