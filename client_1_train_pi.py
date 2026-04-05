import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import numpy as np
import argparse
import time

CLIENT_ID = "Client_1"

BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, "dataset")

MODELS_DIR = os.path.join(BASE_DIR, "Models")

# ✅ FIXED FILENAMES
GLOBAL_WEIGHTS = os.path.join(MODELS_DIR, "global_model.weights.h5")
LOCAL_WEIGHTS = os.path.join(MODELS_DIR, "client_1_update.weights.h5")

os.makedirs(MODELS_DIR, exist_ok=True)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 1

TOMATO_CLASSES = [
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites",
    "Tomato___Target_Spot",
    "Tomato___Yellow_Leaf_Curl_Virus",
    "Tomato___mosaic_virus",
    "Tomato___healthy"
]

NUM_CLASSES = len(TOMATO_CLASSES)

# ✅ FIXED MODEL
def build_cnn():
    model = models.Sequential([
        tf.keras.Input(shape=(224,224,3)),

        layers.Conv2D(32, (3,3), padding='same', activation='relu'),
        layers.MaxPooling2D(),

        layers.Conv2D(64, (3,3), padding='same', activation='relu'),
        layers.MaxPooling2D(),

        layers.GlobalAveragePooling2D(),

        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    args = parser.parse_args()

    print("=========================================")
    print(f"Starting FEDAVG Training for {CLIENT_ID}")
    print(f"{NUM_CLASSES} Classes (Tomato)")
    print("=========================================")

    datagen = ImageDataGenerator(rescale=1./255)

    train_gen = datagen.flow_from_directory(
        DATA_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        classes=TOMATO_CLASSES,
        class_mode='categorical'
    )

    print(f"Found {train_gen.n} images")
    print("Class indices:", train_gen.class_indices)

    model = build_cnn()

    # ✅ LOAD GLOBAL MODEL
    if os.path.exists(GLOBAL_WEIGHTS):
        print("Loading global weights...")
        model.load_weights(GLOBAL_WEIGHTS)
    else:
        print("⚠️ No global model found — training from scratch")

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print("Starting training...")
    start = time.time()

    history = model.fit(train_gen, epochs=EPOCHS, verbose=1)

    end = time.time()

    # ✅ SAVE LOCAL UPDATE
    model.save_weights(LOCAL_WEIGHTS)

    acc = history.history['accuracy'][-1]
    loss = history.history['loss'][-1]

    print(f"Training done in {end-start:.2f}s")
    print(f"Accuracy: {acc*100:.2f}% | Loss: {loss:.4f}")

if __name__ == "__main__":
    main()