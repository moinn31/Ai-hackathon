# -*- coding: utf-8 -*-
"""
=============================================================
  Vegetable Classification System - Custom CNN Trainer
=============================================================
  Builds and trains an ultra-compact Custom CNN from scratch 
  using Keras/TensorFlow. Optimized with GlobalAveragePooling2D 
  to ensure model file size is under 5MB for seamless GitHub push.
=============================================================
  Run:  python train_cnn.py
=============================================================

=============================================================
  VIVA EXPLANATION: CUSTOM CNN vs MOBILENETV2
=============================================================
  1. Architecture Comparison:
     - Custom CNN uses standard Conv2D layers where filters apply 
       convolutions across all input channels simultaneously.
     - MobileNetV2 uses Depthwise Separable Convolutions (spatial 
       filtering per channel followed by 1x1 pointwise linear combination),
       drastically reducing parameter count and computational cost.
       
  2. Trade-offs:
     - Custom CNN: Simple, easy to understand, trains quickly on small datasets,
       but lacks the deep hierarchical feature extraction of ImageNet models.
     - MobileNetV2: Highly robust, superior accuracy via Transfer Learning, 
       but acts as a more complex black-box feature extractor.
       
  3. Layer Functions:
     - Conv2D: Extracts spatial features (edges, textures, shapes).
     - MaxPooling2D: Downsamples feature maps, reducing spatial dimensions.
     - GlobalAveragePooling2D: Averages each feature map into a single number, 
       preventing massive Dense weight matrices and keeping file size under 5MB.
     - Dropout: Randomly zeroes out activations to prevent overfitting.
=============================================================
"""

import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from utils import CLASS_NAMES, IMG_SIZE, BATCH_SIZE

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "balanced")
EPOCHS = 15

def build_custom_cnn():
    """Builds an ultra-compact 3-block CNN architecture from scratch."""
    model = Sequential([
        # Block 1
        Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        
        # Block 2
        Conv2D(64, (3, 3), activation="relu"),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        
        # Block 3
        Conv2D(64, (3, 3), activation="relu"),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        
        # Classification Head (GlobalAveragePooling keeps model size under 3MB!)
        GlobalAveragePooling2D(),
        Dense(64, activation="relu"),
        Dropout(0.5),
        Dense(len(CLASS_NAMES), activation="softmax")
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

def train():
    print("=" * 60)
    print("  [CNN-TRAINER] Training Ultra-Compact Custom CNN")
    print("=" * 60)
    
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=0.2,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
    )

    train_generator = train_datagen.flow_from_directory(
        DATASET_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="sparse",
        subset="training",
        shuffle=True,
    )

    val_generator = train_datagen.flow_from_directory(
        DATASET_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="sparse",
        subset="validation",
        shuffle=False,
    )

    model = build_custom_cnn()
    model.summary()

    h5_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cnn_model.h5")
    keras_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cnn_model.keras")

    early_stop = EarlyStopping(monitor="val_accuracy", patience=4, restore_best_weights=True, verbose=1)
    checkpoint = ModelCheckpoint(keras_path, monitor="val_accuracy", save_best_only=True, verbose=1)

    print("\n>>> Starting training ...\n")
    model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=val_generator,
        callbacks=[early_stop, checkpoint],
    )

    model.save(h5_path)
    model.save(keras_path)
    print(f"\n[SAVED] Custom CNN saved to: {h5_path} and {keras_path}\n")

if __name__ == "__main__":
    train()
