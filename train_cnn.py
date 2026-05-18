# -*- coding: utf-8 -*-
"""
=============================================================================
  Vegetable Classification System - Advanced Custom CNN Trainer (v2)
=============================================================================
  Builds, trains, and evaluates a highly optimized, deeper Custom CNN 
  from scratch using Keras/TensorFlow. Designed specifically to overcome 
  small dataset limitations (500 images), prevent overfitting, and improve 
  generalization between visually similar classes without pretrained models.
=============================================================================
  Run:  python train_cnn.py
=============================================================================

=============================================================================
  VIVA EXPLANATION COMMENTS (DEEP LEARNING ARCHITECTURAL INSIGHTS)
=============================================================================
  1. Why Deeper CNN Improves Learning:
     - Shallow networks can only detect low-level, simple spatial features 
       like primitive edges, lines, and color blobs.
     - Deeper CNNs create a hierarchical feature representation: Block 1 detects 
       edges/textures, Block 2 combines them into shapes/patterns, and Block 3 
       identifies complex object-level semantics (e.g., distinct ridges on okra 
       or surface stripes on pointed gourd). This drastically improves accuracy.

  2. Role of BatchNormalization:
     - Internal Covariate Shift: As weights update during training, the distribution 
       of activations in deeper layers shifts constantly, slowing down convergence.
     - BatchNormalization normalizes layer activations to maintain a mean of 0 
       and variance of 1 across the mini-batch.
     - Benefits: Allows higher learning rates, acts as a subtle regularizer, 
       and prevents vanishing/exploding gradients, leading to highly stable learning.

  3. Role of Dropout:
     - Overfitting occurs when a network memorizes noise or specific training samples 
       instead of learning generalizable underlying features.
     - Dropout randomly zeroes out a percentage of neuron activations (e.g., 25% or 50%) 
       during each training step.
     - Benefits: Forces the network to learn redundant, robust representations rather 
       than relying on specific co-adapted pathways.

  4. Why Strong Data Augmentation is Important:
     - Training a deep CNN on only 500 images easily leads to extreme overfitting.
     - Strong augmentation (rotation, zoom, shifts, shear, brightness adjustments) 
       dynamically generates infinite spatial and lighting variations during training.
     - Benefits: Simulates real-world environmental noise, forces invariance to scale/angle, 
       and effectively multiplies the effective dataset size, enabling exceptional generalization.
=============================================================================
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
from utils import CLASS_NAMES, BATCH_SIZE

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "balanced")

# ──────────────────────────────────────────────
# PART 5: INPUT SIZE EXPERIMENT & OPTIMIZATION
# ──────────────────────────────────────────────
# Using 224x224 image size provides richer spatial resolution for deep convolutions,
# preventing feature collapse when strong rotation/zoom augmentations are applied.
EXPERIMENT_IMG_SIZE = (224, 224)
EPOCHS = 30


def build_improved_cnn():
    """
    PART 1: IMPROVED CNN ARCHITECTURE
    Builds a deeper, highly robust 3-block CNN architecture from scratch.
    Uses He Normal initialization to ensure robust gradient flow across deep layers.
    """
    model = Sequential([
        # Input Layer
        Input(shape=(EXPERIMENT_IMG_SIZE[0], EXPERIMENT_IMG_SIZE[1], 3)),
        
        # Conv Block 1
        Conv2D(32, (3, 3), activation="relu", padding="same", kernel_initializer="he_normal"),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation="relu", kernel_initializer="he_normal"),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Conv Block 2
        Conv2D(64, (3, 3), activation="relu", kernel_initializer="he_normal"),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation="relu", kernel_initializer="he_normal"),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Conv Block 3
        Conv2D(128, (3, 3), activation="relu", kernel_initializer="he_normal"),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.3),
        
        # Classifier Head
        GlobalAveragePooling2D(),
        Dense(128, activation="relu", kernel_initializer="he_normal"),
        Dropout(0.5),
        Dense(len(CLASS_NAMES), activation="softmax", kernel_initializer="glorot_uniform")
    ])
    
    # PART 3: TRAINING IMPROVEMENTS (Adam 0.0001 + categorical_crossentropy)
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train():
    print("=" * 60)
    print("  [CNN-TRAINER] Training Improved Custom CNN (v2)")
    print("=" * 60)
    
    # ──────────────────────────────────────────────
    # PART 2: BETTER DATA AUGMENTATION
    # ──────────────────────────────────────────────
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=0.2,
        rotation_range=30,
        zoom_range=0.3,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
    )

    # Validation generator uses only rescaling (no random augmentation during evaluation)
    val_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=0.2,
    )

    train_generator = train_datagen.flow_from_directory(
        DATASET_DIR,
        target_size=EXPERIMENT_IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="training",
        shuffle=True,
    )

    val_generator = val_datagen.flow_from_directory(
        DATASET_DIR,
        target_size=EXPERIMENT_IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation",
        shuffle=False,
    )

    model = build_improved_cnn()
    model.summary()

    # PART 7: SAVE MODEL (Save as cnn_model_v2.h5 and cnn_model_v2.keras)
    h5_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cnn_model_v2.h5")
    keras_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cnn_model_v2.keras")

    # ──────────────────────────────────────────────
    # PART 3: CALLBACKS
    # ──────────────────────────────────────────────
    # Increased patience slightly to allow deep network to warm up features under strong augmentation
    early_stop = EarlyStopping(monitor="val_accuracy", patience=10, restore_best_weights=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor="val_accuracy", factor=0.3, patience=2, min_lr=1e-6, verbose=1)
    checkpoint = ModelCheckpoint(keras_path, monitor="val_accuracy", save_best_only=True, verbose=1)

    # ──────────────────────────────────────────────
    # PART 4: CLASS WEIGHT HANDLING
    # ──────────────────────────────────────────────
    # Slightly increase weights for ivy_gourd and pointed_gourd to resolve confusion
    class_weights = {}
    for class_name, idx in train_generator.class_indices.items():
        if class_name in ["ivy_gourd", "pointed_gourd"]:
            class_weights[idx] = 1.5
        else:
            class_weights[idx] = 1.0
            
    print("\n>>> Applied Class Weights:", {name: class_weights[idx] for name, idx in train_generator.class_indices.items()})

    print("\n>>> Starting training (Epochs: 30) ...\n")
    model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=val_generator,
        callbacks=[early_stop, reduce_lr, checkpoint],
        class_weight=class_weights,
    )

    # Save final model formats
    model.save(h5_path)
    model.save(keras_path)
    print(f"\n[SAVED] Improved Custom CNN saved to:\n  - {h5_path}\n  - {keras_path}\n")

    # ──────────────────────────────────────────────
    # PART 6: EVALUATION
    # ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  [EVALUATION] Model Performance on Validation Set")
    print("=" * 60)

    val_loss, val_acc = model.evaluate(val_generator, verbose=0)
    print(f"\n>>> Final Validation Accuracy: {val_acc * 100:.2f}%\n")

    # Reset generator to ensure strict ordering for confusion matrix
    val_generator.reset()
    Y_pred = model.predict(val_generator, verbose=0)
    y_pred = np.argmax(Y_pred, axis=1)
    y_true = val_generator.classes

    target_names = [name for name, idx in sorted(val_generator.class_indices.items(), key=lambda x: x[1])]

    print("--- Confusion Matrix ---")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    
    print("\n--- Classification Report ---")
    cr = classification_report(y_true, y_pred, target_names=target_names)
    print(cr)


if __name__ == "__main__":
    train()
