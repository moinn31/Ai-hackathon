# -*- coding: utf-8 -*-
"""
=============================================================
  Local Vegetable Variety Classifier
  Transfer Learning with MobileNetV2 | TensorFlow / Keras
=============================================================
  Classes : green_chilli, ivy_gourd, okra, peas, pointed_gourd
  Dataset : 500 images (100 per class, pre-augmented)
=============================================================
"""

# ──────────────────────────────────────────────
# 1. IMPORTS
# ──────────────────────────────────────────────
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.metrics import confusion_matrix, classification_report

# Suppress excessive TF warnings for cleaner output
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# ──────────────────────────────────────────────
# 2. CONFIGURATION
# ──────────────────────────────────────────────
DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "balanced")
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 15
NUM_CLASSES = 5
LEARNING_RATE = 1e-4
MODEL_SAVE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vegetable_classifier.h5")

CLASS_NAMES = [
    "green_chilli",
    "ivy_gourd",
    "okra",
    "peas",
    "pointed_gourd",
]

print("=" * 60)
print("  [VEG-CLASSIFIER] Local Vegetable Variety Classifier")
print("=" * 60)
print(f"  Dataset path  : {DATASET_DIR}")
print(f"  Image size    : {IMG_SIZE}")
print(f"  Batch size    : {BATCH_SIZE}")
print(f"  Epochs        : {EPOCHS}")
print(f"  Classes       : {NUM_CLASSES}")
print("=" * 60)

# ──────────────────────────────────────────────
# 3. DATA LOADING  &  TRAIN / VALIDATION SPLIT
# ──────────────────────────────────────────────

# Training generator – includes augmentation
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,              # normalise pixel values to [0, 1]
    validation_split=0.2,           # 80-20 train/val split
    rotation_range=20,              # ± 20° rotation
    zoom_range=0.2,                 # ± 20 % zoom
    width_shift_range=0.1,          # ± 10 % horizontal shift
    height_shift_range=0.1,         # ± 10 % vertical shift
    horizontal_flip=True,           # random horizontal flip
)

# Validation generator – only rescaling (NO augmentation)
val_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2,
)

# Training subset
train_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True,
    seed=42,
    classes=CLASS_NAMES,            # enforce consistent label order
)

# Validation subset
val_generator = val_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False,
    seed=42,
    classes=CLASS_NAMES,
)

print(f"\n[OK] Training samples   : {train_generator.samples}")
print(f"[OK] Validation samples : {val_generator.samples}")
print(f"[OK] Class indices      : {train_generator.class_indices}\n")

# ──────────────────────────────────────────────
# 4. MODEL BUILDING  (MobileNetV2 Transfer Learning)
# ──────────────────────────────────────────────

# Load pre-trained MobileNetV2 (without top classification layer)
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3),
)

# Freeze the base model – we only train the custom head first
base_model.trainable = False

# Build custom classification head
x = base_model.output
x = GlobalAveragePooling2D()(x)          # global spatial pooling
x = Dense(128, activation="relu")(x)     # fully connected layer
x = Dropout(0.5)(x)                      # regularisation
output = Dense(NUM_CLASSES, activation="softmax")(x)  # 5-class output

model = Model(inputs=base_model.input, outputs=output)

# ──────────────────────────────────────────────
# 5. COMPILATION
# ──────────────────────────────────────────────
model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# ──────────────────────────────────────────────
# 6. CALLBACKS
# ──────────────────────────────────────────────

# Stop early if validation loss stops improving
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True,
    verbose=1,
)

# Save the best model during training
checkpoint = ModelCheckpoint(
    filepath=MODEL_SAVE_PATH,
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1,
)

# ──────────────────────────────────────────────
# 7. TRAINING
# ──────────────────────────────────────────────
print("\n>>> Starting training ...\n")

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=[early_stop, checkpoint],
)

print("\n[OK] Training complete!\n")

# ──────────────────────────────────────────────
# 8. EVALUATION
# ──────────────────────────────────────────────

# --- 8a. Final accuracy ---
train_loss, train_acc = model.evaluate(train_generator, verbose=0)
val_loss, val_acc = model.evaluate(val_generator, verbose=0)

print("=" * 50)
print(f"  [RESULT] Training Accuracy   : {train_acc * 100:.2f}%")
print(f"  [RESULT] Validation Accuracy : {val_acc * 100:.2f}%")
print(f"  [RESULT] Training Loss       : {train_loss:.4f}")
print(f"  [RESULT] Validation Loss     : {val_loss:.4f}")
print("=" * 50)

# --- 8b. Predictions on validation set ---
val_generator.reset()
y_pred_probs = model.predict(val_generator)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = val_generator.classes

# --- 8c. Classification Report ---
print("\n--- Classification Report ---\n")
print(classification_report(
    y_true, y_pred,
    target_names=CLASS_NAMES,
    digits=4,
))

# --- 8d. Confusion Matrix ---
cm = confusion_matrix(y_true, y_pred)
print("--- Confusion Matrix ---\n")
print(cm)

# ──────────────────────────────────────────────
# 9. VISUALISATION
# ──────────────────────────────────────────────

# --- 9a. Accuracy & Loss Curves ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy plot
axes[0].plot(history.history["accuracy"], label="Train Accuracy", linewidth=2)
axes[0].plot(history.history["val_accuracy"], label="Val Accuracy", linewidth=2)
axes[0].set_title("Training vs Validation Accuracy", fontsize=14)
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy")
axes[0].legend(loc="lower right")
axes[0].grid(True, alpha=0.3)

# Loss plot
axes[1].plot(history.history["loss"], label="Train Loss", linewidth=2)
axes[1].plot(history.history["val_loss"], label="Val Loss", linewidth=2)
axes[1].set_title("Training vs Validation Loss", fontsize=14)
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].legend(loc="upper right")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "training_curves.png"), dpi=150)
plt.show()
print("[SAVED] Training curves saved to training_curves.png")

# --- 9b. Confusion Matrix Heatmap ---
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES,
)
plt.title("Confusion Matrix", fontsize=14)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "confusion_matrix.png"), dpi=150)
plt.show()
print("[SAVED] Confusion matrix saved to confusion_matrix.png\n")

# ──────────────────────────────────────────────
# 10. SAVE FINAL MODEL
# ──────────────────────────────────────────────
model.save(MODEL_SAVE_PATH)
print(f"[SAVED] Model saved to: {MODEL_SAVE_PATH}\n")

# ──────────────────────────────────────────────
# 11. PREDICTION FUNCTION
# ──────────────────────────────────────────────

def predict_image(image_path, model_path=MODEL_SAVE_PATH):
    """
    Predict the vegetable class for a single image.

    Parameters
    ----------
    image_path : str
        Path to the input image file.
    model_path : str
        Path to the saved .h5 model file.

    Returns
    -------
    tuple : (class_name: str, confidence: float)
        Predicted class label and confidence percentage.
    """
    # Load the saved model
    classifier = load_model(model_path)

    # Load and preprocess the image
    img = load_img(image_path, target_size=IMG_SIZE)
    img_array = img_to_array(img) / 255.0        # normalise
    img_array = np.expand_dims(img_array, axis=0) # add batch dimension

    # Predict
    predictions = classifier.predict(img_array, verbose=0)
    class_idx = np.argmax(predictions[0])
    confidence = predictions[0][class_idx] * 100

    class_name = CLASS_NAMES[class_idx]

    print("=" * 45)
    print(f"  [PREDICTION] Class : {class_name}")
    print(f"  [CONFIDENCE]       : {confidence:.2f}%")
    print("=" * 45)

    return class_name, confidence


# ──────────────────────────────────────────────
# 12. QUICK TEST  (uncomment to use)
# ──────────────────────────────────────────────
# predict_image("path/to/your/test_image.jpg")

print("\n[DONE] Pipeline complete! Model is ready for inference.")
print("   Use  predict_image('path/to/image.jpg')  to classify a vegetable.\n")
