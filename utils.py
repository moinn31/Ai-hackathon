# -*- coding: utf-8 -*-
"""
=============================================================
  Vegetable Classification System - Utility Module
=============================================================
  Contains shared configuration, class definitions, and 
  image preprocessing logic for both API and Frontend.
=============================================================

=============================================================
  VIVA EXPLANATION: PREPROCESSING & DATASET APPROACH
=============================================================
  1. Why this dataset approach is valid:
     - Balanced classes (exactly 100 images per class) prevent 
       class imbalance bias during gradient descent.
     - Pre-augmentation and real-time ImageDataGenerator augmentation 
       (rotation, flipping, zooming) introduce spatial variance, 
       helping the model generalize despite having only 500 base images.
       
  2. Preprocessing requirements:
     - Both MobileNetV2 and our Custom CNN expect fixed 224x224 RGB inputs.
     - Pixel normalization (dividing by 255.0) scales activations to [0, 1],
       ensuring numerical stability and faster convergence.
=============================================================
"""

import os
import numpy as np
from PIL import Image
from keras.preprocessing.image import img_to_array

# ──────────────────────────────────────────────
# 1. GLOBAL CONFIGURATION
# ──────────────────────────────────────────────
IMG_SIZE = (224, 224)
BATCH_SIZE = 16

# Internal class labels (must match training order exactly)
CLASS_NAMES = [
    "green_chilli",
    "ivy_gourd",
    "okra",
    "peas",
    "pointed_gourd",
]

# User-friendly display names with local/regional terms
DISPLAY_NAMES = {
    "green_chilli":   "Green Chilli (Marcha)",
    "ivy_gourd":      "Ivy Gourd (Tindoda)",
    "okra":           "Okra (Bhinda)",
    "peas":           "Peas (Vatana)",
    "pointed_gourd":  "Pointed Gourd (Parvad)",
}

# Emoji per class for visual representation
CLASS_EMOJI = {
    "green_chilli":   "🌶️",
    "ivy_gourd":      "🥒",
    "okra":           "🌿",
    "peas":           "🌱",
    "pointed_gourd":  "🍐",
}

# Hex color accents for UI confidence badges
CLASS_COLORS = {
    "green_chilli":   "#e74c3c",  # Red
    "ivy_gourd":      "#27ae60",  # Emerald Green
    "okra":           "#2ecc71",  # Light Green
    "peas":           "#f1c40f",  # Yellow-Green accent
    "pointed_gourd":  "#1abc9c",  # Teal
}

# ──────────────────────────────────────────────
# 2. IMAGE PREPROCESSING
# ──────────────────────────────────────────────
def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Preprocesses a PIL Image for Keras model prediction.
    
    Steps:
      1. Convert to RGB (removes alpha channels if PNG).
      2. Resize to target IMG_SIZE (224x224).
      3. Convert PIL Image to Numpy array.
      4. Normalize pixel values from [0, 255] to [0.0, 1.0].
      5. Expand dimensions to add batch axis -> shape (1, 224, 224, 3).
    """
    # Ensure image is in RGB mode
    if image.mode != "RGB":
        image = image.convert("RGB")
        
    # Resize to 224x224 using high-quality Lanczos resampling
    image = image.resize(IMG_SIZE, Image.Resampling.LANCZOS)
    
    # Convert to numpy array and normalize
    img_array = img_to_array(image) / 255.0
    
    # Expand dimensions: (224, 224, 3) -> (1, 224, 224, 3)
    img_batch = np.expand_dims(img_array, axis=0)
    
    return img_batch
