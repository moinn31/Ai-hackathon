# -*- coding: utf-8 -*-
"""
=============================================================
  Vegetable Classification System - Flask Backend API
=============================================================
  Hosts trained Keras deep learning models and serves predictions 
  via REST API endpoints.
=============================================================
  Run:  python api.py
=============================================================

=============================================================
  VIVA EXPLANATION: FLASK BACKEND ARCHITECTURE
=============================================================
  1. Separation of Concerns:
     - By decoupling the Streamlit UI from direct model inference, 
       we create a microservice architecture. The heavy lifting (GPU/CPU 
       tensor operations) happens on the backend, allowing the frontend 
       to remain lightweight.
       
  2. Multi-Model Support:
     - The API dynamically loads and caches either `vegetable_classifier.keras` 
       (MobileNetV2 Transfer Learning) or `cnn_model.keras` (Custom CNN) 
       based on the `model_type` parameter in the POST request.
       
  3. Preprocessing & Prediction Workflow:
     - Receives binary image -> PIL Image -> utils.preprocess_image() -> 
       model.predict() -> JSON serialization -> Client response.
=============================================================
"""

import os
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
from keras.models import load_model
from utils import CLASS_NAMES, DISPLAY_NAMES, preprocess_image

app = Flask(__name__)

# ──────────────────────────────────────────────
# 1. MODEL CACHING & LOADING
# ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Define paths (prefer modern .keras, fallback to .h5 if needed)
MODEL_PATHS = {
    "mobilenet": os.path.join(BASE_DIR, "vegetable_classifier.keras"),
    "cnn":       os.path.join(BASE_DIR, "cnn_model.keras"),
}
# Fallback paths if .keras is missing but .h5 exists
FALLBACK_PATHS = {
    "mobilenet": os.path.join(BASE_DIR, "vegetable_classifier.h5"),
    "cnn":       os.path.join(BASE_DIR, "cnn_model.h5"),
}

# In-memory dictionary to store loaded Keras models so they aren't reloaded per request
loaded_models = {}

def get_model(model_type="mobilenet"):
    """Dynamically loads and caches the requested Keras model."""
    if model_type not in MODEL_PATHS:
        model_type = "mobilenet"
        
    if model_type not in loaded_models:
        path = MODEL_PATHS[model_type]
        if not os.path.exists(path):
            path = FALLBACK_PATHS[model_type]
            
        print(f"[API] Loading model '{model_type}' from {path} ...")
        loaded_models[model_type] = load_model(path)
        print(f"[API] Model '{model_type}' loaded successfully.")
        
    return loaded_models[model_type]


# ──────────────────────────────────────────────
# 2. REST API ENDPOINTS
# ──────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health_check():
    """Endpoint 2: Health check to verify API status."""
    return jsonify({"status": "API is running"}), 200


@app.route("/predict", methods=["POST"])
def predict_endpoint():
    """
    Endpoint 1: Predicts vegetable class from uploaded image.
    
    Expected Form-Data:
      - file: (image file)
      - model_type: (optional, 'mobilenet' or 'cnn')
    """
    # 1. Validate request
    if "file" not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
        
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400
        
    model_type = request.form.get("model_type", "mobilenet")

    try:
        # 2. Read and preprocess image
        image = Image.open(file.stream)
        img_batch = preprocess_image(image)
        
        # 3. Load model and predict
        model = get_model(model_type)
        predictions = model.predict(img_batch, verbose=0)
        
        # 4. Extract class and confidence
        class_idx = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][class_idx])
        class_key = CLASS_NAMES[class_idx]
        
        # Build probability breakdown dictionary
        probabilities = {
            DISPLAY_NAMES[c]: float(predictions[0][i])
            for i, c in enumerate(CLASS_NAMES)
        }
        
        # 5. Return JSON response matching exact requirement format
        return jsonify({
            "class": class_key,
            "display_name": DISPLAY_NAMES[class_key],
            "confidence": confidence,
            "probabilities": probabilities,
            "model_used": model_type
        }), 200

    except Exception as e:
        print(f"[API ERROR] {str(e)}")
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


if __name__ == "__main__":
    print("=" * 60)
    print("  [FLASK API] Starting Vegetable Classifier Backend")
    print("  Endpoints: GET /health | POST /predict")
    print("=" * 60)
    app.run(host="0.0.0.0", port=5000, debug=False)
