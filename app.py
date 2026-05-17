# -*- coding: utf-8 -*-
"""
=============================================================
  Vegetable Classifier AI  -  Streamlit Web Application
  Transfer Learning with MobileNetV2 | TensorFlow / Keras
=============================================================
  Run:  streamlit run app.py
=============================================================
"""

# ──────────────────────────────────────────────
# IMPORTS
# ──────────────────────────────────────────────
import os
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing.image import img_to_array

# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vegetable_classifier.h5")
IMG_SIZE = (224, 224)

# Internal class labels (must match training order exactly)
CLASS_NAMES = [
    "green_chilli",
    "ivy_gourd",
    "okra",
    "peas",
    "pointed_gourd",
]

# User-friendly display names with local names
DISPLAY_NAMES = {
    "green_chilli":   "Green Chilli (Marcha)",
    "ivy_gourd":      "Ivy Gourd (Tindoda)",
    "okra":           "Okra (Bhinda)",
    "peas":           "Peas (Vatana)",
    "pointed_gourd":  "Pointed Gourd (Parvad)",
}

# Emoji per class for extra visual flair
CLASS_EMOJI = {
    "green_chilli":   "\U0001F336\uFE0F",
    "ivy_gourd":      "\U0001F96C",
    "okra":           "\U0001F33F",
    "peas":           "\U0001F331",
    "pointed_gourd":  "\U0001F952",
}

# Colour accents for the confidence badge
CLASS_COLORS = {
    "green_chilli":   "#e74c3c",
    "ivy_gourd":      "#27ae60",
    "okra":           "#2ecc71",
    "peas":           "#f1c40f",
    "pointed_gourd":  "#1abc9c",
}


# ──────────────────────────────────────────────
# MODEL LOADING  (cached so it loads only once)
# ──────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model ...")
def load_classifier():
    """Load the trained .h5 model from disk."""
    model = load_model(MODEL_PATH)
    return model


# ──────────────────────────────────────────────
# PREPROCESSING
# ──────────────────────────────────────────────
def preprocess(image: Image.Image) -> np.ndarray:
    """
    Resize, normalise, and batch-expand a PIL image
    so it is ready for model.predict().
    """
    image = image.convert("RGB")                     # ensure 3 channels
    image = image.resize(IMG_SIZE)                    # resize to 224x224
    img_array = img_to_array(image) / 255.0           # normalise to [0,1]
    img_array = np.expand_dims(img_array, axis=0)     # add batch dimension
    return img_array


# ──────────────────────────────────────────────
# PREDICTION
# ──────────────────────────────────────────────
def predict(model, image: Image.Image):
    """
    Run inference and return:
      - predicted class key   (str)
      - confidence score      (float, 0-100)
      - full probability dict (dict)
    """
    processed = preprocess(image)
    predictions = model.predict(processed, verbose=0)
    class_idx = int(np.argmax(predictions[0]))
    confidence = float(predictions[0][class_idx]) * 100
    class_key = CLASS_NAMES[class_idx]

    # Build a dict of all class probabilities for the bar chart
    prob_dict = {
        DISPLAY_NAMES[c]: float(predictions[0][i]) * 100
        for i, c in enumerate(CLASS_NAMES)
    }

    return class_key, confidence, prob_dict


# ──────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Vegetable Classifier AI",
    page_icon="\U0001F966",
    layout="centered",
    initial_sidebar_state="collapsed",
)


# ──────────────────────────────────────────────
# CUSTOM CSS  (premium dark-green theme)
# ──────────────────────────────────────────────
st.markdown("""
<style>
/* ---- Import Google Font ---- */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* ---- Global ---- */
html, body, [class*="stApp"] {
    font-family: 'Inter', sans-serif;
}

/* ---- Hero Header ---- */
.hero {
    text-align: center;
    padding: 2rem 1rem 1rem;
}
.hero h1 {
    font-size: 2.6rem;
    font-weight: 800;
    background: linear-gradient(135deg, #2ecc71, #27ae60, #1abc9c);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.3rem;
}
.hero p {
    font-size: 1.05rem;
    opacity: 0.7;
    margin-top: 0;
}

/* ---- Prediction Card ---- */
.pred-card {
    background: linear-gradient(145deg, #0d1f0d, #162916);
    border: 1px solid rgba(46,204,113,0.25);
    border-radius: 16px;
    padding: 2rem 1.5rem;
    margin: 1.5rem 0;
    text-align: center;
    box-shadow: 0 8px 32px rgba(0,0,0,0.35);
}
.pred-emoji {
    font-size: 3rem;
    margin-bottom: 0.3rem;
}
.pred-label {
    font-size: 1.8rem;
    font-weight: 700;
    color: #2ecc71;
}
.pred-conf {
    font-size: 1.1rem;
    color: #bbb;
    margin-top: 0.2rem;
}

/* ---- Probability bars ---- */
.prob-bar-wrap {
    margin: 0.45rem 0;
}
.prob-bar-label {
    font-size: 0.85rem;
    font-weight: 500;
    margin-bottom: 2px;
    color: #ddd;
}
.prob-bar-outer {
    background: rgba(255,255,255,0.08);
    border-radius: 8px;
    height: 22px;
    overflow: hidden;
}
.prob-bar-inner {
    height: 100%;
    border-radius: 8px;
    display: flex;
    align-items: center;
    padding-left: 8px;
    font-size: 0.75rem;
    font-weight: 600;
    color: #fff;
    transition: width 0.6s ease;
}

/* ---- Upload area ---- */
.upload-section {
    border: 2px dashed rgba(46,204,113,0.3);
    border-radius: 16px;
    padding: 1.5rem;
    margin: 1rem 0;
    background: rgba(46,204,113,0.03);
}

/* ---- Footer ---- */
.footer {
    text-align: center;
    padding: 2rem 0 1rem;
    font-size: 0.85rem;
    opacity: 0.5;
}

/* ---- Misc tweaks ---- */
.stTabs [data-baseweb="tab-list"] {
    gap: 2rem;
    justify-content: center;
}
.stTabs [data-baseweb="tab"] {
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>\U0001F966 Vegetable Classifier AI</h1>
    <p>Upload or capture a vegetable photo and let AI identify it instantly.</p>
</div>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# LOAD MODEL
# ──────────────────────────────────────────────
model = load_classifier()


# ──────────────────────────────────────────────
# IMAGE INPUT  (tabs: Upload / Camera)
# ──────────────────────────────────────────────
tab_upload, tab_camera = st.tabs(["\U0001F4C1 Upload Image", "\U0001F4F7 Take Photo"])

image = None  # will hold a PIL.Image if the user provides one

with tab_upload:
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Drag & drop or click to upload",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

with tab_camera:
    camera_photo = st.camera_input("Snap a photo of any vegetable")
    if camera_photo is not None:
        image = Image.open(camera_photo)


# ──────────────────────────────────────────────
# PREDICTION + DISPLAY
# ──────────────────────────────────────────────
if image is not None:
    # Show the uploaded / captured image
    col_img, col_res = st.columns([1, 1], gap="large")

    with col_img:
        st.image(image, caption="Your image", width="stretch")

    with col_res:
        with st.spinner("Classifying ..."):
            class_key, confidence, prob_dict = predict(model, image)

        display_name = DISPLAY_NAMES[class_key]
        emoji = CLASS_EMOJI[class_key]
        color = CLASS_COLORS[class_key]

        # ---- Prediction Card ----
        st.markdown(f"""
        <div class="pred-card">
            <div class="pred-emoji">{emoji}</div>
            <div class="pred-label">{display_name}</div>
            <div class="pred-conf">Confidence: {confidence:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

        # ---- Streamlit progress bar (bonus) ----
        st.progress(int(min(confidence, 100)))

    # ── All-class probability breakdown ──
    st.markdown("---")
    st.subheader("Probability Breakdown")

    # Sort descending so the predicted class is on top
    sorted_probs = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)

    bars_html = ""
    gradient_colors = ["#2ecc71", "#27ae60", "#1abc9c", "#16a085", "#0e6655"]
    for idx, (name, prob) in enumerate(sorted_probs):
        bar_color = gradient_colors[idx % len(gradient_colors)]
        width = max(prob, 2)  # min width so label is visible
        bars_html += f"""
        <div class="prob-bar-wrap">
            <div class="prob-bar-label">{name}</div>
            <div class="prob-bar-outer">
                <div class="prob-bar-inner"
                     style="width:{width}%; background:{bar_color};">
                    {prob:.1f}%
                </div>
            </div>
        </div>
        """

    st.markdown(bars_html, unsafe_allow_html=True)

else:
    # ---- Empty state ----
    st.markdown("")
    st.info(
        "\U0001F4A1 **No image provided yet.**  \n"
        "Upload a vegetable photo or use your camera to get started!",
        icon="\U0001F449",
    )



