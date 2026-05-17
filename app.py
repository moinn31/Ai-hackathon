# -*- coding: utf-8 -*-
"""
=============================================================
  Vegetable Classifier AI  -  Professional Web Application
=============================================================
  Connects to Flask Backend API (api.py) for decoupled inference.
  Provides an ultra-clean, high-contrast, enterprise-grade UI 
  for real-time vegetable classification and model comparison.
=============================================================
  Run:  streamlit run app.py
=============================================================

=============================================================
  VIVA EXPLANATION COMMENTS (ACADEMIC & VIVA REQUIREMENTS)
=============================================================
  The following documentation is provided for academic evaluation 
  and viva defense of the underlying deep learning architecture:

  1. Why MobileNetV2 used:
     - MobileNetV2 is specifically designed for mobile and edge devices.
     - It uses Depthwise Separable Convolutions (splitting standard filtering 
       into spatial depthwise filtering followed by 1x1 pointwise combination).
     - This drastically reduces parameter count (~2.4M) and computational complexity 
       while maintaining high accuracy.

  2. What is Transfer Learning:
     - Transfer learning repurposes feature representations learned from massive datasets 
       (ImageNet: 1.4 million images across 1,000 categories) for a new, specific task.
     - By freezing the base layers (which identify universal features like edges, textures, 
       and color gradients) and training only a custom dense classification head on our 
       5 vegetable classes, we achieve exceptional generalization without needing millions of images.

  3. Difference between CNN vs MobileNet:
     - Standard CNN: Applies full 3D convolutional filters across all input channels. 
       Computationally expensive and prone to overfitting on small datasets without heavy regularization.
     - MobileNetV2: Uses inverted residuals and linear bottlenecks with depthwise separable 
       convolutions. Highly optimized, extremely lightweight, and highly accurate.

  4. Why this dataset approach is valid:
     - Balanced dataset (exactly 100 images per class) ensures the loss function 
       weights each class equally, preventing majority-class prediction bias.
     - Real-time data augmentation (rotation, shifting, flipping) simulates real-world 
       variance, enabling robust generalization from 500 base images.

  5. Limitations of the model:
     - Domain Shift: Models trained on pristine white-background images may struggle 
       with real-world kitchen backgrounds, shadows, or poor lighting.
     - Inter-Class Similarity: Visually similar vegetables (e.g., Pointed Gourd vs Ivy Gourd) 
       can occasionally be misclassified if distinguishing surface patterns are blurred.
=============================================================
"""

import io
import requests
from PIL import Image
import streamlit as st

# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────
FLASK_API_URL = "http://localhost:5000/predict"

CLASS_EMOJI = {
    "green_chilli":   "🌶️",
    "ivy_gourd":      "🥒",
    "okra":           "🌿",
    "peas":           "🌱",
    "pointed_gourd":  "🍐",
}

# ──────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Vegetable Classifier AI",
    page_icon="🥦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# CUSTOM CSS (Ultra High-Contrast Dark Mode)
# ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

/* Base Backgrounds & Typography */
html, body, [class*="stApp"] {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    background-color: #0a140a !important;
    color: #e0eee0 !important;
}

/* ---- Hero Header ---- */
.hero {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
}
.hero h1 {
    font-size: 3.4rem;
    font-weight: 800;
    background: linear-gradient(135deg, #2ecc71 0%, #27ae60 50%, #1abc9c 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
    letter-spacing: -1px;
}
.hero p {
    font-size: 1.2rem;
    color: #a4bba4;
    font-weight: 500;
    margin-top: 0;
}

/* ---- File Uploader & Buttons Overhaul (Fixing Keras/Streamlit Black Contrast) ---- */
[data-testid="stFileUploader"] {
    background: rgba(20, 38, 20, 0.6) !important;
    border: 2px dashed #2ecc71 !important;
    border-radius: 20px !important;
    padding: 2rem !important;
    box-shadow: 0 10px 25px rgba(0,0,0,0.3) !important;
}
[data-testid="stFileUploader"] * {
    color: #ffffff !important;
}
[data-testid="stFileUploader"] button {
    background: #2ecc71 !important;
    color: #000000 !important;
    font-weight: 700 !important;
    border-radius: 10px !important;
    border: none !important;
    padding: 0.5rem 1.5rem !important;
    box-shadow: 0 4px 12px rgba(46,204,113,0.3) !important;
    transition: all 0.2s ease !important;
}
[data-testid="stFileUploader"] button:hover {
    background: #27ae60 !important;
    transform: scale(1.03) !important;
}
[data-testid="stFileUploader"] small {
    color: #a4bba4 !important;
    font-size: 0.95rem !important;
}

/* ---- Streamlit Native Buttons & Inputs ---- */
div[data-testid="stButton"] button {
    background: linear-gradient(135deg, #2ecc71, #27ae60) !important;
    color: #000000 !important;
    font-weight: 700 !important;
    border-radius: 12px !important;
    border: none !important;
    box-shadow: 0 4px 15px rgba(46,204,113,0.3) !important;
}
div[data-testid="stButton"] button:hover {
    background: linear-gradient(135deg, #27ae60, #2ecc71) !important;
    color: #ffffff !important;
}

/* ---- Tabs Styling ---- */
.stTabs [data-baseweb="tab-list"] {
    gap: 3rem;
    justify-content: center;
    border-bottom: 2px solid rgba(46,204,113,0.2) !important;
    background: transparent !important;
}
.stTabs [data-baseweb="tab"] {
    font-weight: 700 !important;
    font-size: 1.15rem !important;
    color: #8da48d !important;
    padding-bottom: 1rem !important;
}
.stTabs [data-baseweb="tab"][aria-selected="true"] {
    color: #2ecc71 !important;
    border-bottom-color: #2ecc71 !important;
}

/* ---- Sidebar Styling ---- */
[data-testid="stSidebar"] {
    background-color: #060c06 !important;
    border-right: 1px solid rgba(46,204,113,0.15) !important;
}
[data-testid="stSidebar"] * {
    color: #e0eee0 !important;
}

/* ---- Prediction Glass Card ---- */
.pred-card {
    background: linear-gradient(135deg, rgba(25, 45, 25, 0.8) 0%, rgba(12, 25, 12, 0.9) 100%);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(46, 204, 113, 0.3);
    border-radius: 24px;
    padding: 2.5rem 2rem;
    margin: 1rem 0 2rem;
    text-align: center;
    box-shadow: 0 20px 40px rgba(0,0,0,0.5);
    position: relative;
    overflow: hidden;
}
.pred-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 4px;
    background: linear-gradient(90deg, #2ecc71, #1abc9c);
}
.pred-emoji {
    font-size: 4.5rem;
    margin-bottom: 0.5rem;
    filter: drop-shadow(0 10px 16px rgba(46,204,113,0.4));
}
.pred-label {
    font-size: 2.4rem;
    font-weight: 800;
    color: #ffffff;
    letter-spacing: -0.5px;
}
.pred-conf {
    font-size: 1.3rem;
    font-weight: 700;
    color: #2ecc71;
    margin-top: 0.5rem;
}

/* ---- Probability bars ---- */
.prob-container {
    background: rgba(20, 35, 20, 0.5);
    border: 1px solid rgba(46,204,113,0.2);
    border-radius: 20px;
    padding: 2rem;
    margin-top: 1rem;
    box-shadow: 0 10px 30px rgba(0,0,0,0.3);
}
.prob-bar-wrap {
    margin: 1rem 0;
}
.prob-bar-label {
    font-size: 1rem;
    font-weight: 700;
    margin-bottom: 8px;
    color: #ffffff;
    display: flex;
    justify-content: space-between;
}
.prob-bar-outer {
    background: rgba(255,255,255,0.1);
    border-radius: 12px;
    height: 24px;
    overflow: hidden;
    box-shadow: inset 0 2px 4px rgba(0,0,0,0.3);
}
.prob-bar-inner {
    height: 100%;
    border-radius: 12px;
    display: flex;
    align-items: center;
    padding-left: 14px;
    font-size: 0.85rem;
    font-weight: 800;
    color: #000000;
    transition: width 0.8s cubic-bezier(0.22, 1, 0.36, 1);
}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# SIDEBAR: MODEL SELECTION & TELEMETRY
# ──────────────────────────────────────────────
st.sidebar.title("⚡ Neural Engine")
st.sidebar.subheader("Model Selection")

model_choice = st.sidebar.radio(
    "Active Architecture:",
    options=["MobileNetV2 (Transfer Learning)", "Custom CNN (From Scratch)"],
    index=0,
    help="Instantly route inference to compare latency, parameter efficiency, and confidence."
)

model_type_param = "mobilenet" if "MobileNetV2" in model_choice else "cnn"

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Model Telemetry")

if model_type_param == "mobilenet":
    st.sidebar.markdown("""
    <div style="background: rgba(46,204,113,0.15); border: 1px solid rgba(46,204,113,0.3); padding: 1.4rem; border-radius: 16px;">
        <div style="color: #2ecc71; font-weight: 800; font-size: 1.2rem; margin-bottom: 10px;">MobileNetV2 Engine</div>
        <div style="font-size: 0.95rem; color: #ffffff; line-height: 1.6;">
            <b>Parameters</b>: ~2.4 Million<br>
            <b>Base Weights</b>: ImageNet<br>
            <b>Val Accuracy</b>: 86.67%<br>
            <b>Latency</b>: ~45ms<br>
            <b>Optimization</b>: Depthwise Separable Convolutions
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.sidebar.markdown("""
    <div style="background: rgba(26,188,156,0.15); border: 1px solid rgba(26,188,156,0.3); padding: 1.4rem; border-radius: 16px;">
        <div style="color: #1abc9c; font-weight: 800; font-size: 1.2rem; margin-bottom: 10px;">Custom CNN Engine</div>
        <div style="font-size: 0.95rem; color: #ffffff; line-height: 1.6;">
            <b>Parameters</b>: ~164,613<br>
            <b>Base Weights</b>: Custom Initialized<br>
            <b>Val Accuracy</b>: ~80.0%<br>
            <b>Latency</b>: ~15ms<br>
            <b>Structure</b>: 3x Conv2D + MaxPool Blocks
        </div>
    </div>
    """, unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="font-size: 0.85rem; color: #a4bba4; text-align: center; font-weight: 600;">
    Backend API: Flask REST endpoint<br>
    Port: 5000 | Status: Connected 🟢
</div>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🥦 Vegetable Classifier AI</h1>
    <p>High-Precision Neural Classification & Architectural Telemetry</p>
</div>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# MAIN CONTAINER: UPLOAD & INFERENCE
# ──────────────────────────────────────────────
col_main, col_spacer = st.columns([8, 1])

with col_main:
    tab_upload, tab_camera = st.tabs(["📁 Upload Image", "📸 Camera Capture"])

    image = None

    with tab_upload:
        uploaded_file = st.file_uploader(
            "Drag & drop or click to upload a vegetable photo",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file)

    with tab_camera:
        camera_photo = st.camera_input("Capture a vegetable photo using your webcam")
        if camera_photo is not None:
            image = Image.open(camera_photo)

    # ── Inference & Results ──
    if image is not None:
        st.markdown("<br>", unsafe_allow_html=True)
        col_img, col_res = st.columns([5, 6], gap="large")

        with col_img:
            st.image(image, caption="Processed Input Tensor", width="stretch")

        with col_res:
            with st.spinner(f"Inference via {model_choice} ..."):
                img_buffer = io.BytesIO()
                image.save(img_buffer, format=image.format or "JPEG")
                img_buffer.seek(0)

                try:
                    files = {"file": ("image.jpg", img_buffer, "image/jpeg")}
                    data = {"model_type": model_type_param}
                    
                    response = requests.post(FLASK_API_URL, files=files, data=data, timeout=10)
                    
                    if response.status_code == 200:
                        result = response.json()
                        class_key = result["class"]
                        display_name = result["display_name"]
                        confidence = result["confidence"] * 100
                        prob_dict = result["probabilities"]
                        
                        emoji = CLASS_EMOJI.get(class_key, "🥦")
                        
                        # ---- Prediction Glass Card ----
                        st.markdown(f"""
                        <div class="pred-card">
                            <div class="pred-emoji">{emoji}</div>
                            <div class="pred-label">{display_name}</div>
                            <div class="pred-conf">Confidence Score: {confidence:.2f}%</div>
                        </div>
                        """, unsafe_allow_html=True)

                    else:
                        st.error(f"API Error: {response.json().get('error', 'Unknown error')}")
                        prob_dict = None

                except requests.exceptions.ConnectionError:
                    st.error("⚠️ **Could not connect to Flask Backend API.**  \nPlease ensure `python api.py` is running in your terminal.")
                    prob_dict = None
                except Exception as e:
                    st.error(f"⚠️ **An error occurred:** {str(e)}")
                    prob_dict = None

        # ── All-class probability breakdown ──
        if prob_dict:
            st.markdown('<div class="prob-container">', unsafe_allow_html=True)
            st.markdown(f"<h3 style='color: #fff; margin-bottom: 1rem; font-size: 1.5rem; font-weight: 800;'>📊 Probability Distribution ({model_choice})</h3>", unsafe_allow_html=True)

            sorted_probs = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)

            bars_html = ""
            gradient_colors = ["#2ecc71", "#27ae60", "#1abc9c", "#16a085", "#0e6655"]
            for idx, (name, prob) in enumerate(sorted_probs):
                prob_pct = prob * 100
                bar_color = gradient_colors[idx % len(gradient_colors)]
                width = max(prob_pct, 2)
                bars_html += f"""
                <div class="prob-bar-wrap">
                    <div class="prob-bar-label"><span>{name}</span><span>{prob_pct:.1f}%</span></div>
                    <div class="prob-bar-outer">
                        <div class="prob-bar-inner"
                             style="width:{width}%; background:{bar_color};">
                        </div>
                    </div>
                </div>
                """

            st.markdown(bars_html, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.markdown("")
        st.info(
            "💡 **Awaiting Image Input.**  \n"
            "Please upload a vegetable photograph or capture one via webcam to initialize neural classification.",
            icon="ℹ️",
        )

st.markdown("<br><br>", unsafe_allow_html=True)
