# 🥦 Local Vegetable Variety Classifier AI 🚀

A complete Deep Learning & Computer Vision pipeline built for the AI Hackathon. This project classifies local vegetable varieties using a fine-tuned **MobileNetV2** transfer learning model and provides a sleek, interactive web interface powered by **Streamlit**.

---

## 🌟 Project Highlights

- **Balanced Pre-Augmented Dataset**: 500 images across 5 distinct local vegetable classes (100 images per class).
- **Transfer Learning**: Built on top of `MobileNetV2` (pre-trained on ImageNet) with custom dense classification heads and dropout regularisation.
- **High Accuracy**: Achieved **~92% Training Accuracy** and **~83% Validation Accuracy** on a lightweight, mobile-friendly architecture.
- **Sleek Web Application**: Interactive UI allowing drag-and-drop image uploads or real-time device camera captures.
- **Rich Visual Feedback**: Instant predictions accompanied by confidence scores, dynamic progress bars, and full probability breakdown charts.

---

## 🏷️ Supported Vegetable Varieties

| Class Label | Local / Common Name | Botanical Name |
| :--- | :--- | :--- |
| 🌶️ `green_chilli` | **Green Chilli** (*Marcha*) | *Capsicum annuum* |
| 🥒 `ivy_gourd` | **Ivy Gourd** (*Tindoda*) | *Coccinia grandis* |
| 🌿 `okra` | **Okra / Lady's Finger** (*Bhinda*) | *Abelmoschus esculentus* |
| 🌱 `peas` | **Green Peas** (*Vatana*) | *Pisum sativum* |
| 🫒 `pointed_gourd` | **Pointed Gourd** (*Parvad / Parwal*) | *Trichosanthes dioica* |

---

## 📂 Repository Structure

```text
Ai-hackathon/
│
├── balanced/                  # 📁 Balanced dataset (5 classes, 100 images each)
├── vegetable_classifier.py    # ⚙️ Phase 1: Model training & evaluation pipeline
├── vegetable_classifier.h5    # 🧠 Trained Keras/TensorFlow model weights
├── training_curves.png        # 📈 Training vs Validation accuracy & loss curves
├── confusion_matrix.png       # 📊 Heatmap of the model's confusion matrix
├── app.py                     # 🌐 Phase 2: Streamlit interactive web application
└── README.md                  # 📖 Project documentation & run instructions
```

---

## 🚀 How to Run the Web Application (Phase 2)

### 1️⃣ Prerequisites & Installation
Ensure you have Python 3.8+ installed. Install the required dependencies using `pip`:

```bash
pip install streamlit tensorflow keras pillow numpy
```

### 2️⃣ Launching the Streamlit App
Run the following command from the root directory of the project:

```bash
streamlit run app.py
```

### 3️⃣ Usage
1. Open the local URL provided by Streamlit (usually `http://localhost:8501`).
2. Choose between the **📁 Upload Image** tab or the **📷 Take Photo** tab (device camera).
3. Watch the AI classify the vegetable instantly and display the confidence breakdown!

---

## 📊 Model Evaluation & Performance

During the Phase 1 training checkpoint (15 epochs), the model demonstrated robust learning capabilities:

- **Training Accuracy**: `91.88%`
- **Validation Accuracy**: `82.67%`
- **Training Loss**: `0.3810`
- **Validation Loss**: `0.4776`

### Per-Class F1-Scores:
- `green_chilli`: **0.98**
- `ivy_gourd`: **0.72**
- `okra`: **0.89**
- `peas`: **0.83**
- `pointed_gourd`: **0.71**


