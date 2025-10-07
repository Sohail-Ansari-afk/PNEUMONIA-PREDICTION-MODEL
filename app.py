import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image
import tempfile
import os

st.set_page_config(page_title="Pneumonia Detector 🫁", layout="centered")

st.title("🩺 Pneumonia Detection using CNN")
st.write("Upload a Chest X-ray image, and the AI model will predict if it's **Normal** or **Pneumonia**.")

# Load your model
@st.cache_resource
def load_cnn_model():
    # Prefer new Keras format if present; fall back to legacy .h5
    candidate_paths = [
        "pneumonia_cnn_model.keras",
        "pneumonia_cnn_model.h5",
    ]

    for path in candidate_paths:
        if os.path.exists(path):
            try:
                return load_model(path)
            except Exception as e:
                st.error(f"Failed to load model from `{path}`: {e}")
                st.stop()

    st.error(
        "Model file not found. Place `pneumonia_cnn_model.keras` or `pneumonia_cnn_model.h5` in the project root."
    )
    st.stop()

model = load_cnn_model()

# Upload image
uploaded_file = st.file_uploader("Upload Chest X-Ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image according to model input shape
    input_shape = getattr(model, "input_shape", None)
    # Default to 128x128x3 if not available
    target_h = 128
    target_w = 128
    target_c = 3
    if input_shape and len(input_shape) == 4:
        # shape: (None, H, W, C)
        if input_shape[1] is not None:
            target_h = int(input_shape[1])
        if input_shape[2] is not None:
            target_w = int(input_shape[2])
        if input_shape[3] is not None:
            target_c = int(input_shape[3])

    img_array = np.array(image)
    if target_c == 1:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        gray = cv2.resize(gray, (target_w, target_h))
        img = gray.astype("float32") / 255.0
        img = np.expand_dims(img, axis=-1)  # (H, W, 1)
    else:
        color = cv2.resize(img_array, (target_w, target_h))
        img = color.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)  # (1, H, W, C)

    # Prediction
    prediction = model.predict(img)[0][0]
    label = "PNEUMONIA" if prediction > 0.5 else "NORMAL"
    confidence = prediction if prediction > 0.5 else 1 - prediction

    st.subheader(f"Prediction: **{label}**")
    st.metric(label="Confidence", value=f"{confidence*100:.2f}%")

    if label == "PNEUMONIA":
        st.warning("⚠️ This X-ray indicates possible pneumonia.")
    else:
        st.success("✅ This X-ray appears normal.")
