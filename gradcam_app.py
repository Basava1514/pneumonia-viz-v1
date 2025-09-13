# streamlit_app.py
import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch

from utils.gradcam_inf import (
    load_vit_model,
    run_inference,
    register_gradcam_hooks,
    generate_gradcam,
    transform   # ✅ use torchvision transform
)

# -------------------------------
# App Config
# -------------------------------
st.set_page_config(page_title="Pneumonia Detection with ViT", layout="wide")
st.title("🫁 Pneumonia Detection using Vision Transformer")

# -------------------------------
# Load model
# -------------------------------
@st.cache_resource
def load_model():
    model = load_vit_model("vit_pneumonia.pt")
    activations, gradients = register_gradcam_hooks(model)
    return model, activations, gradients

model, activations, gradients = load_model()

# -------------------------------
# File uploader
# -------------------------------
uploaded_file = st.file_uploader("Upload a chest X-ray", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("📷 Uploaded Image")
    st.image(image, caption="Input Chest X-ray", use_container_width=True)

    # ---------------------------
    # Run inference
    # ---------------------------
    results = run_inference(model, image)
    st.subheader("📊 Prediction Results")
    st.write(f"**Prediction:** {results['prediction']}")
    st.write(f"**Confidence:** {results['confidence']:.4f}")

    # Show all class probabilities
    st.json(results["all_probs"])

    # ---------------------------
    # Grad-CAM
    # ---------------------------
    st.subheader("🔥 Grad-CAM Visualization")

    # ✅ Use torchvision transform (same as training/inference)
    img_tensor = transform(image).unsqueeze(0)

    cam = generate_gradcam(
        img_tensor.squeeze(0), model,
        activations, gradients,
        class_idx=None
    )

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = np.array(image.resize((224, 224))) * 0.5 + heatmap * 0.5
    overlay = np.uint8(overlay)

    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Original X-ray", use_container_width=True)
    with col2:
        st.image(overlay, caption="Grad-CAM Heatmap", use_container_width=True)
    st.write("Grad-CAM highlights the regions of the image that contributed most to the model's prediction.")   