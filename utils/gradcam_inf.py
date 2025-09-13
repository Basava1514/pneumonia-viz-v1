# utils/inference.py
import os
import sys
import json
import cv2
import numpy as np
from typing import Optional, Tuple, Dict, Any, List
from datetime import datetime
from PIL import Image
import collections

import torch
from torch import nn
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

from transformers import ViTForImageClassification, ViTImageProcessor

# -------------------------------
# Device
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# Image preprocessing
# -------------------------------
transform = Compose([
    Resize((224, 224)),
    ToTensor(),
    Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# -------------------------------
# Load fine-tuned ViT model
# -------------------------------
import torch
import torch.nn as nn
from transformers import ViTForImageClassification

def load_vit_model(checkpoint_path: str, num_classes: int = 1, device: str = "cpu"):
    """
    Load ViT model with custom classifier head that matches checkpoint.
    """
    # Load base pretrained model
    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224"
    )

    # Replace classifier head to match your training setup
    hidden_size = model.config.hidden_size
    model.classifier = nn.Linear(hidden_size, num_classes)

    # Load your saved weights
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)

    model.to(device)
    model.eval()
    return model


# -------------------------------
# Grad-CAM helper
# -------------------------------
def register_gradcam_hooks(model):
    activations = {}
    gradients = {}

    def forward_hook(module, input, output):
        activations["value"] = output

    def backward_hook(module, grad_in, grad_out):
        grads = grad_out[0] if isinstance(grad_out, tuple) else grad_out
        gradients["value"] = grads

    # Hook last encoder layer output
    target_layer = model.vit.encoder.layer[-1].output
    target_layer.register_forward_hook(forward_hook)
    target_layer.register_backward_hook(backward_hook)

    return activations, gradients

def generate_gradcam(image_tensor, model, activations, gradients, class_idx=None):
    """
    Generates Grad-CAM heatmap for given image.
    """
    image_tensor = image_tensor.unsqueeze(0).to(device)

    outputs = model(image_tensor)
    if class_idx is None:
        class_idx = outputs.logits.argmax(dim=-1).item()

    # Backprop
    model.zero_grad()
    loss = outputs.logits[:, class_idx].sum()
    loss.backward()

    # Extract values
    activ = activations["value"]  # shape: [1, seq_len, hidden]
    grads = gradients["value"]    # shape: [1, seq_len, hidden]

    # Global average pooling over hidden dim
    weights = grads.mean(dim=1)  # [1, hidden]
    cam = (activ * weights.unsqueeze(1)).sum(dim=-1)  # [1, seq_len]

    # Remove CLS token, reshape to patch grid
    cam = cam[:, 1:]  # drop [CLS]
    side = int(cam.size(1) ** 0.5)
    cam = cam.reshape(1, side, side).detach().cpu().numpy()
    cam = np.maximum(cam, 0)
    cam = cam / cam.max()

    # Upscale to 224x224
    cam = cv2.resize(cam[0], (224, 224))
    return cam

# -------------------------------
# Run inference
# -------------------------------
import torch
from torchvision import transforms
from PIL import Image
import numpy as np

def run_inference(model, image: Image.Image, device: str = "cpu"):
    """
    Run inference on a single image using binary classification ViT model.
    """

    # Preprocessing (same as training)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
    ])

    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        logits = outputs.logits  # shape [1, 1]

        # Apply sigmoid for binary probability
        probs = torch.sigmoid(logits).cpu().numpy().flatten()

        # Convert to dict: Normal vs Pneumonia
        pred_class = "Pneumonia" if probs[0] > 0.5 else "Normal"

        result = {
            "logit": float(logits.item()),
            "probability": float(probs[0]),
            "prediction": pred_class
        }

    return result
