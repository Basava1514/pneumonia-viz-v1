# streamlit_app.py
import os
import json
import time
import tempfile
from datetime import datetime
from io import BytesIO

import streamlit as st
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import ViTImageProcessor, ViTForImageClassification

# ---- project utils ----
from utils.inference import (
    load_pretrained_model,   # returns (model, meta)
    load_pretrained_unet,
    run_inference,
    export_pdf_report,
    handle_feedback,
)

# =========================
# Paths / constants
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FEEDBACK_DATASET = os.path.join(BASE_DIR, "feedback_dataset.jsonl")  # labeled data for retraining
MODEL_META_JSON = os.path.join(BASE_DIR, "model_meta.json")
FINETUNED_MODEL_PATH = os.path.join(BASE_DIR, "vit_pneumonia.pt")
PRETRAINED_NAME = "google/vit-base-patch16-224-in21k"

RETRAIN_THRESHOLD = 1   # fine-tune whenever >= 1 new labeled example since last run
FINETUNE_EPOCHS = 3
FINETUNE_LR = 1e-4
BATCH_SIZE = 4

os.makedirs(os.path.join(BASE_DIR, "reports"), exist_ok=True)

# =========================
# Helpers for meta state
# =========================
def _read_meta():
    if not os.path.exists(MODEL_META_JSON):
        return {"last_trained_count": 0, "last_trained_at": None}
    try:
        with open(MODEL_META_JSON, "r") as f:
            return json.load(f)
    except Exception:
        return {"last_trained_count": 0, "last_trained_at": None}

def _write_meta(meta: dict):
    try:
        with open(MODEL_META_JSON, "w") as f:
            json.dump(meta, f)
    except Exception:
        pass

def _append_feedback_row(row: dict):
    with open(FEEDBACK_DATASET, "a") as f:
        f.write(json.dumps(row) + "\n")

def _read_labeled_entries():
    if not os.path.exists(FEEDBACK_DATASET):
        return []
    rows = []
    with open(FEEDBACK_DATASET, "r") as f:
        for line in f:
            try:
                j = json.loads(line)
                if "label" in j and j["label"] in [0, 1]:
                    rows.append(j)
            except Exception:
                continue
    return rows

# =========================
# Tiny dataset for fine-tuning
# =========================
class FeedbackDataset(Dataset):
    def __init__(self, entries, processor: ViTImageProcessor):
        self.entries = entries
        self.processor = processor

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        e = self.entries[idx]
        img = Image.open(e["image_path"]).convert("RGB")
        # Ensure identical preprocessing as training/inference
        inputs = self.processor(
            images=img,
            return_tensors="pt",
            do_resize=True,
            size={"height": 224, "width": 224},
        )
        pixel_values = inputs["pixel_values"].squeeze(0)  # [3,224,224]
        label = torch.tensor([e["label"]], dtype=torch.float32)  # BCE with 1 logit
        return pixel_values, label

# =========================
# Fine-tune only classifier head
# =========================
def finetune_classifier_head(base_model: ViTForImageClassification, processor, entries, device):
    """
    Freezes backbone; fine-tunes classifier head on physician-labeled feedback.
    Saves to FINETUNED_MODEL_PATH and returns the updated model.
    """
    model = base_model
    # freeze everything except classifier
    for p in model.parameters():
        p.requires_grad = False
    for p in model.classifier.parameters():
        p.requires_grad = True

    ds = FeedbackDataset(entries, processor)
    dl = DataLoader(ds, batch_size=min(BATCH_SIZE, max(1, len(ds))), shuffle=True)

    model.train()
    model.to(device)
    opt = torch.optim.AdamW(model.classifier.parameters(), lr=FINETUNE_LR)
    loss_fn = nn.BCEWithLogitsLoss()

    for _ in range(FINETUNE_EPOCHS):
        for pixel_values, labels in dl:
            pixel_values = pixel_values.to(device)
            labels = labels.to(device)
            opt.zero_grad()
            out = model(pixel_values=pixel_values)
            loss = loss_fn(out.logits.view(-1, 1), labels)
            loss.backward()
            opt.step()

    # save updated head (full state dict for simplicity)
    torch.save(model.state_dict(), FINETUNED_MODEL_PATH)

    # back to eval
    model.eval()
    return model

def maybe_retrain(vit_model, processor, device):
    """
    If labeled entries >= RETRAIN_THRESHOLD and > last_trained_count → train and hot-swap weights.
    """
    labeled = _read_labeled_entries()
    meta = _read_meta()
    if len(labeled) >= RETRAIN_THRESHOLD and len(labeled) > meta.get("last_trained_count", 0):
        vit_model = finetune_classifier_head(vit_model, processor, labeled, device)
        meta["last_trained_count"] = len(labeled)
        meta["last_trained_at"] = datetime.utcnow().isoformat()
        _write_meta(meta)
        return vit_model, True
    return vit_model, False

# =========================
# Model Loader (cache)
# =========================
@st.cache_resource
def load_models():
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    # utils.inference.load_pretrained_model now returns (model, meta)
    vit_model, meta = load_pretrained_model(device)

    # Build processor with saved mean/std from training if present
    processor = ViTImageProcessor.from_pretrained(PRETRAINED_NAME)
    if isinstance(meta, dict):
        if meta.get("image_mean") is not None and meta.get("image_std") is not None:
            # ensure processor normalization matches training
            processor.image_mean = meta["image_mean"]
            processor.image_std = meta["image_std"]

    # If a finetuned head exists, load it on top
    if os.path.exists(FINETUNED_MODEL_PATH):
        try:
            vit_model.load_state_dict(torch.load(FINETUNED_MODEL_PATH, map_location=device), strict=False)
        except Exception as e:
            st.warning(f"Could not load finetuned head: {e}")

    segmentation_model = load_pretrained_unet(device)
    return vit_model, segmentation_model, processor, device

vit_model, segmentation_model, processor, device = load_models()

# =========================
# UI
# =========================
st.title("🫁 Pneumonia Detection (ViT + U-Net) with XAI & Auto-Retraining")
st.write("Upload an X-ray, review explainability, add physician feedback, and auto-fine-tune the classifier head when enough feedback arrives (threshold = 1).")

uploaded_file = st.file_uploader("Upload Chest X-ray", type=["png", "jpg", "jpeg"])

if uploaded_file:
    # Read the file once as BYTES (fixes the memoryview issue)
    contents = uploaded_file.getvalue()  # bytes

    # Preview (Streamlit accepts bytes, PIL image, numpy array, or URL)
    st.image(contents, caption="Uploaded X-ray", use_container_width=True)

    # Persist the upload to a temp file so we can reference it later
    suffix = os.path.splitext(uploaded_file.name)[1] or ".png"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    if st.button("Run Inference"):
        with st.spinner("Analyzing X-ray..."):
            results = run_inference(
                image_path=tmp_path,
                device=device,
                segmentation_model=segmentation_model,
                vit_model=vit_model,
                processor=processor,
            )

        prob = results["probability"]
        st.subheader("Prediction Result")
        st.write(f"**Pneumonia Probability:** {prob:.2%}")
        st.write(f"**Trust Score:** {results['trust_score']:.2f}/100")
        if "best_thr" in results:
            st.write(f"**Operating Threshold:** {results['best_thr']:.2f}")
        st.markdown(results["explanation"])

        st.subheader("Visual Explanations")
        # NEW: show masks explicitly
        if results.get("lung_mask_binary") is not None:
            st.image(results["lung_mask_binary"], caption="Lung Mask (binary)", use_container_width=True)
        if results.get("lung_mask_soft") is not None:
            st.image(results["lung_mask_soft"], caption="Lung Mask (soft)", use_container_width=True)

        # Existing overlay & explainers
        if results.get("lung_overlay") is not None:
           st.image(results["lung_overlay"], caption="Lung Segmentation Overlay", use_container_width=True)
        if results.get("gradcam") is not None:
           st.image(results["gradcam"], caption="Grad-CAM++", use_container_width=True)
        if results.get("vitcx") is not None:
           st.image(results["vitcx"], caption="ViT-CX", use_container_width=True)
        if results.get("lime") is not None:
            st.image(results["lime"], caption="LIME", use_container_width=True)
        if results.get("combined_vitcx_lime") is not None:
            st.image(results["combined_vitcx_lime"], caption="Combined (GradCAM + LIME)", use_container_width=True)


        # -------------------------
        # Physician Feedback block
        # -------------------------
        st.subheader("📝 Physician Feedback")
        col1, col2 = st.columns([1, 2])
        with col1:
            label_str = st.selectbox(
                "Ground truth label",
                ["Select...", "Pneumonia present", "Pneumonia absent", "Uncertain"]
            )
        with col2:
            feedback_text = st.text_area("Clinical notes (optional)")

        # Map to numeric label for training
        label_val = None
        if label_str == "Pneumonia present":
            label_val = 1
        elif label_str == "Pneumonia absent":
            label_val = 0

        # Save feedback (always)
        if st.button("Save Feedback"):
            handle_feedback({
                "image": os.path.basename(tmp_path),
                "feedback": feedback_text,
                "probability": results["probability"],
                "trust_score": results["trust_score"],
                "timestamp": int(time.time())
            })
            st.success("✅ Feedback saved to feedback_log.json")

            # If a usable label was chosen, also save to training dataset
            if label_val is not None:
                _append_feedback_row({
                    "image_path": tmp_path,      # keep full path so we can load the image
                    "label": label_val,
                    "notes": feedback_text,
                    "prob": results["probability"],
                    "ts": int(time.time())
                })
                st.success("✅ Labeled example added to feedback_dataset.jsonl")

        # -------------------------
        # PDF export
        # -------------------------
        if st.button("Generate PDF Report"):
            report_path = export_pdf_report(tmp_path, results, feedback_text=feedback_text)
            with open(report_path, "rb") as f:
                st.download_button(
                    "📥 Download Report",
                    f,
                    file_name=os.path.basename(report_path),
                    mime="application/pdf"
                )

        # -------------------------
        # Auto-retrain (threshold=1)
        # -------------------------
        if st.button("Run Auto-Retrain (if eligible)"):
            with st.spinner("Checking feedback and fine-tuning head if needed..."):
                vit_model, updated = maybe_retrain(vit_model, processor, device)
            if updated:
                st.success("🎯 Fine-tuning complete. Model updated in memory and saved to vit_pneumonia_finetuned.pt")
            else:
                st.info("ℹ️ Not enough new labeled feedback since last training, or nothing to update yet.")

else:
    st.info("Upload a chest X-ray to begin.")
