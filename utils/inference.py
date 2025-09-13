import sys
import os

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_dir)

import json
import torch
import numpy as np
import cv2
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from transformers import ViTForImageClassification, ViTImageProcessor
from fpdf import FPDF
from models.unet import UNet
from scipy.ndimage import gaussian_filter
import torch.nn.functional as F
import matplotlib.pyplot as plt

# =========================
# Paths and constants
# =========================
PRETRAINED_NAME = "google/vit-base-patch16-224-in21k"
CKPT_DIR = "/Users/basava/Desktop/chest_xray/project"
BEST_CKPT = os.path.join(CKPT_DIR, "vit_pneumonia_best.pt")
MAIN_CKPT = os.path.join(CKPT_DIR, "vit_pneumonia.pt")
DEFAULT_UNET_PATH = os.path.join(CKPT_DIR, "unet_lung_segmentation.pt")

REPORT_DIR = "reports"
FEEDBACK_LOG = "feedback_log.json"
MASK_NPY_DIR = "lung_masks_npy"
os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(MASK_NPY_DIR, exist_ok=True)

# =========================
# Utility
# =========================
def _device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

# =========================
# Model loaders
# =========================
def load_pretrained_model(device):
    """
    Loads ViT with single-logit head and restores your fine-tuned weights.
    Returns (model, meta) where meta may contain image_mean, image_std, best_thr.
    """
    # Create base model with a single-logit head
    model = ViTForImageClassification.from_pretrained(
    PRETRAINED_NAME,
    num_labels=1,
    ignore_mismatched_sizes=True,
    attn_implementation="eager",
    ).to(device).eval()


    meta = {"best_thr": 0.50, "image_mean": None, "image_std": None}

    ckpt_path = BEST_CKPT if os.path.exists(BEST_CKPT) else MAIN_CKPT
    if not os.path.exists(ckpt_path):
        print(f"[ViT] WARNING: Checkpoint not found at {ckpt_path}. Using base pretrained weights.")
        return model, meta

    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        model_state = state["model_state_dict"]
    else:
        model_state = state

    missing, unexpected = model.load_state_dict(model_state, strict=False)
    if missing or unexpected:
        print(f"[ViT] load_state_dict notes | missing: {missing} | unexpected: {unexpected}")

    # Restore meta if available
    meta["image_mean"] = state.get("image_mean", None) if isinstance(state, dict) else None
    meta["image_std"]  = state.get("image_std", None)  if isinstance(state, dict) else None
    meta["best_thr"]   = float(state.get("best_thr", 0.50)) if isinstance(state, dict) else 0.50

    return model, meta


def load_pretrained_unet(device, in_channels=1, out_channels=1, weight_path: str = None):
    unet = UNet(in_channels=in_channels, out_channels=out_channels)
    weight_path = weight_path or DEFAULT_UNET_PATH
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"UNet weights not found at {weight_path}")
    state_dict = torch.load(weight_path, map_location=device)
    unet.load_state_dict(state_dict)
    return unet.to(device).eval()

# =========================
# Preprocessing
# =========================
def _build_processor(image_mean=None, image_std=None):
    proc = ViTImageProcessor.from_pretrained(PRETRAINED_NAME)
    # If meta provided mean/std, override so preprocessing matches training
    if image_mean is not None and image_std is not None:
        proc.image_mean = image_mean
        proc.image_std = image_std
    return proc

def preprocess_for_unet(image_path, device):
    image = Image.open(image_path).convert("L").resize((256, 256))
    img_arr = np.array(image, dtype=np.float32) / 255.0
    return torch.from_numpy(img_arr).unsqueeze(0).unsqueeze(0).to(device)

def preprocess_for_vit(image_path, processor: ViTImageProcessor, device):
    img_pil = Image.open(image_path).convert("RGB")
    # keep explicit resize; drop do_center_crop (not supported)
    inputs = processor(
        images=img_pil,
        return_tensors="pt",
        do_resize=True,
        size={"height": 224, "width": 224},
    )
    image_tensor = inputs["pixel_values"].to(device)            # (1,3,224,224)
    image_224 = np.array(img_pil.resize((224, 224)))            # for visuals
    return image_tensor, image_224

# =========================     
import scipy
from scipy.ndimage import binary_fill_holes

def _postprocess_lung_mask(prob_map: np.ndarray, min_area_ratio: float = 0.02, dilate_iter: int = 2) -> np.ndarray:
    """
    prob_map: float32 [H,W] in [0,1]
    Steps: Otsu binarize -> keep 2 largest CCs -> fill holes -> light dilation.
    Returns binary float32 {0,1}.
    """
    h, w = prob_map.shape
    area = h * w

    # Otsu threshold (fallback 0.5)
    try:
        thr, _ = cv2.threshold((prob_map * 255).astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thr = thr / 255.0
    except Exception:
        thr = 0.5

    m = (prob_map >= thr).astype(np.uint8)

    # Keep only the 2 largest connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if num_labels > 1:
        sizes = stats[1:, cv2.CC_STAT_AREA]
        order = np.argsort(sizes)[::-1]  # largest first
        keep = set([1 + i for i in order[:2]])  # component ids to keep
        m = np.isin(labels, list(keep)).astype(np.uint8)

    # Remove tiny blobs by area ratio
    min_area = max(1, int(min_area_ratio * area))
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    m2 = np.zeros_like(m)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            m2[labels == i] = 1
    m = m2

    # Fill holes
    m = binary_fill_holes(m.astype(bool)).astype(np.uint8)

    # Gentle smooth + dilation to avoid under-segmentation
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN,  np.ones((3,3), np.uint8))
    if dilate_iter > 0:
        m = cv2.dilate(m, np.ones((3,3), np.uint8), iterations=dilate_iter)

    return m.astype(np.float32)


# =========================
# Lung mask helpers
# =========================
def get_or_generate_lung_mask(image_path, unet_model, device):
    """
    Returns (soft_mask_224 [H,W] float32 in [0,1], binary_mask_224 {0,1} float32).
    Caches SOFT mask to .npy; postprocesses to binary each call.
    """
    base = os.path.splitext(os.path.basename(image_path))[0]
    mask_npy_path = os.path.join(MASK_NPY_DIR, f"{base}.npy")

    # ---- Run UNet @224 for alignment with ViT ----
    img_gray = Image.open(image_path).convert("L").resize((224, 224))
    unet_input = torch.from_numpy(np.array(img_gray, dtype=np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(device)

    if os.path.exists(mask_npy_path):
        soft = np.load(mask_npy_path).astype(np.float32)
        # tolerate cached 256 -> resize to 224
        if soft.shape != (224, 224):
            soft = cv2.resize(soft, (224, 224), interpolation=cv2.INTER_LINEAR)
    else:
        with torch.no_grad():
            logits = unet_model(unet_input)  # expect (1,1,224,224) or (1,1,256,256)
            prob = torch.sigmoid(logits).squeeze().cpu().numpy().astype(np.float32)
        # If UNet outputs 256, downscale to 224 for consistency
        soft = prob if prob.shape == (224, 224) else cv2.resize(prob, (224, 224), interpolation=cv2.INTER_LINEAR)
        np.save(mask_npy_path, soft)

    binary = _postprocess_lung_mask(soft)
    return soft, binary


def make_lung_visuals(base_rgb: np.ndarray, lung_mask_224_binary: np.ndarray, target_size=None):
    target_h, target_w = (target_size or base_rgb.shape[:2])

    base = base_rgb
    if base.ndim == 2:
        base = cv2.cvtColor(base, cv2.COLOR_GRAY2RGB)
    if base.shape[:2] != (target_h, target_w):
        base = cv2.resize(base, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    lung = lung_mask_224_binary
    if lung.shape != (target_h, target_w):
        lung = cv2.resize(lung, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    lung = (lung > 0.5).astype(np.float32)

    # Pure binary mask image for UI
    lung_mask_img = (lung * 255).astype(np.uint8)

    # Cyan overlay
    cyan = np.zeros_like(base, dtype=np.uint8); cyan[...,1]=255; cyan[...,2]=255
    overlay = (base.astype(np.float32) * (1.0 - 0.35*lung[...,None]) +
               cyan.astype(np.float32) * (0.35*lung[...,None]))
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    return Image.fromarray(lung_mask_img), Image.fromarray(overlay)


# =========================
# Grad-CAM++ (with safe fallback hook)
# =========================
def grad_cam_plus_plus_vit(model, image_tensor, lung_mask: np.ndarray, class_idx=None, device="cpu", target_size=None):
    import torch
    th, tw = target_size or (224, 224)
    model = model.to(device).eval()
    image_tensor = image_tensor.to(device)

    activations, gradients = {}, {}

    # Preferred hook target (works in current HF ViT)
    target_module = None
    try:
        target_module = model.vit.encoder.layer[-1].output.dense
    except Exception:
        # Fallback: hook the whole block output if dense not available
        target_module = model.vit.encoder.layer[-1].output

    def fwd_hook(_m,_i,o): activations["value"] = o
    def bwd_hook(_m,_gi,go): gradients["value"] = go[0]
    h1 = target_module.register_forward_hook(fwd_hook)
    h2 = target_module.register_full_backward_hook(bwd_hook)

    outputs = model(image_tensor)
    if class_idx is None:
        class_idx = int(outputs.logits.argmax(dim=-1).item())

    model.zero_grad(set_to_none=True)
    scalar = outputs.logits.view(-1)[0] if outputs.logits.numel()==1 else outputs.logits[:,class_idx].sum()
    scalar.backward()

    activ = activations.get("value", None)
    grads = gradients.get("value", None)
    if activ is None or grads is None:
        h1.remove(); h2.remove()
        raise RuntimeError("Grad-CAM++ hooks failed to capture activations/gradients")

    numerator = grads.pow(2)
    denominator = 2*grads.pow(2) + (activ*grads.pow(3)).sum(dim=-1, keepdim=True)
    denominator = torch.where(denominator!=0, denominator, torch.ones_like(denominator))
    alphas = numerator / denominator
    weights = (alphas * F.relu(grads)).sum(dim=-1)

    cam = (weights.unsqueeze(-1) * activ).sum(dim=-1)  # (B, Seq)
    cam = cam[:,1:]  # drop CLS
    side = int(cam.size(1)**0.5)
    cam = cam.reshape(1, side, side).detach().cpu().numpy()[0]
    cam = np.maximum(cam, 0)
    cam /= (cam.max() + 1e-8)
    cam = cv2.resize(cam, (tw, th), interpolation=cv2.INTER_LINEAR)

    lung = cv2.resize(lung_mask, (tw, th), interpolation=cv2.INTER_NEAREST)
    lung = (lung > 0.5).astype(np.float32)
    cam = cam * lung
    if cam.max() > 0: cam /= cam.max()
    cam = gaussian_filter(cam, sigma=4)

    base = image_tensor.squeeze(0).permute(1,2,0).detach().cpu().numpy()
    base = ((base - base.min())/(base.max()-base.min()+1e-8)*255).astype(np.uint8)
    base = cv2.resize(base, (tw, th), interpolation=cv2.INTER_LINEAR)
    if base.ndim==2: base = cv2.cvtColor(base, cv2.COLOR_GRAY2RGB)

    heatmap = cv2.applyColorMap((cam*255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(base, 0.6, heatmap, 0.4, 0)

    h1.remove(); h2.remove()
    return cam.astype(np.float32), Image.fromarray(overlay)

# =========================
# ViT-CX
# =========================
def vit_cx(model, vit_tensor, lung_mask: np.ndarray, class_idx=None, device="cpu", target_size=None):
    th, tw = target_size or (224, 224)
    try:
        model = model.to(device).eval()
        vit_tensor = vit_tensor.to(device)

        outputs = model(vit_tensor, output_attentions=True)
        if class_idx is None:
            class_idx = int(outputs.logits.argmax(dim=-1).item())

        attn = outputs.attentions[-1].mean(1)     # (B, Seq, Seq)
        cls_attn = attn[:,0,1:]                   # (B, Seq-1)
        side = int(cls_attn.size(-1) ** 0.5)
        attn_map = cls_attn.reshape(1, side, side).detach().cpu().numpy()[0]
        attn_map = attn_map / (attn_map.max() + 1e-8)

        attn_map = cv2.resize(attn_map, (tw, th), interpolation=cv2.INTER_LINEAR)
        lung = cv2.resize(lung_mask, (tw, th), interpolation=cv2.INTER_NEAREST)
        lung = (lung > 0.5).astype(np.float32)
        attn_map = attn_map * lung
        if attn_map.max() > 0: attn_map /= attn_map.max()

        base = vit_tensor.squeeze(0).permute(1,2,0).detach().cpu().numpy()
        base = ((base - base.min())/(base.max()-base.min()+1e-8)*255).astype(np.uint8)
        base = cv2.resize(base, (tw, th), interpolation=cv2.INTER_LINEAR)
        if base.ndim==2: base = cv2.cvtColor(base, cv2.COLOR_GRAY2RGB)

        heatmap = cv2.applyColorMap((attn_map*255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(base, 0.6, heatmap, 0.4, 0)

        return attn_map.astype(np.float32), Image.fromarray(overlay)
    except Exception as e:
        print(f"ERROR in ViT-CX: {e}")
        return None, None

# =========================
# LIME
# =========================
def lime_explanation(img_np: np.ndarray, model, processor, lung_mask: np.ndarray, device="cpu", target_size=None):
    from lime import lime_image
    from skimage.segmentation import mark_boundaries

    th, tw = target_size or (224, 224)
    try:
        model = model.to(device).eval()
        explainer = lime_image.LimeImageExplainer()

        def predict_fn(images):
            pil_imgs = [Image.fromarray(x.astype(np.uint8)) for x in images]
            inputs = processor(
                images=pil_imgs,
                return_tensors="pt",
                do_resize=True, size={"height":224,"width":224},
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                logits = model(**inputs).logits
                probs = torch.sigmoid(logits).detach().cpu().numpy()
            if probs.ndim == 1: probs = probs.reshape(-1,1)
            # LIME expects 2-class probs
            return np.concatenate([1 - probs, probs], axis=1)

        base = cv2.resize(img_np.astype(np.uint8), (tw, th), interpolation=cv2.INTER_LINEAR)

        explanation = explainer.explain_instance(
            base, predict_fn, top_labels=1, hide_color=0, num_samples=1000
        )
        temp, mask = explanation.get_image_and_mask(
            explanation.top_labels[0], positive_only=True, num_features=10, hide_rest=False
        )

        raw_map = (mask > 0).astype(np.float32)
        lime_img = (mark_boundaries(temp/255.0, mask).astype(np.float32) * 255).astype(np.uint8)

        lung = cv2.resize(lung_mask, (tw, th), interpolation=cv2.INTER_NEAREST)
        lung = (lung > 0.5).astype(np.float32)

        raw_map = raw_map * lung
        lime_img = (lime_img.astype(np.float32) * lung[...,None]).astype(np.uint8)

        return raw_map.astype(np.float32), Image.fromarray(lime_img)
    except Exception as e:
        print(f"ERROR in LIME: {e}")
        return None, None

# =========================
# Combined (Grad-CAM++ + LIME)
# =========================
def combined_explanation(model, vit_tensor, processor, lung_mask_np, device="cpu", target_size=None):
    th, tw = target_size or (224, 224)
    try:
        grad_raw, grad_img = grad_cam_plus_plus_vit(model, vit_tensor, lung_mask_np, device=device, target_size=(th, tw))
        base_np = vit_tensor.squeeze().permute(1, 2, 0).detach().cpu().numpy()
        base_np = ((base_np - base_np.min()) / (base_np.max() - base_np.min() + 1e-8) * 255).astype(np.uint8)
        base_np = cv2.resize(base_np, (tw, th), interpolation=cv2.INTER_LINEAR)

        lime_raw, lime_img = lime_explanation(base_np, model, processor, lung_mask_np, device=device, target_size=(th, tw))

        if grad_img is None and lime_img is None:
            return None, None
        if grad_img is None:
            return lime_raw, lime_img
        if lime_img is None:
            return grad_raw, grad_img

        grad_np = cv2.resize(np.array(grad_img), (tw, th), interpolation=cv2.INTER_LINEAR)
        lime_np = cv2.resize(np.array(lime_img), (tw, th), interpolation=cv2.INTER_LINEAR)
        combined = cv2.addWeighted(grad_np.astype(np.uint8), 0.6, lime_np.astype(np.uint8), 0.8, 0)

        g = cv2.resize(grad_raw.astype(np.float32), (tw, th), interpolation=cv2.INTER_NEAREST)
        l = cv2.resize(lime_raw.astype(np.float32), (tw, th), interpolation=cv2.INTER_NEAREST)
        raw_map = np.maximum(g, l).astype(np.float32)

        return raw_map, Image.fromarray(combined)
    except Exception as e:
        print(f"ERROR during combined explanation: {e}")
        return None, None

# =========================
# Metrics
# =========================
def compute_lung_focus(xai_map, lung_mask_np):
    try:
        if isinstance(xai_map, Image.Image):
            xai_map = np.array(xai_map)
        xai_gray = xai_map.mean(axis=2) if xai_map.ndim == 3 else xai_map
        xai_gray = (xai_gray - xai_gray.min()) / (xai_gray.max() - xai_gray.min() + 1e-8)

        h, w = xai_gray.shape
        lung = cv2.resize(lung_mask_np, (w, h), interpolation=cv2.INTER_NEAREST)
        lung = (lung > 0.5).astype(np.float32)

        attr = (xai_gray > 0.6).astype(np.float32)
        num = float((attr * lung).sum())
        den = float(attr.sum())
        return (num / den) if den else 0.0
    except Exception:
        return 0.0

def generate_trust_score(prob, gradcam_pct, vitcx_pct, lime_pct, combined_pct):
    all_percents = [gradcam_pct, vitcx_pct, lime_pct, combined_pct]
    meaningful = [p for p in all_percents if p > 0.01]
    avg_focus = np.mean(meaningful) if meaningful else 0.0
    confidence_factor = 1 + (abs(prob - 0.5) * 2)
    trust_score = avg_focus * 100 * confidence_factor
    return float(max(0.0, min(100.0, trust_score)))

def generate_clinical_explanation(prob, gradcam_pct, vitcx_pct, lime_pct, combined_pct, trust_score, best_thr=None):
    diagnosis = "Pneumonia likely present" if prob >= (best_thr if best_thr is not None else 0.5) else "No clear signs of pneumonia"
    confidence_level = abs(prob - 0.5)
    if confidence_level > 0.4:
        confidence = "high confidence"
    elif confidence_level > 0.2:
        confidence = "moderate confidence"
    else:
        confidence = "low confidence"

    xai_summary = []
    if gradcam_pct > 0.05:
        xai_summary.append(f"Grad-CAM: {gradcam_pct:.1%} focus in lung regions")
    if vitcx_pct > 0.05:
        xai_summary.append(f"ViT-CX: {vitcx_pct:.1%} attention within lung regions")
    if lime_pct > 0.05:
        xai_summary.append(f"LIME: {lime_pct:.1%} important features in lungs")
    if combined_pct > 0.05:
        xai_summary.append(f"Combined ViT-CX+LIME: {combined_pct:.1%} focus in lung regions")
    if not xai_summary:
        xai_summary.append("No significant focus detected in lung regions across explainers.")

    avg_focus = np.mean([p for p in [gradcam_pct, vitcx_pct, lime_pct, combined_pct] if p > 0.01]) if any([gradcam_pct, vitcx_pct, lime_pct, combined_pct]) else 0.0
    if avg_focus > 0.6:
        focus_statement = "The model demonstrates strong focus on clinically relevant lung areas."
    elif avg_focus > 0.3:
        focus_statement = "The model shows moderate focus on lung regions."
    else:
        focus_statement = "The model's focus on lung regions appears limited."

    formatted_summary = "\n- " + "\n- ".join(xai_summary)
    thr_note = f"(operating threshold: {best_thr:.2f})" if best_thr is not None else ""
    return f"""**Clinical Assessment {thr_note}:**
- Diagnosis: {diagnosis} ({confidence}, probability: {prob:.1%})

**XAI Explanation Focus:**
{formatted_summary}

**Interpretation:**
- {focus_statement}

**Trust Score:** {trust_score:.2f}/100  
(This score reflects the overall consistency and clinical relevance of the model's explanations.)
"""

def _blend_two_overlays(grad_img, lime_img, size_hw):
    import cv2, numpy as np
    th, tw = size_hw
    if grad_img is None and lime_img is None:
        return None
    if grad_img is None:
        return lime_img
    if lime_img is None:
        return grad_img
    g = cv2.resize(np.array(grad_img), (tw, th), interpolation=cv2.INTER_LINEAR)
    l = cv2.resize(np.array(lime_img), (tw, th), interpolation=cv2.INTER_LINEAR)
    blended = cv2.addWeighted(g.astype(np.uint8), 0.6, l.astype(np.uint8), 0.8, 0)
    from PIL import Image
    return Image.fromarray(blended)


# =========================
# Inference entry point
# =========================
def run_inference(image_path, device=None, segmentation_model=None, vit_model=None, processor=None, target_size=None):
    device = device or _device()
    if vit_model is None:
        vit_model, meta = load_pretrained_model(device)
    else:
        meta = {"image_mean": None, "image_std": None, "best_thr": 0.50}
    if processor is None:
        processor = _build_processor(image_mean=meta.get("image_mean"), image_std=meta.get("image_std"))
    if segmentation_model is None:
        segmentation_model = load_pretrained_unet(device)

    vit_tensor, img_np_224 = preprocess_for_vit(image_path, processor, device)

    # NEW: get soft & binary masks @224
    soft_mask_224, bin_mask_224 = get_or_generate_lung_mask(image_path, segmentation_model, device)

    th, tw = (target_size or img_np_224.shape[:2])
    lung_mask_img, lung_overlay_img = make_lung_visuals(img_np_224, bin_mask_224, target_size=(th, tw))

    with torch.no_grad():
        outputs = vit_model(vit_tensor.to(device))
        prob = torch.sigmoid(outputs.logits).flatten()[0].item()

    # Explainers
    gradcam_map, gradcam_img   = grad_cam_plus_plus_vit(vit_model, vit_tensor, bin_mask_224, device=device, target_size=(th, tw))
    vitcx_map, vitcx_img       = vit_cx(vit_model, vit_tensor, bin_mask_224, device=device, target_size=(th, tw))
    lime_map, lime_img         = lime_explanation(img_np_224, vit_model, processor, bin_mask_224, device=device, target_size=(th, tw))

    # Combine without re-running
    combined_img = _blend_two_overlays(gradcam_img, lime_img, (th, tw))
    combined_map = None
    if gradcam_map is not None or lime_map is not None:
        g = cv2.resize(gradcam_map.astype(np.float32), (tw, th), interpolation=cv2.INTER_NEAREST) if gradcam_map is not None else 0.0
        l = cv2.resize(lime_map.astype(np.float32), (tw, th), interpolation=cv2.INTER_NEAREST) if lime_map is not None else 0.0
        combined_map = np.maximum(g, l).astype(np.float32)

    # Metrics (use binary mask for focus)
    gradcam_pct  = compute_lung_focus(gradcam_map,  bin_mask_224) if gradcam_map  is not None else 0.0
    vitcx_pct    = compute_lung_focus(vitcx_map,    bin_mask_224) if vitcx_map    is not None else 0.0
    lime_pct     = compute_lung_focus(lime_map,     bin_mask_224) if lime_map     is not None else 0.0
    combined_pct = compute_lung_focus(combined_map, bin_mask_224) if combined_map is not None else 0.0

    trust_score = generate_trust_score(prob, gradcam_pct, vitcx_pct, lime_pct, combined_pct)
    best_thr = meta.get("best_thr", 0.50)
    explanation = generate_clinical_explanation(prob, gradcam_pct, vitcx_pct, lime_pct, combined_pct, trust_score, best_thr=best_thr)

    # For UI: also return the soft mask visualization (grayscale heat)
    soft_vis = (np.clip(soft_mask_224, 0, 1) * 255).astype(np.uint8)
    soft_vis = Image.fromarray(soft_vis)

    return {
        "probability": prob,
        "best_thr": best_thr,
        "lung_mask_binary": lung_mask_img,   # NEW explicit binary mask image
        "lung_mask_soft": soft_vis,          # NEW soft mask image
        "lung_overlay": lung_overlay_img,
        "gradcam": gradcam_img,
        "vitcx": vitcx_img,
        "lime": lime_img,
        "combined_vitcx_lime": combined_img,
        "explanation": explanation,
        "percentages": {"gradcam": gradcam_pct, "vitcx": vitcx_pct, "lime": lime_pct, "combined": combined_pct},
        "trust_score": trust_score,
        "lung_coverage": float((bin_mask_224 > 0.5).mean()),
    }

# =========================
# PDF & feedback
# =========================
def _ensure_pil(img):
    if img is None:
        return None
    if isinstance(img, Image.Image):
        return img
    try:
        return Image.fromarray(np.array(img))
    except Exception:
        return None

def export_pdf_report(filename, results, feedback_text=None):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    def sanitize_text(text):
        if text is None:
            return ""
        return text.encode('latin-1', errors='replace').decode('latin-1')

    pdf.cell(0, 10, sanitize_text("Pneumonia Detection Report"), 0, 1, 'C')
    pdf.ln(10)

    pdf.set_font('', 'B')
    pdf.cell(0, 10, sanitize_text("Study Information"), 0, 1)
    pdf.set_font('')
    pdf.cell(0, 10, sanitize_text(f"Image: {filename}"), 0, 1)
    try:
        mtime = os.path.getmtime(filename)
        pdf.cell(0, 10, sanitize_text(f"Date (mtime): {mtime:.0f}"), 0, 1)
    except Exception:
        pass
    pdf.ln(5)

    pdf.set_font('', 'B')
    pdf.cell(0, 10, sanitize_text("Clinical Findings"), 0, 1)
    pdf.set_font('')
    pdf.multi_cell(0, 10, sanitize_text(results.get('explanation', '')))
    pdf.ln(5)

    to_embed = [
        ("LUNG MASK", _ensure_pil(results.get("lung_mask"))),
        ("LUNG OVERLAY", _ensure_pil(results.get("lung_overlay"))),
        ("GRAD-CAM", _ensure_pil(results.get("gradcam"))),
        ("VIT-CX", _ensure_pil(results.get("vitcx"))),
        ("LIME", _ensure_pil(results.get("lime"))),
        ("COMBINED VIT-CX + LIME", _ensure_pil(results.get("combined_vitcx_lime"))),
    ]

    img_paths = []
    for title, img in to_embed:
        if img is None:
            continue
        img_path = os.path.join(REPORT_DIR, f"temp_{title.replace(' ', '_').lower()}.png")
        try:
            img.save(img_path)
            img_paths.append(img_path)
            pdf.set_font('', 'B')
            pdf.cell(0, 10, sanitize_text(title), 0, 1)
            pdf.set_font('')
            pdf.image(img_path, w=180)
            pdf.ln(5)
        except Exception as e:
            print(f"[PDF] Skipping {title}: {e}")

    if feedback_text:
        pdf.set_font('', 'B')
        pdf.cell(0, 10, sanitize_text("Physician Feedback"), 0, 1)
        pdf.set_font('')
        pdf.multi_cell(0, 10, sanitize_text(feedback_text))

    report_path = os.path.join(REPORT_DIR, f"report_{os.path.splitext(os.path.basename(filename))[0]}.pdf")
    pdf.output(report_path)

    for path in img_paths:
        try:
            os.remove(path)
        except OSError:
            pass

    return report_path

def handle_feedback(feedback_data):
    with open(FEEDBACK_LOG, 'a') as f:
        json.dump(feedback_data, f)
        f.write('\n')

# =========================
# Visualization helper
# =========================
def plot_gradcam_comparison(xray_img: np.ndarray, gradcam_map: np.ndarray, lung_mask: np.ndarray):
    """
    Compare raw Grad-CAM vs lung-masked Grad-CAM side by side.
    """
    gradcam_norm = (gradcam_map - gradcam_map.min()) / (gradcam_map.max() - gradcam_map.min() + 1e-8)

    h, w = gradcam_norm.shape
    if xray_img.shape[:2] != (h, w):
        xray_img = cv2.resize(xray_img, (w, h), interpolation=cv2.INTER_LINEAR)
    if xray_img.ndim == 2:
        xray_img = cv2.cvtColor(xray_img, cv2.COLOR_GRAY2RGB)

    heatmap = cv2.applyColorMap(np.uint8(255 * gradcam_norm), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    raw_overlay = cv2.addWeighted(xray_img.astype(np.uint8), 0.6, heatmap.astype(np.uint8), 0.4, 0)

    lung_mask_resized = cv2.resize(lung_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    masked_cam = gradcam_norm * lung_mask_resized
    masked_heatmap = cv2.applyColorMap(np.uint8(255 * masked_cam), cv2.COLORMAP_JET)
    masked_heatmap = cv2.cvtColor(masked_heatmap, cv2.COLOR_BGR2RGB)
    masked_overlay = cv2.addWeighted(xray_img.astype(np.uint8), 0.6, masked_heatmap.astype(np.uint8), 0.4, 0)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1); plt.imshow(raw_overlay);    plt.title("Raw Grad-CAM");        plt.axis("off")
    plt.subplot(1, 2, 2); plt.imshow(masked_overlay); plt.title("Lung-Masked Grad-CAM"); plt.axis("off")
    plt.show()
