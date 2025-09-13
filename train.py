import os
import sys
import math
import json
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
from torch import nn, optim
from torchvision import transforms
from transformers import ViTForImageClassification, ViTImageProcessor
from tqdm import tqdm
from PIL import Image
import numpy as np
import random

# --- Metrics ---
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    accuracy_score, f1_score, confusion_matrix
)

# 🔧 Ensure project root is on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils.data import PneumoniaXrayDataset
from utils.prepare_labels import prepare_binary_labels
from models.unet import UNet  # ✅ import works

# --- Repro ---
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
set_seed(42)

# === Paths ===
dataset_root = "/Users/basava/Desktop/chest_xray/chest_xray/chest_xray"
mask_root = "/Users/basava/Desktop/chest_xray/project"
image_dir = os.path.join(dataset_root, "train")
lung_mask_dir = os.path.join(dataset_root, "lung_masks")
test_image_dir = os.path.join(dataset_root, "test")
test_lung_mask_dir = os.path.join(dataset_root, "test_lung_masks")
unet_model_path = os.path.join(mask_root, "unet_lung_segmentation.pt")

# === Device ===
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"ℹ️ Using device: {device}")

# Used for AMP (only enable on CUDA to avoid warnings elsewhere)
use_cuda = (device.type == "cuda")

# === Generate SOFT lung masks using trained UNet ===
def generate_soft_masks(image_folder, mask_folder, model_path, resize=(224, 224), device=None):
    os.makedirs(mask_folder, exist_ok=True)
    if device is None:
        device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

    print(f"🫁 Loading UNet model from {model_path} on {device}...")
    model = UNet(in_channels=1, out_channels=1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()

    exts = (".png", ".jpg", ".jpeg")
    image_files = []
    for root, _, files in os.walk(image_folder):
        for f in files:
            if f.lower().endswith(exts):
                image_files.append(os.path.join(root, f))

    print(f"🫁 Generating SOFT lung masks for {len(image_files)} images in {image_folder}...")
    for img_path in tqdm(image_files, unit="img"):
        rel_path = os.path.relpath(img_path, image_folder)
        out_dir = os.path.join(mask_folder, os.path.dirname(rel_path))
        os.makedirs(out_dir, exist_ok=True)

        mask_path = os.path.join(out_dir, os.path.splitext(os.path.basename(img_path))[0] + ".npy")
        if os.path.exists(mask_path):
            continue

        img = Image.open(img_path).convert("L").resize(resize)
        img_tensor = torch.tensor(np.array(img) / 255.0, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            pred_mask = torch.sigmoid(model(img_tensor)).squeeze().cpu().numpy()

        np.save(mask_path, pred_mask.astype(np.float32))  # SOFT mask

    print(f"✅ Soft masks saved to {mask_folder}")

# === Step 1: Generate masks for train and test (once) ===
if not os.path.exists(lung_mask_dir) or not any(os.scandir(lung_mask_dir)):
    generate_soft_masks(image_dir, lung_mask_dir, unet_model_path, device=device)
else:
    print(f"ℹ️ Train masks already exist in {lung_mask_dir}")

if not os.path.exists(test_lung_mask_dir) or not any(os.scandir(test_lung_mask_dir)):
    generate_soft_masks(test_image_dir, test_lung_mask_dir, unet_model_path, device=device)
else:
    print(f"ℹ️ Test masks already exist in {test_lung_mask_dir}")

# === Step 2: Labels ===
label_dict = prepare_binary_labels(image_dir)  # {path->0/1}
print(f"Found {len(label_dict)} label entries from prepare_binary_labels()")


# === Step 3: Transforms (⚠️ use ViT mean/std) ===
processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
mean = processor.image_mean
std = processor.image_std
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

# === Step 4: Dataset ===
dataset = PneumoniaXrayDataset(image_dir, label_dict, lung_mask_dir, transform)
if len(dataset) == 0:
    raise ValueError("❌ No valid samples found. Check dataset paths.")

# Labels array for split/weights
labels = np.array([dataset[i][1] for i in range(len(dataset))], dtype=np.float32)  # 0/1

# --- Stratified train/val split (80/20) ---
rng = np.random.default_rng(42)
idx_pos = np.where(labels == 1)[0]
idx_neg = np.where(labels == 0)[0]
rng.shuffle(idx_pos); rng.shuffle(idx_neg)

val_frac = 0.20
n_pos_val = max(1, int(len(idx_pos) * val_frac))
n_neg_val = max(1, int(len(idx_neg) * val_frac))

val_idx = np.concatenate([idx_pos[:n_pos_val], idx_neg[:n_neg_val]])
train_idx = np.concatenate([idx_pos[n_pos_val:], idx_neg[n_neg_val:]])
rng.shuffle(train_idx); rng.shuffle(val_idx)

train_ds = Subset(dataset, train_idx)
val_ds   = Subset(dataset, val_idx)

# --- Weighted sampler on TRAIN only ---
train_labels = labels[train_idx]
pos_count = float(train_labels.sum()) + 1e-8
neg_count = float((train_labels == 0).sum()) + 1e-8
pos_weight = neg_count / pos_count  # for BCEWithLogitsLoss

class_counts = np.array([neg_count, pos_count])
class_weights = 1.0 / class_counts
sample_weights = np.where(train_labels == 1, class_weights[1], class_weights[0])

train_loader = DataLoader(
    train_ds,
    batch_size=16,
    sampler=WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True
    ),
    num_workers=4 if device.type != "mps" else 0,
    pin_memory=(device.type == "cuda"),
    drop_last=False,
)
val_loader = DataLoader(
    val_ds,
    batch_size=32,
    shuffle=False,
    num_workers=4 if device.type != "mps" else 0,
    pin_memory=(device.type == "cuda"),
)

# === Step 5: Model (single-logit head, eager attention to avoid warning) ===
id2label = {0: "PneumoniaProb"}
label2id = {"PneumoniaProb": 0}
model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224-in21k",
    num_labels=1,                          # single logit
    id2label=id2label,
    label2id=label2id,
    problem_type="single_label_classification",
    ignore_mismatched_sizes=True,          # replaces classifier safely
    attn_implementation="eager",
).to(device)

# === Step 6: Loss/Opt/Sched ===
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
optimizer = optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)

num_epochs = 5
steps_per_epoch = math.ceil(len(train_loader))
total_steps = steps_per_epoch * num_epochs
warmup = int(0.1 * total_steps)

def lr_lambda(step):
    if step < warmup:
        return step / max(1, warmup)
    # cosine decay
    progress = (step - warmup) / max(1, total_steps - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * progress))

scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

# ✅ New AMP API (gets rid of deprecation warnings)
scaler = torch.amp.GradScaler(enabled=use_cuda)

# === Eval ===
@torch.no_grad()
def evaluate(model, dataloader):
    model.eval()
    all_logits, all_targets = [], []
    total_loss, n = 0.0, 0
    for x, y in dataloader:
        x = x.to(device, non_blocking=True)
        y = y.to(device).float().view(-1)
        logits = model(x).logits.view(-1)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        n += x.size(0)
        all_logits.append(logits.detach().cpu())
        all_targets.append(y.detach().cpu())

    logits = torch.cat(all_logits).numpy()
    targets = torch.cat(all_targets).numpy()
    probs = 1 / (1 + np.exp(-logits))  # sigmoid

    try:
        auroc = roc_auc_score(targets, probs)
    except ValueError:
        auroc = float("nan")
    auprc = average_precision_score(targets, probs)

    # Threshold via Youden's J on a small grid
    thresholds = np.linspace(0.05, 0.95, 19)
    best_t, best_j = 0.5, -1
    for t in thresholds:
        preds = (probs >= t).astype(int)
        tp = ((preds == 1) & (targets == 1)).sum()
        tn = ((preds == 0) & (targets == 0)).sum()
        fp = ((preds == 1) & (targets == 0)).sum()
        fn = ((preds == 0) & (targets == 1)).sum()
        tpr = tp / max(1, (tp + fn))
        fpr = fp / max(1, (fp + tn))
        j = tpr - fpr
        if j > best_j:
            best_j, best_t = j, t

    preds = (probs >= best_t).astype(int)
    acc = accuracy_score(targets, preds)
    f1 = f1_score(targets, preds, zero_division=0)
    cm = confusion_matrix(targets, preds).tolist()

    val_loss = total_loss / max(1, n)
    return {"loss": val_loss, "auroc": float(auroc), "auprc": float(auprc),
            "acc": float(acc), "f1": float(f1), "thr": float(best_t), "cm": cm}

# === Train epochs (with validation + best saving) ===
best_auroc = -1.0
best_ckpt = "vit_pneumonia_best.pt"

def train_epochs(model, train_loader, val_loader, epochs):
    global best_auroc
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for x, y in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device).float().view(-1)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", enabled=use_cuda):
                logits = model(x).logits.view(-1)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += loss.item() * x.size(0)
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
            })

        train_loss = total_loss / len(train_loader.dataset)

        # Validate
        metrics = evaluate(model, val_loader)
        print(f"✅ Epoch {epoch}: "
              f"train_loss={train_loss:.4f} | "
              f"val_loss={metrics['loss']:.4f} | "
              f"AUROC={metrics['auroc']:.4f} | "
              f"AUPRC={metrics['auprc']:.4f} | "
              f"Acc={metrics['acc']:.4f} | F1={metrics['f1']:.4f} | thr={metrics['thr']:.2f}")
        print(f"Confusion Matrix (val): {metrics['cm']}")

        # Save best by AUROC
        if metrics["auroc"] == metrics["auroc"] and metrics["auroc"] > best_auroc:
            best_auroc = metrics["auroc"]
            torch.save({
                "model_state_dict": model.state_dict(),
                "image_mean": mean,
                "image_std": std,
                "pretrained_name": "google/vit-base-patch16-224-in21k",
                "num_labels": 1,
                "best_thr": metrics["thr"],
            }, best_ckpt)
            print(f"💾 Saved new best model to {best_ckpt} (AUROC {best_auroc:.4f})")

# === Optional: 1-epoch head-only warmup for stability ===
WARMUP_EPOCHS = 1  # set to 0 to disable

if WARMUP_EPOCHS > 0:
    for name, p in model.named_parameters():
        if not name.startswith("classifier."):
            p.requires_grad = False
    print(f"🧊 Warmup: training classifier head only for {WARMUP_EPOCHS} epoch(s)")
    train_epochs(model, train_loader, val_loader, WARMUP_EPOCHS)
    for p in model.parameters():
        p.requires_grad = True
    remaining = max(0, num_epochs - WARMUP_EPOCHS)
    if remaining > 0:
        print(f"🔥 Unfrozen backbone: training for {remaining} epoch(s)")
        train_epochs(model, train_loader, val_loader, remaining)
else:
    train_epochs(model, train_loader, val_loader, num_epochs)

# === Step 8: Save model + normalization metadata (so inference matches exactly) ===
save_path = "vit_pneumonia.pt"
torch.save(
    {
        "model_state_dict": model.state_dict(),
        "image_mean": mean,
        "image_std": std,
        "pretrained_name": "google/vit-base-patch16-224-in21k",
        "num_labels": 1,
    },
    save_path
)
print(f"✅ Trained ViT model saved as {save_path}")
