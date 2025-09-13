import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from models.unet import UNet
import glob
from tqdm import tqdm

# ======================
# Dataset class
# ======================
class LungSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

        self.images = sorted(
            glob.glob(os.path.join(image_dir, "cxrimage_*.png")) +
            glob.glob(os.path.join(image_dir, "cxrimage_*.jpeg"))
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        base_name = os.path.basename(img_path).replace("cxrimage_", "").split(".")[0]

        mask_path_candidates = glob.glob(os.path.join(self.mask_dir, f"cxrmask_{base_name}.png")) + \
                               glob.glob(os.path.join(self.mask_dir, f"cxrmask_{base_name}.jpeg"))
        if not mask_path_candidates:
            raise FileNotFoundError(f"No mask found for image {img_path}")
        mask_path = mask_path_candidates[0]

        image = Image.open(img_path).convert("L")  # Grayscale
        mask = Image.open(mask_path).convert("L")  # Grayscale mask

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        mask = (mask > 0).float()
        return image, mask

# === Evaluation Metrics ===
def dice_score(pred, target):
    smooth = 1e-6
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    intersection = (pred_flat * target_flat).sum()
    return (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)

def pixel_accuracy(pred, target):
    pred = pred.view(-1)
    target = target.view(-1)
    return (pred == target).float().mean().item()

def precision_score(pred, target):
    pred = pred.view(-1)
    target = target.view(-1)
    tp = ((pred == 1) & (target == 1)).float().sum()
    fp = ((pred == 1) & (target == 0)).float().sum()
    return tp / (tp + fp + 1e-6)

def recall_score(pred, target):
    pred = pred.view(-1)
    target = target.view(-1)
    tp = ((pred == 1) & (target == 1)).float().sum()
    fn = ((pred == 0) & (target == 1)).float().sum()
    return tp / (tp + fn + 1e-6)

def iou_score(pred, target):
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + 1e-6) / (union + 1e-6)

# === Full Evaluation ===
@torch.no_grad()
def evaluate(model_path, image_dir, mask_dir):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    dataset = LungSegmentationDataset(image_dir, mask_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    device = torch.device("mps" if torch.backends.mps.is_available() else
                          "cuda" if torch.cuda.is_available() else "cpu")
    print(f"⚙️ Evaluating on device: {device}")

    model = UNet(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    total_dice = total_acc = total_prec = total_rec = total_iou = 0.0
    for img, mask in tqdm(loader, desc="📊 Evaluating"):
        img, mask = img.to(device), mask.to(device)
        pred = (model(img) > 0.5).float()

        total_dice += dice_score(pred, mask)
        total_acc += pixel_accuracy(pred, mask)
        total_prec += precision_score(pred, mask)
        total_rec += recall_score(pred, mask)
        total_iou += iou_score(pred, mask)

    n = len(loader)
    print(f"\n✅ Evaluation Results:")
    print(f"Pixel Accuracy : {total_acc / n:.4f}")
    print(f"Dice Score     : {total_dice / n:.4f}")
    print(f"Precision      : {total_prec / n:.4f}")
    print(f"Recall         : {total_rec / n:.4f}")
    print(f"IoU Score      : {total_iou / n:.4f}")

# === Fast Evaluation for Streamlit ===
@torch.no_grad()
def evaluate_fast(model_path, image_dir, mask_dir, num_samples=10):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    dataset = LungSegmentationDataset(image_dir, mask_dir, transform=transform)
    if len(dataset) == 0:
        return {"accuracy": 0, "dice": 0, "precision": 0, "recall": 0, "iou": 0}

    subset = torch.utils.data.Subset(dataset, list(range(min(num_samples, len(dataset)))))
    loader = DataLoader(subset, batch_size=1, shuffle=False)

    device = torch.device("mps" if torch.backends.mps.is_available() else
                          "cuda" if torch.cuda.is_available() else "cpu")

    model = UNet(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    total_dice = total_acc = total_prec = total_rec = total_iou = 0.0
    for img, mask in loader:
        img, mask = img.to(device), mask.to(device)
        pred = (model(img) > 0.5).float()

        total_dice += dice_score(pred, mask)
        total_acc += pixel_accuracy(pred, mask)
        total_prec += precision_score(pred, mask)
        total_rec += recall_score(pred, mask)
        total_iou += iou_score(pred, mask)

    n = len(loader)
    return {
        "accuracy": float(total_acc / n),
        "dice": float(total_dice / n),
        "precision": float(total_prec / n),
        "recall": float(total_rec / n),
        "iou": float(total_iou / n)
    }

if __name__ == "__main__":
    model_path = "/Users/basava/Desktop/chest_xray/project/unet_lung_segmentation.pt"
    image_dir = "/Users/basava/Desktop/chest_xray/lung_segmentation/image"
    mask_dir = "/Users/basava/Desktop/chest_xray/lung_segmentation/mask"
    evaluate(model_path, image_dir, mask_dir)
