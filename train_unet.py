import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms import Compose, ToTensor, Resize
from PIL import Image
from tqdm import tqdm
from models.unet import UNet
import glob
import numpy as np

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

        image = Image.open(img_path).convert("L")
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        mask = (mask > 0).float()
        return image, mask


# ======================
# EarlyStopping Callback
# ======================
class EarlyStopping:
    def __init__(self, patience=5, delta=0.0, path="unet_lung_segmentation.pt"):
        """
        patience: how many epochs to wait before stopping when loss doesn't improve
        delta: minimum change to qualify as improvement
        path: file to save the best model
        """
        self.patience = patience
        self.delta = delta
        self.path = path
        self.counter = 0
        self.best_loss = np.Inf
        self.early_stop = False

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.save_checkpoint(model)
            self.counter = 0
        else:
            self.counter += 1
            print(f"⚠️ EarlyStopping counter: {self.counter} / {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.path)
        print(f"✅ Validation loss improved → model saved to {self.path}")


# ======================
# Paths & Data
# ======================
image_dir = "/Users/basava/Desktop/chest_xray/lung_segmentation/image"
mask_dir = "/Users/basava/Desktop/chest_xray/lung_segmentation/mask"

transform = Compose([
    Resize((256, 256)),
    ToTensor(),
])

dataset = LungSegmentationDataset(image_dir, mask_dir, transform=transform)
if len(dataset) == 0:
    raise ValueError(f"No images found in {image_dir}")

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

device = torch.device("mps" if torch.backends.mps.is_available() else
                      "cuda" if torch.cuda.is_available() else "cpu")

model = UNet(in_channels=1, out_channels=1).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=1e-3,
    steps_per_epoch=len(train_loader), epochs=20
)

# ======================
# Training with EarlyStopping
# ======================
num_epochs = 20
early_stopping = EarlyStopping(patience=5, delta=0.0001, path="unet_lung_segmentation.pt")

for epoch in range(num_epochs):
    # Training
    model.train()
    train_loss = 0.0
    train_bar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] Training", leave=False)
    for images, masks in train_bar:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        scheduler.step()
        train_loss += loss.item()
        train_bar.set_postfix(loss=loss.item())

    # Validation
    model.eval()
    val_loss = 0.0
    val_bar = tqdm(val_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] Validation", leave=False)
    with torch.no_grad():
        for images, masks in val_bar:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            val_loss += loss.item()
            val_bar.set_postfix(val_loss=loss.item())

    avg_train_loss = train_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)

    print(f"Epoch [{epoch+1}/{num_epochs}] "
          f"Train Loss: {avg_train_loss:.4f} | "
          f"Val Loss: {avg_val_loss:.4f}")

    # Call EarlyStopping
    early_stopping(avg_val_loss, model)

    if early_stopping.early_stop:
        print("⏹ Early stopping triggered — training stopped.")
        break

# ======================
# Load Best Model
# ======================
model.load_state_dict(torch.load("unet_lung_segmentation.pt"))
print("✅ Loaded best saved model weights")
