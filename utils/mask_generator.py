import os, sys
import torch
import numpy as np
from glob import glob
from torchvision import transforms
from PIL import Image

# ✅ Ensure correct path for UNet
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.unet import UNet  

def generate_lung_masks(input_dir, output_dir):
    """
    Generate lung segmentation masks using unet_lung_segmentation.pt
    Saves only:
      - Soft mask (.npy) for training with natural lung edges
    """

    # ✅ Locate model
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    model_path = os.path.join(project_root, "unet_lung_segmentation.pt")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Model checkpoint not found at: {model_path}")

    # ✅ Find images recursively
    image_paths = []
    for ext in ("*.jpg", "*.png", "*.jpeg"):
        image_paths.extend(glob(os.path.join(input_dir, "**", ext), recursive=True))

    if not image_paths:
        print(f"❌ No images found in {input_dir}")
        return

    os.makedirs(output_dir, exist_ok=True)

    # ✅ Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    saved_count = 0
    with torch.no_grad():
        for img_path in image_paths:
            try:
                # ✅ Use grayscale input (not RGB)
                image = Image.open(img_path).convert("L")
                x = transform(image).unsqueeze(0).to(device)

                # ✅ Get soft mask probabilities
                mask = torch.sigmoid(model(x)).squeeze().cpu().numpy()  # [256,256] in [0,1]

                # ✅ Resize back to original image size
                mask_resized = np.array(Image.fromarray(mask).resize(image.size, Image.BILINEAR))

                # ✅ Save only soft mask (.npy)
                rel_path = os.path.relpath(img_path, input_dir)
                base_name = os.path.splitext(rel_path)[0]
                soft_save_path = os.path.join(output_dir, base_name + "_soft.npy")

                os.makedirs(os.path.dirname(soft_save_path), exist_ok=True)
                np.save(soft_save_path, mask_resized.astype(np.float32))

                print(f"✅ Saved soft mask: {soft_save_path}")
                saved_count += 1
            except Exception as e:
                print(f"⚠️ Failed for {img_path}: {e}")

    print(f"🫁 Done: {saved_count} soft lung masks saved as .npy")
