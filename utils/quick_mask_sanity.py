# quick_mask_sanity.py
import os, numpy as np
from glob import glob

image_root = "/Users/basava/Desktop/chest_xray/chest_xray/chest_xray/train"
mask_root  = "/Users/basava/Desktop/chest_xray/chest_xray/chest_xray/lung_masks"

img_paths = [p for ext in ("*.png","*.jpg","*.jpeg")
             for p in glob(os.path.join(image_root, "**", ext), recursive=True)]

missing, badshape, notfloat = 0, 0, 0
for p in img_paths[:5000]:  # cap just in case
    rel = os.path.relpath(p, image_root)
    base, _ = os.path.splitext(rel)
    mpath = os.path.join(mask_root, base + ".npy")
    if not os.path.exists(mpath):
        missing += 1
        continue
    m = np.load(mpath)
    if m.ndim == 3: m = m.squeeze()
    if m.shape != (224, 224):
        badshape += 1
    if m.dtype != np.float32 or np.nanmin(m) < 0 or np.nanmax(m) > 1.0:
        notfloat += 1

print(f"Missing masks: {missing}")
print(f"Bad shape (!=224x224): {badshape}")
print(f"Bad dtype/range (not float32 in [0,1]): {notfloat}")
