# utils/data.py
import os
import math
import random
from typing import Dict, List, Tuple, Optional

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

def _safe_load_mask(mask_path: str, target_hw: Tuple[int, int]) -> np.ndarray:
    """
    Loads a .npy mask in [H,W] float32, resizes to target_hw if needed.
    If missing/corrupt -> returns ones (no-op mask).
    """
    H, W = target_hw
    try:
        m = np.load(mask_path)
        if m.ndim == 3:
            m = m.squeeze()
        m = m.astype(np.float32)
        if m.shape != (H, W):
            # very light resize via PIL to keep things simple
            m_img = Image.fromarray((np.clip(m, 0, 1) * 255).astype(np.uint8))
            m_img = m_img.resize((W, H), Image.BILINEAR)
            m = np.array(m_img, dtype=np.float32) / 255.0
        m = np.clip(m, 0.0, 1.0)
        return m
    except Exception:
        return np.ones((H, W), dtype=np.float32)

def _mask_guided_random_crop(img_np: np.ndarray, mask: np.ndarray, crop_prob: float = 0.30) -> Tuple[np.ndarray, np.ndarray]:
    """
    Randomly crop around lung region with some padding, then resize back to original.
    img_np: [H,W,3], mask: [H,W]
    """
    if random.random() > crop_prob:
        return img_np, mask

    H, W, _ = img_np.shape
    ys, xs = np.where(mask > 0.25)
    if len(xs) < 10:  # not enough foreground, skip
        return img_np, mask

    y1, y2 = np.min(ys), np.max(ys)
    x1, x2 = np.min(xs), np.max(xs)
    # pad bbox a bit
    pad_y = int(0.10 * (y2 - y1 + 1))
    pad_x = int(0.10 * (x2 - x1 + 1))
    y1 = max(0, y1 - pad_y); y2 = min(H - 1, y2 + pad_y)
    x1 = max(0, x1 - pad_x); x2 = min(W - 1, x2 + pad_x)

    # jitter the bbox slightly
    j = lambda a, b: random.randint(a, b) if b >= a else a
    y1 = max(0, y1 - j(0, pad_y)); y2 = min(H - 1, y2 + j(0, pad_y))
    x1 = max(0, x1 - j(0, pad_x)); x2 = min(W - 1, x2 + j(0, pad_x))

    # avoid degenerate crop
    if (y2 - y1) < 16 or (x2 - x1) < 16:
        return img_np, mask

    crop_img = img_np[y1:y2+1, x1:x2+1, :]
    crop_mask = mask[y1:y2+1, x1:x2+1]

    # resize back
    to_H, to_W = H, W
    crop_img_pil = Image.fromarray(crop_img)
    crop_mask_pil = Image.fromarray((np.clip(crop_mask, 0, 1) * 255).astype(np.uint8))
    crop_img = np.array(crop_img_pil.resize((to_W, to_H), Image.BILINEAR))
    crop_mask = np.array(crop_mask_pil.resize((to_W, to_H), Image.BILINEAR)) / 255.0
    return crop_img, crop_mask.astype(np.float32)

def _lung_only_brightness_jitter(img_np: np.ndarray, mask: np.ndarray, alpha_range=(0.9, 1.1), prob: float = 0.5) -> np.ndarray:
    """
    Multiply pixel intensity only inside lungs by random alpha.
    img_np: [H,W,3] uint8
    """
    if random.random() > prob:
        return img_np
    alpha = random.uniform(*alpha_range)
    out = img_np.astype(np.float32)
    lung = (mask > 0.25).astype(np.float32)[..., None]
    out = out * (1 - lung) + (out * alpha) * lung
    return np.clip(out, 0, 255).astype(np.uint8)

def _random_cutout_outside_lungs(img_np: np.ndarray, mask: np.ndarray, prob: float = 0.3, num_holes: Tuple[int, int] = (1, 3)) -> np.ndarray:
    """
    Apply random rectangular occlusions OUTSIDE the lung region.
    """
    if random.random() > prob:
        return img_np
    H, W, _ = img_np.shape
    out = img_np.copy()
    min_h, max_h = num_holes
    k = random.randint(min_h, max_h)
    for _ in range(k):
        h = random.randint(H // 12, H // 6)
        w = random.randint(W // 12, W // 6)
        y = random.randint(0, max(0, H - h))
        x = random.randint(0, max(0, W - w))
        # If most of the rectangle overlaps lungs, skip
        region = mask[y:y+h, x:x+w]
        if region.size == 0:
            continue
        if (region > 0.25).mean() > 0.2:
            continue
        out[y:y+h, x:x+w, :] = 0
    return out

class PneumoniaXrayDataset(Dataset):
    """
    Loads CXR images and optional SOFT lung masks (.npy) to perform lung-aware augmentations.
    Accepts absolute or relative image paths in label_dict.
    """

    def __init__(
        self,
        image_root: str,
        label_dict: Dict[str, int],
        mask_root: Optional[str] = None,
        transform=None,
        use_soft_mask: bool = True,
        target_size: Tuple[int, int] = (224, 224),
        aug_lung_crop_prob: float = 0.30,
        aug_lung_brightness_prob: float = 0.50,
        aug_cutout_outside_prob: float = 0.30,
    ):
        self.image_root = os.path.abspath(image_root)
        self.mask_root = mask_root
        self.transform = transform
        self.use_soft_mask = use_soft_mask and (mask_root is not None)
        self.H, self.W = target_size
        self.aug_lung_crop_prob = aug_lung_crop_prob
        self.aug_lung_brightness_prob = aug_lung_brightness_prob
        self.aug_cutout_outside_prob = aug_cutout_outside_prob

        def resolve_path(p: str) -> Optional[str]:
            """Return a valid file path for p (absolute or relative to image_root), else None."""
            # 1) Absolute & normalized
            cand = os.path.normpath(os.path.expanduser(p))
            if os.path.isabs(cand) and os.path.isfile(cand):
                return cand
            # 2) Relative to image_root
            cand2 = os.path.normpath(os.path.join(self.image_root, p))
            if os.path.isfile(cand2):
                return cand2
            # 3) If p already starts with image_root but minor differences, normalize again
            if cand.startswith(self.image_root) and os.path.isfile(cand):
                return cand
            # 4) Last resort: search by basename under image_root
            base = os.path.basename(p)
            for r, _, files in os.walk(self.image_root):
                if base in files:
                    return os.path.join(r, base)
            return None

        resolved = []
        for p, y in label_dict.items():
            rp = resolve_path(p)
            if rp is not None:
                resolved.append((rp, int(y)))
        self.samples: List[Tuple[str, int]] = resolved

        # Helpful debug if empty
        if len(self.samples) == 0:
            print("⚠️ PneumoniaXrayDataset: no samples resolved.")
            print(f"  image_root = {self.image_root}")
            # show a couple of entries from label_dict for inspection
            try:
                import itertools
                preview = list(itertools.islice(label_dict.items(), 3))
                print("  label_dict sample:", preview)
            except Exception:
                pass

    def __len__(self):
        return len(self.samples)

    def _mask_path_for_image(self, img_path: str) -> Optional[str]:
        if not self.use_soft_mask:
            return None
        rel = os.path.relpath(img_path, self.image_root)
        rel_base, _ = os.path.splitext(rel)
        return os.path.join(self.mask_root, rel_base + ".npy")

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB").resize((self.W, self.H), Image.BILINEAR)
        img_np = np.array(img)

        if self.use_soft_mask:
            mask_path = self._mask_path_for_image(img_path)
            mask = _safe_load_mask(mask_path, (self.H, self.W))
        else:
            mask = np.ones((self.H, self.W), dtype=np.float32)

        img_np, mask = _mask_guided_random_crop(img_np, mask, crop_prob=self.aug_lung_crop_prob)
        img_np = _lung_only_brightness_jitter(img_np, mask, prob=self.aug_lung_brightness_prob)
        img_np = _random_cutout_outside_lungs(img_np, mask, prob=self.aug_cutout_outside_prob)

        img = Image.fromarray(img_np)
        if self.transform is not None:
            x = self.transform(img)
        else:
            x = torch.from_numpy(img_np.transpose(2,0,1)).float() / 255.0

        y = torch.tensor(label, dtype=torch.long)
        return x, y

