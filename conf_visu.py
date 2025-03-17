#!/usr/bin/env python3
import os
from pathlib import Path
import pickle
import random
import re

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from torchvision.transforms import (
    Resize,
    InterpolationMode,
    ToTensor,
)
from tqdm import tqdm
from model import get_model
import matplotlib.pyplot as plt


#########################
# Helper Functions
#########################
def natural_keys(text):
    """Split text into a list of strings and integers for natural sorting."""
    return [int(c) if c.isdigit() else c for c in re.split(r"(\d+)", str(text))]


def compute_iou(true_mask, pred_mask, num_classes=3):
    """
    Computes the mean Intersection-over-Union (IoU) between true and predicted masks.
    Both masks are numpy arrays of shape (H, W). Pixels with value -1 are ignored.
    """
    ious = []
    for cls in range(num_classes):
        true_cls = true_mask == cls
        pred_cls = pred_mask == cls
        intersection = np.logical_and(true_cls, pred_cls).sum()
        union = np.logical_or(true_cls, pred_cls).sum()
        if union == 0:
            ious.append(1.0)
        else:
            ious.append(intersection / union)
    return np.mean(ious)


def visualize_three(orig, true_mask, pred_overlay, iou):
    """
    Displays three images in one row:
      Left: Original image.
      Center: True mask overlaid on the original image.
      Right: Confidence-modulated prediction overlay on the original image.
    The right panel title shows the IoU value.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(orig, cmap="gray")
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(orig, cmap="gray")
    axes[1].imshow(true_mask, cmap="jet", alpha=0.5)
    axes[1].set_title("True Mask Overlay")
    axes[1].axis("off")

    axes[2].imshow(orig, cmap="gray")
    axes[2].imshow(pred_overlay)
    axes[2].set_title(f"Prediction Overlay (IoU = {iou:.2f})")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()


#########################
# Dedicated Test Dataset
#########################
class CementTestDataset(Dataset):
    """
    Test dataset for prediction. Loads raw images from .npy files and returns:
       - image: processed to 160x160 for model input,
       - true_mask: ground-truth mask (resized to 160x160 if needed),
       - key: file stem (used to retrieve the raw image for dimension-checking).
    """

    def __init__(
        self,
        images_dir,
        labels_csv=None,
        preload=True,
        cache_file="test_image_paths.pkl",
    ):
        self.images_dir = Path(images_dir)
        # Scan for .npy image files.
        self.image_paths = sorted(
            list(self.images_dir.glob("*.npy")), key=lambda p: natural_keys(p.stem)
        )
        if not self.image_paths:
            raise ValueError(f"No .npy files found in {self.images_dir}")
        self.preload = preload
        self.cache_file = Path(cache_file)

        # Optionally load labels (if provided); for test, labels may be available.
        self.labels_df = pd.read_csv(labels_csv, index_col=0) if labels_csv else None

        # Cache file paths if desired.
        if self.cache_file.exists():
            with open(self.cache_file, "rb") as f:
                self.image_paths = pickle.load(f)
        else:
            with open(self.cache_file, "wb") as f:
                pickle.dump(self.image_paths, f)

        # Preload images into memory if desired.
        self.data_cache = {}
        if self.preload:
            for p in self.image_paths:
                self.data_cache[p.stem] = np.load(p).astype(np.float32)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        key = image_path.stem
        # Load image from cache or disk.
        image = (
            self.data_cache[key]
            if self.preload
            else np.load(image_path, mmap_mode="r").astype(np.float32)
        )
        # For labels, if available, get the corresponding row from the CSV.
        if self.labels_df is not None:
            if key not in self.labels_df.index:
                raise KeyError(f"Key {key} not found in labels CSV.")
            mask_row = self.labels_df.loc[key].values
            valid_values = np.array([v for v in mask_row if v != -1], dtype=np.int64)
            width = int(len(valid_values) / 160)
            mask = valid_values.reshape(160, width)
        else:
            mask = np.zeros((160, 160), dtype=np.int64)

        # For model input, we want to resize images (and masks) to 160x160.
        # Convert image (and mask) from numpy array to PIL, then resize.
        image_pil = Image.fromarray(image)
        mask_pil = Image.fromarray(mask.astype(np.uint8))

        # If the raw image width is not 160 (i.e. 272), we resize to 160x160 for model input.
        if image.shape[1] != 160:
            image_pil = image_pil.resize((160, 160), Image.BILINEAR)
            mask_pil = mask_pil.resize((160, 160), Image.NEAREST)
        else:
            image_pil = image_pil.resize((160, 160), Image.BILINEAR)
            mask_pil = mask_pil.resize((160, 160), Image.NEAREST)

        image_tensor = ToTensor()(image_pil)
        mask_tensor = torch.from_numpy(np.array(mask_pil)).long()

        # Normalize image.
        image_min, image_max = image_tensor.min(), image_tensor.max()
        if image_max - image_min > 0:
            image_tensor = (image_tensor - image_min) / (image_max - image_min)
        else:
            image_tensor = image_tensor - image_min

        return image_tensor, mask_tensor, key


#########################
# DataLoader Function for Test
#########################
def get_test_dataloader(images_dir, labels_csv=None, batch_size=1, num_workers=4):
    dataset = CementTestDataset(images_dir, labels_csv=labels_csv)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return dataloader


#########################
# Main: Evaluate and Visualize Worst Samples
#########################
def main():
    # -----------------------------
    # Configuration – adjust paths and parameters.
    # -----------------------------
    images_dir = Path(
        "./DATA/X_train/images"
    )  # Folder with raw training images (used as test here)
    labels_csv = Path("./DATA/Y_train.csv")  # Ground-truth labels CSV
    checkpoint_path = Path("checkpoints/model_60_64_0.0001_att.pth")
    num_classes = 3  # e.g., background=0, plus two classes
    iou_threshold = 0.5  # Only display samples with IoU below this threshold.

    test_loader = get_test_dataloader(
        images_dir, labels_csv=labels_csv, batch_size=1, num_workers=4
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(num_classes)
    if checkpoint_path.exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    worst_samples = []  # List to store samples with IoU below threshold.

    # Define a threshold for confidence (e.g., 0.7 means 70% confidence).
    confidence_threshold = 0.5

    # Process each test sample.
    with torch.no_grad():
        for images, true_masks, keys in tqdm(
            test_loader, desc="Evaluating samples", unit="sample"
        ):
            images = images.to(device)
            outputs = model(images)  # Model outputs: [1, num_classes, 160, 160]
            probs = F.softmax(outputs, dim=1)
            confidence, pred_class = torch.max(probs, dim=1)  # Both: [1, 160, 160]
            pred_mask = pred_class.cpu().numpy()[0]  # (160, 160)
            conf_map = confidence.cpu().numpy()[0]  # (160, 160)
            key = keys[0]

            # Load the original raw image to determine its true width.
            raw_img = np.load(images_dir / f"{key}.npy").astype(np.float32)
            if raw_img.ndim == 3 and raw_img.shape[-1] == 1:
                raw_img = raw_img.squeeze(-1)
            orig_width = raw_img.shape[1]  # Either 160 or 272.

            # Adjust prediction and confidence to match original dimensions.
            if orig_width == 272:
                # Upsample prediction from 160x160 to 160x272 using nearest neighbor.
                pred_pil = Image.fromarray(pred_mask.astype(np.uint8))
                resizer = Resize((160, 272), interpolation=InterpolationMode.NEAREST)
                pred_adj = np.array(resizer(pred_pil))

                conf_pil = Image.fromarray((conf_map * 255).astype(np.uint8))
                conf_resizer = Resize(
                    (160, 272), interpolation=InterpolationMode.NEAREST
                )
                conf_adj = np.array(conf_resizer(conf_pil)).astype(float) / 255.0
            elif orig_width == 160:
                # Pad prediction with -1 on the right to get 160x272.
                padded_pred = -1 * np.ones((160, 272), dtype=np.int32)
                padded_pred[:, :160] = pred_mask
                pred_adj = padded_pred

                padded_conf = np.zeros((160, 272), dtype=float)
                padded_conf[:, :160] = conf_map
                conf_adj = padded_conf
            else:
                print(f"Unexpected width for {key}: {orig_width}")
                continue

            # Adjust true mask similarly.
            true_mask_np = true_masks.cpu().numpy()[0]
            if true_mask_np.shape[1] == 160 and orig_width == 272:
                true_pil = Image.fromarray(true_mask_np.astype(np.uint8))
                resizer = Resize((160, 272), interpolation=InterpolationMode.NEAREST)
                true_adj = np.array(resizer(true_pil))
            elif true_mask_np.shape[1] == 160 and orig_width == 160:
                padded_true = -1 * np.ones((160, 272), dtype=np.int32)
                padded_true[:, :160] = true_mask_np
                true_adj = padded_true
            else:
                true_adj = true_mask_np

            # Compute IoU.
            iou = compute_iou(true_adj, pred_adj, num_classes)

            if iou < iou_threshold:
                # Create a confidence-based overlay.
                cmap = plt.cm.get_cmap("jet")
                # Normalize the prediction for the colormap (assuming classes 0,1,2).
                normed = pred_adj.astype(float)
                overlay = cmap(normed / (num_classes - 1))  # This gives an RGBA image.

                # Now, modulate the overlay's alpha channel based on confidence.
                # For pixels where confidence < threshold, set alpha low (e.g., 0.3);
                # for pixels where confidence >= threshold, set alpha high (e.g., 0.8).
                alpha_mod = np.where(conf_adj < confidence_threshold, 0.3, 0.8)
                overlay[..., 3] = alpha_mod

                # Immediately display the sample:
                fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                axes[0].imshow(raw_img, cmap="gray")
                axes[0].set_title("Original Image")
                axes[0].axis("off")

                axes[1].imshow(raw_img, cmap="gray")
                axes[1].imshow(true_adj, cmap="jet", alpha=0.5)
                axes[1].set_title("True Mask Overlay")
                axes[1].axis("off")

                axes[2].imshow(raw_img, cmap="gray")
                axes[2].imshow(overlay)
                axes[2].set_title(f"Prediction Overlay (IoU = {iou:.2f})")
                axes[2].axis("off")
                plt.tight_layout()
                plt.show()

    print("Evaluation complete. Displayed samples with IoU below threshold.")


if __name__ == "__main__":
    main()
