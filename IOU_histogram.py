#!/usr/bin/env python3
import re
from pathlib import Path
import pickle
import random

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms import Resize, InterpolationMode, ToTensor
from tqdm import tqdm
import matplotlib.pyplot as plt

from model import get_model  # Assumed to return a segmentation model
from data import (
    get_test_dataloader,
)  # Assumed to return (image, true_mask, key) for test set


#########################
# Helper Functions
#########################
def natural_keys(text):
    """Splits a string into a list of strings and integers for natural sorting."""
    return [int(c) if c.isdigit() else c for c in re.split(r"(\d+)", str(text))]


def compute_iou(true_mask, pred_mask, num_classes=3):
    """
    Computes mean Intersection-over-Union (IoU) between true and predicted masks.
    Both masks are numpy arrays of shape (H, W) and pixels with value -1 are ignored.
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
      - Left: Original raw image.
      - Center: Original image with true mask overlay.
      - Right: Original image with prediction overlay (alpha modulated by confidence).
    The right panel title includes the IoU.
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
# Collect Results Function
#########################
def collect_results(test_loader, model, device, images_dir, num_classes=3):
    """
    Iterates over the test DataLoader and produces a list of dictionaries,
    each containing:
       - "key": sample identifier,
       - "iou": IoU computed between the adjusted prediction and true mask,
       - "pred_mask": the adjusted predicted mask (shape: 160x272),
       - "conf_map": the adjusted confidence map (shape: 160x272),
       - "original": the raw original image loaded from disk,
       - "true_mask": the adjusted true mask.

    Adjustments:
      - If the raw image width is 272, upsample the 160x160 prediction (and confidence)
        to 160x272 using nearest neighbor interpolation.
      - If the raw image width is 160, pad the 160x160 prediction (and confidence) on the right.
    """
    results = []
    images_dir = Path(images_dir)
    model.eval()
    with torch.no_grad():
        for images, true_masks, keys in tqdm(
            test_loader, desc="Collecting results", unit="sample"
        ):
            images = images.to(device)
            outputs = model(images)  # Expected shape: [1, num_classes, 160, 160]
            probs = F.softmax(outputs, dim=1)
            confidence, pred_class = torch.max(probs, dim=1)  # [1, 160, 160]
            pred_mask = pred_class.cpu().numpy()[0]  # shape: (160,160)
            conf_map = confidence.cpu().numpy()[0]  # shape: (160,160)
            key = keys[0]

            # Load raw image from disk.
            raw_img_path = images_dir / f"{key}.npy"
            raw_img = np.load(raw_img_path).astype(np.float32)
            if raw_img.ndim == 3 and raw_img.shape[-1] == 1:
                raw_img = raw_img.squeeze(-1)
            orig_width = raw_img.shape[1]  # 160 or 272

            # Adjust predicted mask and confidence.
            if orig_width == 272:
                pred_pil = Image.fromarray(pred_mask.astype(np.uint8))
                resizer = Resize((160, 272), interpolation=InterpolationMode.NEAREST)
                pred_mask_adj = np.array(resizer(pred_pil))

                conf_pil = Image.fromarray((conf_map * 255).astype(np.uint8))
                conf_resizer = Resize(
                    (160, 272), interpolation=InterpolationMode.NEAREST
                )
                conf_adj = np.array(conf_resizer(conf_pil)).astype(float) / 255.0
            elif orig_width == 160:
                padded_pred = -1 * np.ones((160, 272), dtype=np.int32)
                padded_pred[:, :160] = pred_mask
                pred_mask_adj = padded_pred

                padded_conf = np.zeros((160, 272), dtype=float)
                padded_conf[:, :160] = conf_map
                conf_adj = padded_conf
            else:
                print(f"Unexpected width for {key}: {orig_width}")
                continue

            # Adjust the true mask similarly.
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
            iou = compute_iou(true_adj, pred_mask_adj, num_classes)

            results.append(
                {
                    "key": key,
                    "iou": iou,
                    "pred_mask": pred_mask_adj,
                    "conf_map": conf_adj,
                    "original": raw_img,
                    "true_mask": true_adj,
                }
            )
    return results


#########################
# Compute Confidence by IoU Decile
#########################
def compute_confidence_by_iou_decile(results, num_classes=3, decile_bins=10):
    """
    Computes, for each class, the mean pixel confidence aggregated over IoU deciles.
    Returns the decile thresholds and a dictionary mapping each class to a list of mean confidence values for each decile.
    Also plots a bar chart.
    """
    iou_values = np.array([res["iou"] for res in results])
    thresholds = np.percentile(iou_values, np.linspace(0, 100, decile_bins + 1))

    conf_by_bin = {
        cls: {i: [] for i in range(decile_bins)} for cls in range(num_classes)
    }
    for res in results:
        iou = res["iou"]
        pred_mask = res["pred_mask"]
        conf_map = res["conf_map"]
        bin_idx = None
        for i in range(decile_bins):
            if i == decile_bins - 1:
                if iou >= thresholds[i]:
                    bin_idx = i
            else:
                if thresholds[i] <= iou < thresholds[i + 1]:
                    bin_idx = i
                    break
        if bin_idx is None:
            continue
        for cls in range(num_classes):
            cls_mask = pred_mask == cls
            cls_conf = conf_map[cls_mask]
            if cls_conf.size > 0:
                conf_by_bin[cls][bin_idx].extend(cls_conf.tolist())
    mean_conf = {cls: [] for cls in range(num_classes)}
    for cls in range(num_classes):
        for i in range(decile_bins):
            if conf_by_bin[cls][i]:
                mean_conf[cls].append(np.mean(conf_by_bin[cls][i]))
            else:
                mean_conf[cls].append(0.0)

    bins = np.arange(decile_bins)
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 6))
    for cls in range(num_classes):
        ax.bar(bins + cls * width, mean_conf[cls], width, label=f"Class {cls}")
    ax.set_xlabel("IoU Decile Bin (0=lowest IoU)")
    ax.set_ylabel("Mean Confidence")
    ax.set_title("Mean Pixel Confidence per Class by IoU Decile")
    ax.set_xticks(bins + width)
    ax.set_xticklabels([f"Bin {i}" for i in range(decile_bins)])
    ax.legend()
    plt.tight_layout()
    plt.show()

    return thresholds, mean_conf


#########################
# Reliability Diagram Functions
#########################
def compute_reliability_data_from_results(results, num_bins=10):
    """
    From a list of results (each a dictionary with keys:
        "conf_map": per-pixel confidence (shape: 160x272),
        "true_mask": adjusted true mask (shape: 160x272),
        "pred_mask": adjusted predicted mask (shape: 160x272)),
    this function computes overall pixel confidence and correctness, bins them,
    and returns:
       - bin_centers: centers of the confidence bins,
       - mean_conf: average confidence per bin,
       - empirical_accuracy: fraction of pixels predicted correctly per bin.
    Pixels with ignore label (-1) in the true mask are excluded.
    """
    all_confidences = []
    all_correct = []

    for res in results:
        conf = res["conf_map"].flatten()
        true_mask = res["true_mask"].flatten()
        pred_mask = res["pred_mask"].flatten()
        # Exclude ignore pixels.
        valid = true_mask != -1
        if np.sum(valid) == 0:
            continue
        all_confidences.extend(conf[valid])
        # 1 if prediction is correct, 0 otherwise.
        all_correct.extend((true_mask[valid] == pred_mask[valid]).astype(np.float32))

    all_confidences = np.array(all_confidences)
    all_correct = np.array(all_correct)

    # Create bins between 0 and 1.
    bins = np.linspace(0, 1, num_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    mean_conf = []
    empirical_accuracy = []
    for i in range(num_bins):
        # Find pixels with confidence in the bin.
        mask = (all_confidences >= bins[i]) & (all_confidences < bins[i + 1])
        if np.sum(mask) > 0:
            mean_conf.append(np.mean(all_confidences[mask]))
            empirical_accuracy.append(np.mean(all_correct[mask]))
        else:
            mean_conf.append(0)
            empirical_accuracy.append(0)
    return bin_centers, mean_conf, empirical_accuracy


def plot_reliability_diagram(bin_centers, mean_conf, empirical_accuracy):
    """
    Plots a reliability diagram:
       - x-axis: Confidence bin centers.
       - y-axis: Empirical accuracy in each bin.
    Also plots the line y=x, which represents perfect calibration.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(bin_centers, mean_conf, marker="o", label="Mean Confidence")
    plt.plot(bin_centers, empirical_accuracy, marker="x", label="Empirical Accuracy")
    plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title("Reliability Diagram")
    plt.legend()
    plt.grid(True)
    plt.show()


def compute_reliability_data_by_class(results, target_class, num_bins=10):
    """
    Computes reliability data for a specific target class from a list of results.
    Each result dictionary must contain "conf_map", "pred_mask", and "true_mask",
    where all arrays have shape (H, W).

    Only pixels where the predicted class equals target_class are considered.

    Returns:
       - bin_centers: centers of the confidence bins,
       - mean_conf: average confidence per bin,
       - empirical_accuracy: fraction of pixels predicted correctly (true == target_class)
         per bin.
    """
    confidences = []
    correct = []
    for res in results:
        conf_map = res["conf_map"].flatten()  # Confidence per pixel.
        pred_mask = res["pred_mask"].flatten()  # Predicted class per pixel.
        true_mask = res["true_mask"].flatten()  # Ground truth.

        # Only consider pixels where the predicted class is the target_class.
        indices = pred_mask == target_class
        if np.sum(indices) == 0:
            continue
        confidences.extend(conf_map[indices])
        # Pixel is correct if true_mask equals target_class.
        correct.extend((true_mask[indices] == target_class).astype(np.float32))

    confidences = np.array(confidences)
    correct = np.array(correct)

    bins = np.linspace(0, 1, num_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    mean_conf = []
    empirical_accuracy = []
    for i in range(num_bins):
        mask = (confidences >= bins[i]) & (confidences < bins[i + 1])
        if np.sum(mask) > 0:
            mean_conf.append(np.mean(confidences[mask]))
            empirical_accuracy.append(np.mean(correct[mask]))
        else:
            mean_conf.append(0)
            empirical_accuracy.append(0)

    return bin_centers, mean_conf, empirical_accuracy


def plot_reliability_diagram_by_class(
    bin_centers, mean_conf, empirical_accuracy, target_class
):
    """
    Plots a reliability diagram for a given target class.
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 6))
    plt.plot(bin_centers, mean_conf, marker="o", label="Mean Confidence")
    plt.plot(bin_centers, empirical_accuracy, marker="x", label="Empirical Accuracy")
    plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title(f"Reliability Diagram for Class {target_class}")
    plt.legend()
    plt.grid(True)
    plt.show()


def compute_pixel_accuracy_map(results, valid_width=160):
    """
    Given a list of result dictionaries (each with "true_mask" and "pred_mask" arrays of shape (160,272)),
    computes a pixel-wise accuracy map over the valid region (first `valid_width` columns).
    Pixels with ignore value (-1) in the true mask are not counted.

    Returns:
      accuracy_map: 2D numpy array of shape (160, valid_width) with values in [0,1].
    """
    H = 160
    W = valid_width
    correct_count = np.zeros((H, W), dtype=np.float64)
    total_count = np.zeros((H, W), dtype=np.float64)

    for res in results:
        # Only consider the valid region: first 160 columns.
        true_mask = res["true_mask"][:, :W]
        pred_mask = res["pred_mask"][:, :W]

        # Valid pixels: true_mask != -1.
        valid = true_mask != -1
        # Correct predictions where valid.
        correct = (true_mask == pred_mask) & valid

        correct_count += correct.astype(np.float64)
        total_count += valid.astype(np.float64)

    # Avoid division by zero.
    with np.errstate(divide="ignore", invalid="ignore"):
        accuracy_map = np.where(total_count > 0, correct_count / total_count, 0)

    return accuracy_map


def compute_class1_pixel_accuracy_map(results, valid_width=160, H=160):
    """
    Computes a pixel-wise accuracy map only for pixels where the true label is class 1.
    For each pixel (i,j) in the valid region (first valid_width columns), it calculates:
      accuracy = (# of times pixel correctly predicted as class 1) / (# of times pixel is class 1 in ground truth)

    Args:
      results (list): A list of result dictionaries. Each result must contain:
                      "true_mask": numpy array of shape (H, W)
                      "pred_mask": numpy array of shape (H, W)
      valid_width (int): Number of columns to consider (e.g., 160)
      H (int): Height of the masks (e.g., 160)

    Returns:
      accuracy_map: 2D numpy array of shape (H, valid_width) with values in [0,1]
    """
    correct_count = np.zeros((H, valid_width), dtype=np.float64)
    total_count = np.zeros((H, valid_width), dtype=np.int32)

    for res in results:
        # Consider only the valid region.
        true_mask = res["true_mask"][:, :valid_width]
        pred_mask = res["pred_mask"][:, :valid_width]

        # Create a boolean mask where ground truth is class 1.
        cls1_mask = true_mask == 2

        # For those pixels, count how many times the prediction is also class 1.
        correct_count += ((pred_mask == 2) & cls1_mask).astype(np.float64)
        # Also count the total number of times the true mask is class 1.
        total_count += cls1_mask.astype(np.int32)

    # Avoid division by zero.
    with np.errstate(divide="ignore", invalid="ignore"):
        accuracy_map = np.where(total_count > 0, correct_count / total_count, 0)

    return accuracy_map


def compute_pixel_confusion_matrices(results, num_classes=3, valid_width=160, H=160):
    """
    Computes an averaged confusion matrix for each pixel (i,j) over all samples.
    Returns an array of shape (H, valid_width, num_classes, num_classes).
    """
    # Initialize accumulators.
    confusion_acc = np.zeros(
        (H, valid_width, num_classes, num_classes), dtype=np.float64
    )
    counts = np.zeros((H, valid_width), dtype=np.int32)

    # Loop over all samples.
    for res in results:
        # Use only the valid region: first 160 columns.
        true_mask = res["true_mask"][:, :valid_width]  # shape: (H, valid_width)
        pred_mask = res["pred_mask"][:, :valid_width]  # shape: (H, valid_width)

        for i in range(H):
            for j in range(valid_width):
                if true_mask[i, j] == -1:
                    continue  # ignore
                t = int(true_mask[i, j])
                p = int(pred_mask[i, j])
                confusion_acc[i, j, t, p] += 1
                counts[i, j] += 1

    # Normalize per pixel.
    confusion_matrices = np.zeros_like(confusion_acc)
    for i in range(H):
        for j in range(valid_width):
            if counts[i, j] > 0:
                confusion_matrices[i, j] = confusion_acc[i, j] / counts[i, j]
            else:
                confusion_matrices[i, j] = np.zeros((num_classes, num_classes))
    return confusion_matrices


def average_confusion_in_blocks(confusion_matrices, block_size=5):
    """
    Given confusion_matrices of shape (H, W, num_classes, num_classes),
    average them over non-overlapping blocks of size (block_size x block_size).
    Returns an array of shape (H//block_size, W//block_size, num_classes, num_classes).
    """
    H, W, num_classes, _ = confusion_matrices.shape
    new_H = H // block_size
    new_W = W // block_size
    # Reshape to create blocks:
    blocks = confusion_matrices.reshape(
        new_H, block_size, new_W, block_size, num_classes, num_classes
    )
    # Average over block dimensions (axis 1 and 3)
    block_avg = blocks.mean(axis=(1, 3))
    return block_avg


def create_confusion_mosaic(block_confusions, display_block_size=5):
    """
    Creates a mosaic image from block-averaged confusion matrices.
    Each block (of shape (num_classes, num_classes)) is mapped to grayscale
    (or a chosen colormap) and then expanded to display_block_size x display_block_size pixels.

    The final mosaic will have shape (n_blocks * display_block_size, n_blocks * display_block_size).
    For readability, you might want to visualize each averaged confusion matrix as a small image.
    Here we simply take the averaged confusion matrix for each block, flatten it into a 1D vector,
    and reshape it into (display_block_size, display_block_size) after min-max normalization.
    (This is one way to visualize a 3x3 matrix in a 5x5 block.)

    Adjust the mapping as needed.
    """
    nH, nW, num_classes, _ = block_confusions.shape
    # For visualization, we first flatten each 3x3 confusion matrix to a 9-element vector.
    # Then, we rescale these 9 values to [0, 1] and reshape them to (display_block_size, display_block_size).
    mosaic = np.zeros((nH * display_block_size, nW * display_block_size))
    for i in range(nH):
        for j in range(nW):
            block = block_confusions[i, j]  # shape: (num_classes, num_classes)
            flat = block.flatten()  # shape: (9,)
            # Normalize flat values to [0, 1] for visualization.
            if flat.max() - flat.min() > 0:
                flat_norm = (flat - flat.min()) / (flat.max() - flat.min())
            else:
                flat_norm = flat
            # Resize flat_norm to a display block (e.g., 5x5) using simple interpolation.
            # For simplicity, we use PIL's Image.resize.
            flat_img = (flat_norm * 255).astype(np.uint8).reshape((3, 3))
            flat_pil = Image.fromarray(flat_img, mode="L")
            resized = flat_pil.resize(
                (display_block_size, display_block_size), resample=Image.BILINEAR
            )
            mosaic[
                i * display_block_size : (i + 1) * display_block_size,
                j * display_block_size : (j + 1) * display_block_size,
            ] = (
                np.array(resized) / 255.0
            )
    return mosaic


from sklearn.metrics import confusion_matrix
import seaborn as sns


def compute_global_confusion(results, num_classes=3, valid_width=160):
    """
    Computes a global 3x3 confusion matrix over all samples in results.

    Args:
        results (list): List of result dictionaries, each containing:
                        - "true_mask": numpy array of shape (H, W)
                        - "pred_mask": numpy array of shape (H, W)
        num_classes (int): Number of classes (default 3).
        valid_width (int): Number of columns to consider (e.g., 160).

    Returns:
        cm (np.array): Confusion matrix of shape (num_classes, num_classes) with raw counts.
    """
    all_true = []
    all_pred = []
    for res in results:
        # Consider only the valid region (first valid_width columns)
        true_mask = res["true_mask"][:, :valid_width]
        pred_mask = res["pred_mask"][:, :valid_width]
        # Exclude ignore pixels (true value == -1)
        valid = true_mask != -1
        all_true.extend(true_mask[valid].flatten())
        all_pred.extend(pred_mask[valid].flatten())
    cm = confusion_matrix(all_true, all_pred, labels=list(range(num_classes)))
    return cm


def plot_normalized_confusion_matrix(cm, class_names=None):
    """
    Normalizes the confusion matrix row-wise (each row sums to 1) and plots it with percentages.
    """
    # Normalize row-wise: avoid division by zero.
    with np.errstate(divide="ignore", invalid="ignore"):
        cm_normalized = np.divide(
            cm,
            cm.sum(axis=1, keepdims=True),
            out=np.zeros_like(cm, dtype=float),
            where=cm.sum(axis=1, keepdims=True) != 0,
        )

    if class_names is None:
        class_names = [f"Class {i}" for i in range(cm.shape[0])]

    plt.figure(figsize=(6, 5))
    # Use annot with format to show percentages.
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Normalized Global Confusion Matrix (Rows Sum to 1)")
    plt.show()


#########################
# Main Function
#########################
def main():
    # -----------------------------
    # Configuration – adjust paths and parameters.
    # -----------------------------
    images_dir = "./DATA/X_train/images"  # Raw images (npy)
    labels_csv = "./DATA/Y_train.csv"  # True labels CSV
    checkpoint_path = Path("checkpoints/model_60_64_0.0001_att.pth")
    num_classes = 3
    batch_size = 1

    test_loader = get_test_dataloader(
        images_dir, labels_csv, batch_size=batch_size, num_workers=4
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(num_classes)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    if False:
        results = collect_results(test_loader, model, device, images_dir, num_classes)
        print(f"Collected results for {len(results)} samples.")

        # Additionally, compute and plot mean confidence by IoU decile.
        thresholds, mean_conf = compute_confidence_by_iou_decile(
            results, num_classes, decile_bins=10
        )
        print("IoU thresholds per decile:", thresholds)
        print("Mean confidence per class per decile:", mean_conf)

    if False:
        results = pickle.load(open("results.pkl", "rb"))
        bin_centers, mean_conf, empirical_accuracy = (
            compute_reliability_data_from_results(results, num_bins=10)
        )
        plot_reliability_diagram(bin_centers, mean_conf, empirical_accuracy)

    if False:
        results = pickle.load(open("results.pkl", "rb"))
        for cls in range(3):  # for classes 0, 1, 2
            bin_centers, mean_conf, empirical_accuracy = (
                compute_reliability_data_by_class(
                    results, target_class=cls, num_bins=10
                )
            )
            plot_reliability_diagram_by_class(
                bin_centers, mean_conf, empirical_accuracy, target_class=cls
            )

    if True:
        results = pickle.load(open("results.pkl", "rb"))

        accuracy_map = compute_class1_pixel_accuracy_map(
            results, valid_width=160, H=160
        )

        plt.figure(figsize=(6, 6))
        plt.imshow(accuracy_map, cmap="viridis", vmin=0, vmax=1)
        plt.title("Pixel-wise Accuracy Map for Class 1")
        plt.colorbar(label="Accuracy")
        plt.show()

    if False:
        results = pickle.load(open("results.pkl", "rb"))

        # Compute per-pixel averaged confusion matrices over the valid region (first 160 columns).
        conf_matrices = compute_pixel_confusion_matrices(
            results, num_classes=3, valid_width=160, H=160
        )
        # Average these matrices over 5x5 pixel blocks.
        block_confusions = average_confusion_in_blocks(conf_matrices, block_size=5)
        # Create a mosaic image: each block in the mosaic represents the averaged confusion matrix.
        mosaic = create_confusion_mosaic(block_confusions, display_block_size=5)

        plt.figure(figsize=(8, 8))
        plt.imshow(mosaic, cmap="viridis", interpolation="nearest")
        plt.title("Averaged Pixel-wise Confusion Matrices (5x5 Blocks)")
        plt.colorbar(label="Normalized Value")
        plt.show()

    if False:
        results = pickle.load(open("results.pkl", "rb"))
        cm = compute_global_confusion(results, num_classes=3, valid_width=160)
        print("Raw Confusion Matrix:")
        print(cm)
        plot_normalized_confusion_matrix(
            cm, class_names=["Class 0", "Class 1", "Class 2"]
        )


if __name__ == "__main__":
    main()
