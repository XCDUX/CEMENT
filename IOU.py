import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from model import get_model  # Your model from segmentation_models_pytorch
from data import get_test_dataloader  # Your train/test split function
from pathlib import Path


def accumulate_iou_metrics(true_mask, pred_mask, num_classes):
    intersections = np.zeros(num_classes, dtype=np.float64)
    unions = np.zeros(num_classes, dtype=np.float64)
    for cls in range(num_classes):
        true_cls = true_mask == cls
        pred_cls = pred_mask == cls
        intersections[cls] = np.logical_and(true_cls, pred_cls).sum()
        unions[cls] = np.logical_or(true_cls, pred_cls).sum()
    return intersections, unions


def compute_mean_iou(train_loader, model, num_classes, device):
    total_intersections = np.zeros(num_classes, dtype=np.float64)
    total_unions = np.zeros(num_classes, dtype=np.float64)

    model.eval()
    with torch.no_grad():
        for images, true_masks, _ in tqdm(
            train_loader, desc="Computing IoU", unit="batch"
        ):
            images = images.to(device)
            # Obtain model outputs (assumed shape: [B, num_classes, H, W])
            outputs = model(images)
            # Compute predictions via argmax.
            preds = torch.argmax(outputs, dim=1).cpu().numpy()  # shape: [B, H, W]
            true_masks_np = true_masks.cpu().numpy()  # shape: [B, H, W]

            # Process each sample in the batch.
            for t_mask, p_mask in zip(true_masks_np, preds):
                intersections, unions = accumulate_iou_metrics(
                    t_mask, p_mask, num_classes
                )
                total_intersections += intersections
                total_unions += unions

    # Compute IoU per class (handle division by zero)
    ious = np.divide(
        total_intersections,
        total_unions,
        out=np.zeros_like(total_intersections),
        where=total_unions != 0,
    )
    return ious


def main():
    # -----------------------------
    # Configuration – adjust paths and parameters as needed.
    # -----------------------------
    images_dir = "./DATA/X_train/images"  # Path to training images
    labels_csv = "./DATA/Y_train.csv"  # Path to training labels CSV
    batch_size = 16
    num_classes = 3  # For example: background=0, plus two classes: 1 and 2.
    checkpoint_path = Path("checkpoints/model_60_64_0.0001_att.pth")

    # Get train and test dataloaders.
    train_loader = get_test_dataloader(images_dir, labels_csv, batch_size=batch_size)

    # Set up device and load model.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(num_classes)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)

    # Compute IoU per class over the training dataset.
    ious = compute_mean_iou(train_loader, model, num_classes, device)
    for cls in range(num_classes):
        print(f"Mean IoU for class {cls}: {ious[cls]:.4f}")


if __name__ == "__main__":
    main()
