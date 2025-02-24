import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import get_model
from data import (
    get_train_test_dataloaders,
)  # our data loading pipeline with train-test split

# Set training parameters according to challenge recommendations
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
NUM_EPOCHS = 15
NUM_CLASSES = 3  # Adjust this depending on the number of segmentation classes

# Paths to your training images and CSV labels (update these accordingly)
IMAGES_DIR = "DATA/X_train/images"
LABELS_CSV = "DATA/Y_train.csv"


def compute_iou(pred, target, num_classes):
    """
    Compute the Intersection over Union (IoU) for each class and average them.
    pred: tensor of shape (B, H, W) with predicted class indices.
    target: tensor of shape (B, H, W) with ground truth indices.
    """
    ious = []
    pred = pred.view(-1)
    target = target.view(-1)

    for cls in range(num_classes):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = (pred_inds & target_inds).sum().float()
        union = (pred_inds | target_inds).sum().float()
        if union == 0:
            iou = torch.tensor(1.0, device=pred.device)  # if no target, IoU is 1
        else:
            iou = intersection / union
        ious.append(iou)
    return sum(ious) / len(ious)


def train():
    # Use GPU if available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Obtain train and test DataLoaders.
    train_loader, test_loader = get_train_test_dataloaders(
        IMAGES_DIR, LABELS_CSV, batch_size=BATCH_SIZE, train_ratio=0.8
    )

    model = get_model(NUM_CLASSES).to(device)

    # Loss function and optimizer.
    criterion = nn.CrossEntropyLoss(ignore_index=-1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        running_iou = 0.0
        num_batches = 0

        # Training loop.
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)  # [B, num_classes, H, W]
            outputs = F.interpolate(
                outputs, size=masks.shape[1:], mode="bilinear", align_corners=True
            )
            if torch.any(masks < 0) or torch.any(masks >= NUM_CLASSES):
                print("Invalid mask values found:", masks.unique())

            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            batch_iou = compute_iou(preds, masks, NUM_CLASSES)
            running_iou += batch_iou.item()
            num_batches += 1

        avg_train_loss = running_loss / num_batches
        avg_train_iou = running_iou / num_batches

        # Evaluation on test data.
        model.eval()
        test_loss = 0.0
        test_iou = 0.0
        test_batches = 0
        with torch.no_grad():
            for images, masks in test_loader:
                images = images.to(device)
                masks = masks.to(device)

                outputs = model(images)
                outputs = F.interpolate(
                    outputs, size=masks.shape[1:], mode="bilinear", align_corners=True
                )
                loss = criterion(outputs, masks)
                test_loss += loss.item()

                preds = torch.argmax(outputs, dim=1)
                batch_iou = compute_iou(preds, masks, NUM_CLASSES)
                test_iou += batch_iou.item()
                test_batches += 1

        avg_test_loss = test_loss / test_batches
        avg_test_iou = test_iou / test_batches

        print(
            f"Epoch [{epoch+1}/{NUM_EPOCHS}] - "
            f"Train Loss: {avg_train_loss:.4f}, Train IoU: {avg_train_iou:.4f} | "
            f"Test Loss: {avg_test_loss:.4f}, Test IoU: {avg_test_iou:.4f}"
        )

    # Save the model checkpoint.
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(
        model.state_dict(),
        os.path.join(
            "checkpoints", f"model_{NUM_EPOCHS}_{BATCH_SIZE}_{LEARNING_RATE}.pth"
        ),
    )
    print(
        f"Training complete. Model saved to checkpoints/model_{NUM_EPOCHS}_{BATCH_SIZE}_{LEARNING_RATE}.pth"
    )


if __name__ == "__main__":
    train()
