import os
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, ConcatDataset, random_split
import torch.nn.functional as F
from model import get_model
from data import CementDataset, get_train_test_dataloaders
import torch.nn as nn
import torch.optim as optim
from train import compute_iou
from predict import get_pred_dataloader, CementTestDataset, natural_keys
from tqdm import tqdm
from PIL import Image
import pandas as pd
import random
from torchvision.transforms import Resize, InterpolationMode

# Paths to the labeled and unlabeled datasets
LABELED_IMAGES_DIR = Path("DATA/X_train/images")
LABELED_LABELS_CSV = Path("DATA/Y_train.csv")
UNLABELED_IMAGES_DIR = Path("DATA/X_unlabeled/images")
MODEL_PATH = Path("checkpoints/model_60_16_0.0001_b7.pth")
PRED_DIR = Path("DATA/X_unlabeled/predictions")
LABELS = "DATA/Y_train.csv"
OUTPUT_CSV = "DATA/X_unlabeled/y_pseudo_label.csv"
DEST_FOLDER = Path("DATA/merged_dataset/images")
MERGED_CSV = "DATA/merged_dataset/y_labels.csv"
DEST_FOLDER.mkdir(parents=True, exist_ok=True)
SSL_THRESHOLD = 0.9  # Confidence threshold for pseudo-label selection
BATCH_SIZE = 16
NUM_EPOCHS = 100
LEARNING_RATE = 1e-4
NUM_WORKERS = min(os.cpu_count(), 8)
NUM_CLASSES = 3
PRED_DIR.mkdir(parents=True, exist_ok=True)
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

# Load last trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("==============================")
print("Using device:", device)
print("==============================")
model = get_model(num_classes=3).to(device)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

if True:
    print("Preparing unlabeled dataset")
    print("==============================")

    unlabeled_loader = get_pred_dataloader(UNLABELED_IMAGES_DIR)

    print("Making predictions.")
    print("==============================")
    with torch.no_grad():
        for images, keys in tqdm(
            unlabeled_loader, desc="Predicting samples", unit="batch"
        ):
            images = images.to(device)
            outputs = model(images)  # Shape: [B, num_classes, H, W]
            preds = torch.argmax(outputs, dim=1).cpu().numpy()  # Shape: [B, H, W]

            # Assuming batch_size=1, use the key from the dataset to name the file.
            key = keys[0]
            np.save(PRED_DIR / f"{key}.npy", preds[0])

if True:
    # -----------------------------
    # Post-process predictions and write CSV
    # -----------------------------
    # Get a sorted list of keys from the original test images using natural sort.
    dataset_keys = sorted(
        [p.stem for p in UNLABELED_IMAGES_DIR.glob("*.npy")], key=natural_keys
    )

    predictions = {"test": {}}
    # For each key, load the corresponding prediction and original image,
    # then adapt the 160x160 mask into a 160x272 mask as needed.
    for key in tqdm(dataset_keys, desc="Processing predictions", unit="image"):
        pred_file = PRED_DIR / f"{key}.npy"
        if not pred_file.exists():
            print(f"Prediction file missing for key: {key}")
            continue

        # Load the prediction (assumed to be 160x160)
        prediction = np.load(pred_file)

        if prediction.shape[1] != 160:
            print(f"Unexpected prediction width for key {key}: {prediction.shape[1]}")
            continue
        mask_160 = prediction  # shape: (160,160)

        # Load the corresponding original image to check its width.
        image_path = UNLABELED_IMAGES_DIR / f"{key}.npy"
        image = np.load(image_path).astype(np.float32)
        if image.ndim == 3 and image.shape[-1] == 1:
            image = image.squeeze(-1)
        orig_width = image.shape[1]

        if orig_width == 272:
            # Upsample the 160x160 mask to 160x272 using nearest neighbor interpolation.
            # Convert the mask to a PIL image.
            mask_pil = Image.fromarray(mask_160.astype(np.uint8))
            resizer = Resize((160, 272), interpolation=InterpolationMode.NEAREST)
            mask_resized = np.array(resizer(mask_pil))
            final_mask = mask_resized.flatten()
        elif orig_width == 160:
            # Pad the 160x160 mask with -1 on the right to get 160x272.
            padded_mask = -1 * np.ones((160, 272), dtype=int)
            # Place the 160x160 mask in the left side.
            padded_mask[:, :160] = mask_160
            final_mask = padded_mask.flatten()
        else:
            print(f"Unexpected original image width for {key}: {orig_width}")
            continue

        predictions["test"].update({key: final_mask})

    # Create DataFrame ensuring the order follows dataset_keys.
    print("Creating DataFrame")
    df = pd.DataFrame(
        [predictions["test"][key] for key in dataset_keys],
        index=dataset_keys,
        dtype="int",
    )
    df.to_csv(OUTPUT_CSV)
    print(f"Predictions CSV saved to: {OUTPUT_CSV}")
    print("Data successfully loaded.")
    print("==============================")

    import shutil

    print("Merging Datasets.")
    print("==============================")
    # Move files from both folders to the merged dataset
    for src_folder in [LABELED_IMAGES_DIR, UNLABELED_IMAGES_DIR]:
        for image_path in src_folder.glob("*.npy"):  # Adjust file extension if needed
            dest_path = DEST_FOLDER / image_path.name
            if dest_path.exists():
                print(f"⚠️ Warning: Duplicate file {image_path.name}, skipping.")
            else:
                shutil.copy(image_path, dest_path)  # Copy instead of move if needed

    print(f"✅ Merged images into: {DEST_FOLDER}")

    df1 = pd.read_csv(LABELS, index_col=0)
    df2 = pd.read_csv(OUTPUT_CSV, index_col=0)

    # Merge CSVs (avoid duplicate keys by keeping existing labels)
    df_merged = pd.concat([df1, df2[~df2.index.isin(df1.index)]], axis=0)

    # Save the merged CSV
    df_merged.to_csv(MERGED_CSV)
    print(f"✅ Merged CSV saved to: {MERGED_CSV}")

train_loader, test_loader = get_train_test_dataloaders(
    str(DEST_FOLDER), MERGED_CSV, batch_size=BATCH_SIZE, train_ratio=0.9
)

best_val_loss = float("inf")
patience_counter = 0
PATIENCE = 5
criterion = nn.CrossEntropyLoss(ignore_index=-1).to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print("SSL Training started")
print("==============================")
for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    running_iou = 0.0
    num_batches = 0

    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        outputs = F.interpolate(
            outputs, size=masks.shape[1:], mode="bilinear", align_corners=True
        )

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

    model.eval()
    test_loss = 0.0
    test_iou = 0.0
    test_batches = 0

    with torch.no_grad():
        for images, masks in test_loader:
            images, masks = images.to(device), masks.to(device)
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

    # Early stopping check
    if avg_test_loss < best_val_loss:
        best_val_loss = avg_test_loss
        patience_counter = 0
        torch.save(
            model.state_dict(),
            os.path.join(
                "checkpoints",
                f"ssl_model_b7.pth",
            ),
        )
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

print("==============================")

print("Training complete. Model saved as checkpoints/ssl_model.pth")
