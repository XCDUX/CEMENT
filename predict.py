import os  # PACKAGES
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from model import get_model  # This returns a model from segmentation_models_pytorch
from PIL import Image

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.transforms import Resize, InterpolationMode, ToTensor


class CementTestDataset(Dataset):
    """
    Dataset for prediction that only loads raw test images (stored as .npy files).
    This dataset does not require a CSV of labels.

    Pre-processing:
      - Loads images from .npy files.
      - If the image size is 160x272, it resizes to 160x160.
      - Applies per-patch min-max normalization.
    """

    def __init__(self, images_dir, transform=None):
        self.images_dir = Path(images_dir)
        self.image_paths = sorted(list(self.images_dir.glob("*.npy")))
        if not self.image_paths:
            raise ValueError(f"No .npy files found in {self.images_dir}")
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        key = image_path.stem  # filename without extension

        # Load the image from the .npy file (expect shape (H, W) or (H, W, 1))
        image = np.load(image_path).astype(np.float32)
        if image.ndim == 3 and image.shape[-1] == 1:
            image = image.squeeze(-1)

        # Pre-processing: if the image width is not 160 (i.e. originally 160x272), resize to 160x160.
        if image.shape[1] != 160:
            resize_img = Resize((160, 160), interpolation=InterpolationMode.BILINEAR)
            image = Image.fromarray(image)
            image = np.array(resize_img(image))
        else:
            image = Image.fromarray(image)

        # Convert image to tensor and apply per-patch min-max normalization.
        image = ToTensor()(image)  # shape: [1, 160, 160]
        image_min, image_max = image.min(), image.max()
        if image_max - image_min > 0:
            image = (image - image_min) / (image_max - image_min)
        else:
            image = image - image_min  # constant image

        if self.transform:
            image = self.transform(image)

        # Return the processed image along with its key (filename) for later identification.
        return image, key


def get_pred_dataloader(images_dir, batch_size=1, num_workers=4):
    """
    Utility function to create a DataLoader for raw test images.
    """
    dataset = CementTestDataset(images_dir)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return dataloader


if True:  # FILE LOADING
    base_data_path = Path("DATA/X_test")  # e.g. "data/valid"
    valid_images_dir = (
        base_data_path / "images"
    )  # Folder containing valid images (.npy files)

    # Checkpoint path for the trained model
    checkpoint_path = Path("checkpoints/model_final.pth")

    # Where to save predictions (as .npy files)
    pred_dir = base_data_path / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    # Final CSV output file (as expected by the challenge)
    output_csv = base_data_path / "y_test_csv_file.csv"

    # Number of labels for output masks:
    # If background is 0 and you have 2 additional classes, your model should output 3 channels.
    num_classes = 3

    # The challenge expects all output masks to have width = 272, so:
    size_labels = 272

# -----------------------------
# Set up device and load model
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_model(num_classes)  # This should create a Unet with EfficientNet-B3 encoder
if checkpoint_path.exists():
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.to(device)
model.eval()

# -----------------------------
# Create dataset and dataloader for the valid set
# -----------------------------
# We use the same CementDataset to load images. The labels are not needed for prediction,
# but our dataset class expects a CSV; if you don't need it for valid, you can provide a dummy file.
valid_loader = get_pred_dataloader(valid_images_dir)

# -----------------------------
# Run prediction and save output masks
# -----------------------------
from tqdm import tqdm

if False:  # MAKING PREDICTIONS
    with torch.no_grad():
        for images, keys in tqdm(valid_loader, desc="Predicting samples", unit="batch"):
            images = images.to(device)
            outputs = model(images)  # Shape: [B, num_classes, H, W]
            preds = torch.argmax(outputs, dim=1).cpu().numpy()  # Shape: [B, H, W]

            # Assuming batch_size=1, use the key from the dataset to name the file.
            key = keys[0]
            np.save(pred_dir / f"{key}.npy", preds[0])


from torchvision.transforms import Resize, InterpolationMode
import re


def natural_keys(text):
    return [int(c) if c.isdigit() else c for c in re.split(r"(\d+)", text)]


# -----------------------------
# Post-process predictions and write CSV
# -----------------------------
# Get a sorted list of keys from the original test images using natural sort.
dataset_keys = sorted(
    [p.stem for p in valid_images_dir.glob("*.npy")], key=natural_keys
)

predictions = {"test": {}}
# For each key, load the corresponding prediction and original image,
# then adapt the 160x160 mask into a 160x272 mask as needed.
for key in tqdm(dataset_keys, desc="Processing predictions", unit="image"):
    pred_file = pred_dir / f"{key}.npy"
    if not pred_file.exists():
        print(f"Prediction file missing for key: {key}")
        continue

    # Load the prediction (assumed to be 160x160)
    prediction = np.load(pred_file)  # shape: (160, W) -- ideally W==160
    if prediction.shape[1] != 160:
        print(f"Unexpected prediction width for key {key}: {prediction.shape[1]}")
        continue
    mask_160 = prediction  # shape: (160,160)

    # Load the corresponding original image to check its width.
    image_path = valid_images_dir / f"{key}.npy"
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
print("Creating DataFrame to match challenge format")
df = pd.DataFrame(
    [predictions["test"][key] for key in dataset_keys], index=dataset_keys, dtype="int"
)
df.to_csv(output_csv)
print(f"Predictions CSV saved to: {output_csv}")
