import os
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from torchvision.transforms import Resize, InterpolationMode, ToTensor


def augment(image, mask):
    # TODO: implement augmentation logic
    return image, mask


class CementDataset(Dataset):
    def __init__(self, images_dir, labels_csv, transform=None, augment_fn=augment):
        self.images_dir = Path(images_dir)
        self.labels_df = pd.read_csv(labels_csv, index_col=0)
        self.transform = transform
        self.augment_fn = augment_fn

        # Assume image file names (without extension) match the CSV index.
        self.image_paths = sorted(list(self.images_dir.glob("*.*")))
        if not self.image_paths:
            raise ValueError(f"No image files found in {self.images_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        image_path = self.image_paths[idx]
        key = image_path.stem  # key matching the CSV row
        image = np.load(image_path).astype(np.float32)

        # Retrieve corresponding mask from CSV.
        # The CSV file is expected to have a row per image (indexed by file name)
        # with a flattened list of pixel values, where padded values are -1.
        if key not in self.labels_df.index:
            raise KeyError(f"Key {key} not found in labels CSV.")
        mask_row = self.labels_df.loc[key].values
        # Remove padded values (-1) and reshape. We assume a height of 160.
        valid_values = np.array([v for v in mask_row if v != -1], dtype=np.int64)
        # Determine the width from the total number of valid pixels.
        width = int(len(valid_values) / 160)
        mask = valid_values.reshape(160, width)

        # Pre-processing: resize if needed.
        # If image (or mask) width is not 160 (i.e. original size is 160 x 272),
        # resize to 160 x 160.
        if image.shape[1] != 160:
            # For the mask, use nearest neighbor to preserve labels.
            resize_mask = Resize((160, 160), interpolation=InterpolationMode.NEAREST)
            mask = Image.fromarray(mask.astype(np.uint8))
            mask = np.array(resize_mask(mask))

            # For the image, use bilinear interpolation.
            resize_img = Resize((160, 160), interpolation=InterpolationMode.BILINEAR)
            image = Image.fromarray(image)
            image = np.array(resize_img(image))
        else:
            # Convert arrays to PIL images if not already.
            image = Image.fromarray(image)
            mask = Image.fromarray(mask.astype(np.uint8))

        # Convert image to tensor and apply min-max normalization per patch.
        image = ToTensor()(image)  # results in shape [1, 160, 160]
        image_min, image_max = image.min(), image.max()
        if image_max - image_min > 0:
            image = (image - image_min) / (image_max - image_min)
        else:
            image = image - image_min  # image is constant

        # Convert mask to tensor (do not normalize mask values)
        mask = torch.from_numpy(np.array(mask)).long()

        # Apply augmentation if provided.
        image, mask = self.augment_fn(image, mask)

        # Apply additional transforms if provided.
        if self.transform:
            image = self.transform(image)

        return image, mask


def get_train_test_dataloaders(
    images_dir, labels_csv, batch_size=64, train_ratio=0.8, num_workers=4
):
    """
    Splits the CementDataset into training and testing sets and returns corresponding DataLoaders.

    Parameters:
      images_dir (str): Path to the directory with .npy image files.
      labels_csv (str): Path to the CSV file with flattened mask labels.
      batch_size (int): Batch size for DataLoaders.
      train_ratio (float): Fraction of the data to use for training.
      num_workers (int): Number of worker processes for DataLoaders.

    Returns:
      train_loader, test_loader: DataLoaders for the training and testing sets.
    """
    dataset = CementDataset(images_dir, labels_csv)
    train_size = int(len(dataset) * train_ratio)
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, test_loader


def visualize_sample(image, mask, mask_alpha=0.5, mask_cmap="jet"):

    # Convert tensors to numpy arrays if needed.
    if isinstance(image, torch.Tensor):
        image = image.squeeze(0).cpu().numpy()
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()

    # Create figure with 2 subplots in one row.
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Left: original image (grayscale)
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # Right: original image with mask overlay.
    axes[1].imshow(image, cmap="gray")
    # Overlay the mask using a colormap and transparency.
    axes[1].imshow(mask, cmap=mask_cmap, alpha=mask_alpha)
    axes[1].set_title("Image with Mask Overlay")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Example usage for debugging the dataloader.
    # Replace 'path/to/images' and 'path/to/labels.csv' with actual paths.
    images_dir = "./DATA/X_train/images"
    labels_csv = "./DATA/Y_train.csv"

    # Visualize the first batch from the train loader.
    train_loader, test_loader = get_train_test_dataloaders(
        images_dir, labels_csv, batch_size=8, train_ratio=0.8
    )
    for batch_idx, (images, masks) in enumerate(train_loader):
        print(f"Batch {batch_idx}:")
        print(f"  Images shape: {images.shape}")
        print(f"  Masks shape: {masks.shape}")
        # Visualize each sample in the batch on a separate row.
        for i in range(images.size(0)):
            visualize_sample(images[i], masks[i])
        break  # Only process the first batch
