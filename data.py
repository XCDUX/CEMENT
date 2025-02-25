import os
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import pickle
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from torchvision.transforms import (
    RandomHorizontalFlip,
    RandomVerticalFlip,
    RandomRotation,
    ColorJitter,
    Resize,
    InterpolationMode,
    ToTensor,
)
import random
from scipy.ndimage import gaussian_filter, map_coordinates


def elastic_transform(image, mask, alpha, sigma, random_state=None):
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.size
    dx = (
        gaussian_filter(
            (random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0
        )
        * alpha
    )
    dy = (
        gaussian_filter(
            (random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0
        )
        * alpha
    )

    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
    indices = np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1))

    distorted_image = map_coordinates(np.array(image), indices, order=1).reshape(shape)
    distorted_mask = map_coordinates(np.array(mask), indices, order=0).reshape(shape)

    return Image.fromarray(distorted_image), Image.fromarray(distorted_mask)


def augment(image, mask):
    if random.random() > 0.5:
        image = RandomHorizontalFlip(p=1)(image)
        mask = RandomHorizontalFlip(p=1)(mask)
    if random.random() > 0.5:
        image = RandomVerticalFlip(p=1)(image)
        mask = RandomVerticalFlip(p=1)(mask)
    if random.random() > 0.5:
        angle = random.uniform(-30, 30)
        image = RandomRotation((angle, angle))(image)
        mask = RandomRotation((angle, angle))(mask)
    if random.random() > 0.5:
        # Convert grayscale to RGB for ColorJitter
        image = image.convert("RGB")
        brightness_factor = random.uniform(0.8, 1.2)
        contrast_factor = random.uniform(0.8, 1.2)
        image = ColorJitter(brightness=brightness_factor, contrast=contrast_factor)(
            image
        )
        # Convert back to grayscale
        image = image.convert("L")
    if random.random() > 0.5:
        image, mask = elastic_transform(image, mask, alpha=2, sigma=0.2)

    return image, mask


class CementDataset(Dataset):
    def __init__(
        self,
        images_dir,
        labels_csv,
        transform=elastic_transform,
        augment_fn=augment,
        preload=True,
        cache_file="image_paths.pkl",
    ):
        self.images_dir = Path(images_dir)
        self.labels_df = pd.read_csv(labels_csv, index_col=0) if labels_csv else None
        self.transform = transform
        self.augment_fn = augment_fn
        self.preload = preload  # Ensure preload is initialized
        self.cache_file = Path(cache_file)

        # Load cached file paths or scan directory if cache is missing
        if self.cache_file.exists():
            print("[INFO] Loading image paths from cache...")
            with open(self.cache_file, "rb") as f:
                self.image_paths = pickle.load(f)
        else:
            print("[INFO] Scanning directory for image files...")
            self.image_paths = sorted(list(self.images_dir.glob("*.npy")))
            with open(self.cache_file, "wb") as f:
                pickle.dump(self.image_paths, f)

        if not self.image_paths:
            raise ValueError(f"No image files found in {self.images_dir}")

        # Preload dataset into memory if enabled
        self.data_cache = {}
        if self.preload:
            print("[INFO] Preloading data into RAM...")
            for image_path in self.image_paths:
                self.data_cache[image_path.stem] = np.load(image_path).astype(
                    np.float32
                )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        key = image_path.stem
        image = (
            self.data_cache[key]
            if self.preload
            else np.load(image_path, mmap_mode="r").astype(np.float32)
        )

        if self.labels_df is not None and key not in self.labels_df.index:
            raise KeyError(f"Key {key} not found in labels CSV.")
        mask_row = (
            self.labels_df.loc[key].values
            if self.labels_df is not None
            else np.zeros((160, 160))
        )

        valid_values = np.array([v for v in mask_row if v != -1], dtype=np.int64)
        width = int(len(valid_values) / 160) if self.labels_df is not None else 160
        mask = (
            valid_values.reshape(160, width)
            if self.labels_df is not None
            else np.zeros((160, 160))
        )

        image = Image.fromarray(image)
        mask = Image.fromarray(mask.astype(np.uint8))

        if self.augment_fn:
            image, mask = self.augment_fn(image, mask)

        image = image.resize((160, 160), Image.BILINEAR)
        mask = mask.resize((160, 160), Image.NEAREST)

        image = ToTensor()(image)
        mask = torch.from_numpy(np.array(mask)).long()

        image_min, image_max = image.min(), image.max()
        if image_max - image_min > 0:
            image = (image - image_min) / (image_max - image_min)

        return image, mask


def get_train_test_dataloaders(
    images_dir, labels_csv, batch_size=64, train_ratio=0.8, num_workers=None
):
    if num_workers is None:
        num_workers = 8  # Dynamically adjust workers

    dataset = CementDataset(images_dir, labels_csv)
    train_size = int(len(dataset) * train_ratio)
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
    )
    return train_loader, test_loader
