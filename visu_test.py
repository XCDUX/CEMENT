import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def visualize_sample(image, mask, mask_alpha=0.5, mask_cmap="jet"):

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(image, cmap="gray")
    axes[1].imshow(mask, cmap=mask_cmap, alpha=mask_alpha)
    axes[1].set_title("Prediction Overlay")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()


def main():
    # Define paths (adjust these paths to your environment)
    base_data_path = Path("./DATA/X_test")
    images_dir = base_data_path / "images"  # Folder with original .npy images
    csv_path = (
        base_data_path / "y_test_csv_file.csv"
    )  # CSV file with flattened predictions

    df = pd.read_csv(csv_path, header=None, index_col=0)
    count = 0

    for key in df.index:
        # Load the corresponding original image.
        image_path = images_dir / f"{key}.npy"
        if not image_path.exists():
            print(f"Image file not found: {image_path}")
            continue
        image = np.load(image_path).astype(np.float32)
        # If image has a singleton channel, squeeze it.
        if image.ndim == 3 and image.shape[-1] == 1:
            image = image.squeeze(-1)

        # Retrieve the flattened prediction.
        flat_pred = df.loc[key].values.astype(np.int32)
        print(image.shape)
        if image.shape[1] == 160:
            mask = flat_pred[: 160 * 160].reshape(160, 160)
        else:
            mask = flat_pred.reshape(160, 272)

        # Visualize: display original image and prediction overlay.
        visualize_sample(image, mask)
        count += 1
        if count == 5:
            break


if __name__ == "__main__":
    main()
