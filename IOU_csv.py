import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from torchvision.transforms import Resize, InterpolationMode
from tqdm import tqdm
import pickle
import pandas as pd
import re
from data import get_test_dataloader
from model import get_model


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


def collect_results(
    test_loader,
    model,
    device,
    images_dir,
    num_classes=3,
    results_pickle_path="results.pkl",
    results_csv_path="results_summary.csv",
):
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

    After processing, the function saves:
      - A pickle file containing the full results list.
      - A CSV file summarizing each sample (key and IoU).
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
            orig_width = raw_img.shape[1]  # Should be 160 or 272

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

            # Compute IoU between true_adj and pred_mask_adj.
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

    # Save the full results to a pickle file.
    results_pickle = Path(results_pickle_path)
    with open(results_pickle, "wb") as f:
        pickle.dump(results, f)
    print(f"Results saved to pickle: {results_pickle}")

    # Create a CSV summary of keys and IoU.
    summary = [(r["key"], r["iou"]) for r in results]
    df_summary = pd.DataFrame(summary, columns=["key", "iou"])
    df_summary.to_csv(results_csv_path, index=False)
    print(f"Results summary CSV saved to: {results_csv_path}")

    return results


# Example usage:
if __name__ == "__main__":
    # Paths and parameters – adjust as needed.
    images_dir = "./DATA/X_train/images"  # Raw training images directory
    labels_csv = "./DATA/Y_train.csv"  # True labels CSV
    batch_size = 1
    num_classes = 3
    checkpoint_path = Path("checkpoints/model_60_64_0.0001_att.pth")

    # Assuming get_test_dataloader returns (image, true_mask, key)
    from data import get_test_dataloader

    test_loader = get_test_dataloader(
        images_dir, labels_csv, batch_size=batch_size, num_workers=4
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(num_classes)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)

    results = collect_results(test_loader, model, device, images_dir, num_classes)
