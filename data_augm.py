import re
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import shutil
from torchvision.transforms import Resize, InterpolationMode

#########################
# Helper Functions
#########################


def natural_keys(text):
    # Ensure the input is a string before splitting.
    return [int(c) if c.isdigit() else c for c in re.split(r"(\d+)", str(text))]


def group_patches_by_well_section(patch_dir):
    """
    Groups patch file paths by well and section.
    Assumes filenames follow the pattern:
      well_<wellID>_section_<sectionID>_patch_<patchID>.npy
    Returns a dict of dicts:
      { well_id: { section_id: [list of patch paths sorted naturally] } }
    """
    patch_dir = Path(patch_dir)
    groups = {}
    for patch_path in patch_dir.glob("*.npy"):
        parts = re.split(r"[_\.]", patch_path.stem)
        # Expected parts: ["well", wellID, "section", sectionID, "patch", patchID]
        if len(parts) < 6:
            continue
        well_id = parts[1]
        section_id = parts[3]
        groups.setdefault(well_id, {}).setdefault(section_id, []).append(patch_path)
    # Sort each list naturally
    for well in groups:
        for section in groups[well]:
            groups[well][section] = sorted(
                groups[well][section], key=lambda p: natural_keys(p.stem)
            )
    return groups


def reconstruct_section(patch_paths, df_labels):
    """
    Reconstructs a section image (or label) by vertically stacking patches.
    For labels, each row in df_labels corresponds to a patch, with key same as patch filename.
    For each patch:
      - If the flattened label's tail (from index 160*160 onward) is all -1, then the patch is originally 160x160.
      - Otherwise, it is 160x272.
    Returns a 2D numpy array representing the reconstructed section.
    """
    patches = []
    for patch_path in patch_paths:
        key = patch_path.stem
        # For images, load the patch directly.
        patch = np.load(patch_path)
        # For labels, use the row from the CSV.
        if df_labels is not None:
            flat_label = df_labels.loc[key].values.astype(np.int32)
            # Determine shape based on padded region:
            if np.all(flat_label[160 * 160 :] == -1):
                patch = flat_label[: 160 * 160].reshape(160, 160)
            else:
                patch = flat_label.reshape(160, 272)
        patches.append(patch)
    section = np.vstack(patches)
    return section


def convert_section_to_target(section_img, target_width=272):
    """
    Converts a reconstructed section image or label to the target width.
    If the section's width is 160, pads the right with -1 (for labels) or replicates border (for images).
    If it is already target_width, returns as is.
    """
    current_width = section_img.shape[1]
    if current_width == target_width:
        return section_img
    elif current_width == 160:
        # For labels, pad with -1.
        # For images, you might want to upsample or pad with a specific value.
        # Here, we assume labels: pad with -1.
        padded = -1 * np.ones(
            (section_img.shape[0], target_width), dtype=section_img.dtype
        )
        padded[:, :160] = section_img
        return padded
    else:
        # Alternatively, upsample using nearest neighbor (works for both images and labels if desired)
        pil_img = Image.fromarray(section_img.astype(np.uint8))
        resizer = Resize(
            (section_img.shape[0], target_width),
            interpolation=InterpolationMode.NEAREST,
        )
        upsampled = np.array(resizer(pil_img))
        return upsampled


def sliding_window_extract_vertical(section_img, window_height=160, vertical_stride=80):
    """
    Extract overlapping patches from a section image using a vertical sliding window.
    Each patch spans the full width of the section (which is target width, 272).
    Returns a list of flattened patches.
    """
    patches = []
    img_h, img_w = section_img.shape
    for i in range(0, img_h - window_height + 1, vertical_stride):
        patch = section_img[i : i + window_height, :]
        patches.append(patch.flatten())
    return patches


def copy_original_patches(patch_dir, out_dir):
    """
    Copies the original patches from patch_dir into out_dir.
    """
    patch_dir = Path(patch_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in patch_dir.glob("*.npy"):
        shutil.copy(p, out_dir / p.name)


#########################
# Main Processing Function
#########################


def main():
    # Directories (adjust these paths to your environment)
    orig_images_dir = Path("./DATA/X_train/images")  # Original image patches
    labels_csv_path = Path(
        "./DATA/Y_train.csv"
    )  # Original labels CSV (each row flattened, index=patch name)
    new_dataset_dir = Path(
        "./DATA/augmented_dataset_strict"
    )  # New folder to contain both original and augmented patches
    new_dataset_dir.mkdir(exist_ok=True)
    new_csv_path = Path("./DATA/augmented_dataset_strict/labels.csv")

    # Load original labels CSV into a DataFrame.
    df_labels = pd.read_csv(labels_csv_path, header=None, index_col=0)

    # Group patches by well and section.
    groups = group_patches_by_well_section(orig_images_dir)

    # First, copy all original patches (images and labels) into the new dataset.
    # For images, just copy the file.
    # copy_original_patches(orig_images_dir, new_dataset_dir / "images")
    # For labels, we assume each row from df_labels is already in the proper 160*272 flattened form.
    # We will create a new labels dictionary for the new dataset.
    new_labels = {}
    for key in df_labels.index:
        new_labels[key] = df_labels.loc[key].values.astype(np.int32)

    # Now, process each well and each section to create augmented patches.
    vertical_stride = 80
    window_height = 160
    target_width = 272

    for well_id, sections in groups.items():
        for section_id, patch_paths in sections.items():
            # Reconstruct the section for images and labels separately.
            # For images: load from file paths (assumed shape: (160, W) where W is 160 or 272)
            section_img = np.vstack([np.load(p) for p in patch_paths])
            # For labels: reconstruct using the CSV rows.
            section_label = reconstruct_section(patch_paths, df_labels)

            # Convert both section image and section label to target width (272)
            section_img_target = convert_section_to_target(section_img, target_width)
            section_label_target = convert_section_to_target(
                section_label, target_width
            )

            # Save the reconstructed section images and labels (optional)
            well_section_id = f"well_{well_id}_section_{section_id}"
            np.save(
                new_dataset_dir / "images" / f"{well_section_id}_reconstructed.npy",
                section_img_target,
            )
            # We don't necessarily need to save the full label section, but you could:
            np.save(
                new_dataset_dir / "labels" / f"{well_section_id}_reconstructed.npy",
                section_label_target,
            )

            # Extract new augmented patches using vertical sliding window from the reconstructed section.
            new_patches = sliding_window_extract_vertical(
                section_img_target, window_height, vertical_stride
            )
            new_label_patches = sliding_window_extract_vertical(
                section_label_target, window_height, vertical_stride
            )

            print(
                f"Well {well_id} Section {section_id}: extracted {len(new_patches)} augmented patches."
            )

            # Save each new patch into the new dataset folder and add its label to new_labels.
            for idx, (img_patch_flat, label_patch_flat) in enumerate(
                zip(new_patches, new_label_patches)
            ):
                patch_key = f"{well_section_id}_aug_patch_{idx}"
                # Save image patch
                np.save(
                    new_dataset_dir / "images" / f"{patch_key}.npy",
                    img_patch_flat.reshape(window_height, target_width),
                )
                # Save label patch in our new_labels dict
                new_labels[patch_key] = label_patch_flat

    # Create a new CSV from new_labels dictionary.
    # Ensure natural order based on keys.
    sorted_keys = sorted(new_labels.keys(), key=natural_keys)
    df_new = pd.DataFrame(
        [new_labels[k] for k in sorted_keys], index=sorted_keys, dtype=int
    )
    # Flatten each row is already flattened.
    df_new.to_csv(new_csv_path, header=False)
    print(f"New augmented labels CSV saved to: {new_csv_path}")


if __name__ == "__main__":
    main()
