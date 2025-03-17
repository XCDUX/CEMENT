import numpy as np
import pandas as pd
from pathlib import Path


def curate_predictions(
    input_csv, output_csv, height=160, width=272, left_fraction=0.75
):
    """
    Loads a CSV of flattened predictions (each row of length height*width), and curates them
    by replacing class labels in the right (1 - left_fraction) portion of the image.

    In this example, for columns starting at int(width * left_fraction):
      - if pixel == 1, replace with 2
      - if pixel == 2, replace with 1
      - leave background (0) unchanged.

    Parameters:
        input_csv (str or Path): Path to the input predictions CSV.
        output_csv (str or Path): Path to save the curated CSV.
        height (int): The height of the mask (default 160).
        width (int): The width of the mask (default 272).
        left_fraction (float): Fraction of the image on the left that remains unchanged.
    """
    df = pd.read_csv(input_csv, header=None, index_col=0)
    new_rows = {}
    start_col = int(
        width * left_fraction
    )  # for width=272 and left_fraction=0.25, start_col ~68

    for idx, row in df.iterrows():
        flat = row.values.astype(int)
        mask = flat.reshape(height, width)
        curated_mask = mask.copy()
        # Select the region to modify: all rows, columns from start_col to the end.
        region = curated_mask[:, start_col:width]
        # Use vectorized operations to swap classes:
        # Create a boolean mask for class 1 and class 2.
        swap_2 = region == 2
        swap_1 = region == 1
        region[swap_2] = 0
        region[swap_1] = 0
        # Place the modified region back into the curated_mask.
        curated_mask[:, start_col:width] = region
        # Save the flattened curated mask in our new dictionary.
        new_rows[idx] = curated_mask.flatten()

    # Create a new DataFrame from the curated predictions.
    df_new = pd.DataFrame.from_dict(new_rows, orient="index")
    df_new.to_csv(output_csv, header=False)
    print(f"Curated predictions CSV saved to: {output_csv}")


if __name__ == "__main__":
    # Adjust these paths as needed.
    input_csv = Path("./y_test_b2.csv")
    output_csv = Path("./y_b2_post_processed.csv")
    curate_predictions(input_csv, output_csv)
