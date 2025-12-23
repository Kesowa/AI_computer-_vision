import os
import numpy as np
from PIL import Image

def otsu_threshold_from_image(img_uint8):
    """Compute Otsu threshold for a uint8 image."""
    hist = np.bincount(img_uint8.flatten(), minlength=256).astype(np.float64)
    total = hist.sum()
    if total == 0:
        return 127

    bins = np.arange(256)
    sum_total = (bins * hist).sum()
    sumB = 0.0
    wB = 0.0
    max_var = -1.0
    threshold = 0

    for t in range(256):
        wB += hist[t]
        if wB == 0:
            continue
        wF = total - wB
        if wF == 0:
            break
        sumB += t * hist[t]
        mB = sumB / wB
        mF = (sum_total - sumB) / wF
        var_between = wB * wF * (mB - mF) ** 2
        if var_between > max_var:
            max_var = var_between
            threshold = t

    return int(threshold)


def threshold_png(
    input_png,
    output_png=None,
    threshold=None,
    invert=False,
    debug=True
):
    """
    Apply thresholding to a PNG image.

    Parameters:
    - input_png: path to input PNG (grayscale or RGB)
    - output_png: output path (auto-generated if None)
    - threshold: None = auto Otsu, else manual [0–255]
    - invert: invert binary output
    - debug: print statistics
    """

    # Load image
    img = Image.open(input_png)

    # Convert to grayscale if RGB
    if img.mode != "L":
        img = img.convert("L")

    img_np = np.asarray(img, dtype=np.uint8)

    if debug:
        print("=" * 80)
        print("INPUT IMAGE")
        print("=" * 80)
        print(f"Path: {input_png}")
        print(f"Size: {img_np.shape}")
        print(f"Dtype: {img_np.dtype}")
        print(f"Min: {img_np.min()}, Max: {img_np.max()}")

    # Auto output path
    if output_png is None:
        base, ext = os.path.splitext(input_png)
        tag = "auto" if threshold is None else str(threshold)
        inv = "_inv" if invert else ""
        output_png = f"{base}_thr{tag}{inv}{ext}"

    # Determine threshold
    if threshold is None:
        threshold = otsu_threshold_from_image(img_np)

    if debug:
        print("\nTHRESHOLD")
        print("-" * 40)
        print(f"Threshold value: {threshold}")

    # Apply threshold
    mask = img_np > threshold

    if invert:
        out = np.where(mask, 0, 255).astype(np.uint8)
    else:
        out = np.where(mask, 255, 0).astype(np.uint8)

    white_pixels = int(mask.sum())
    total_pixels = img_np.size

    if debug:
        print("\nRESULT")
        print("-" * 40)
        print(f"White pixels: {white_pixels:,} / {total_pixels:,} "
              f"({100.0 * white_pixels / total_pixels:.2f}%)")

    # Save output
    Image.fromarray(out, mode="L").save(output_png)

    if debug:
        print(f"Output saved to: {output_png}")
        print("=" * 80)

    return output_png


# ------------------------------
# MAIN
# ------------------------------
if __name__ == "__main__":
    input_png = "/home/debian/Downloads/tree_detection/rooftop_output_onnx.png"

    # Option 1: Auto Otsu
    # threshold_png(
    #     input_png=input_png,
    #     threshold=None,   # Auto Otsu
    #     invert=False,
    #     debug=True
    # )

    # Option 2: Manual threshold (uncomment if needed)
    threshold_png(
        input_png=input_png,
        threshold=15,
        invert=False,
        debug=True
    )
