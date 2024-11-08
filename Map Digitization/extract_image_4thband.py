import rasterio
import numpy as np
import cv2
import vtracer as vt
import matplotlib.pyplot as plt

def extract_largest_feature_from_band4(image_path, output_path):
    # Open the TIFF file and read only Band 4
    with rasterio.open(image_path) as src:
        band4 = src.read(4)  # Read Band 4, which contains meaningful data

    # Display histogram for Band 4 to confirm the pixel range
    plt.hist(band4.ravel(), bins=256, range=[0, 255])
    plt.title("Histogram of Band 4")
    plt.show()

    # Ensure the image is binary (0 and 255) by applying a threshold
    _, binary_image = cv2.threshold(band4, 127, 255, cv2.THRESH_BINARY)

    # Invert the binary image if necessary (if lines are black on white background)
    inverted_binary_image = cv2.bitwise_not(binary_image)

    # Apply morphological operations to connect lines and remove small gaps
    kernel = np.ones((5, 5), np.uint8)
    closed_image = cv2.morphologyEx(inverted_binary_image, cv2.MORPH_CLOSE, kernel)
    dilated_image = cv2.dilate(closed_image, kernel, iterations=3)

    # Remove small components (noise) using connected components
    num_labels, labels_im, stats, _ = cv2.connectedComponentsWithStats(dilated_image, connectivity=4)
    largest_area = max(stats[1:, cv2.CC_STAT_AREA])
    area_threshold = 0.06 * largest_area

    # Keep components larger than the area threshold
    component_mask = np.zeros_like(band4, dtype=np.uint8)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= area_threshold:
            component_mask[labels_im == i] = 255

    # Clean up further with morphological opening
    kernel_small = np.ones((2, 2), np.uint8)
    component_mask_cleaned = cv2.morphologyEx(component_mask, cv2.MORPH_OPEN, kernel_small)

    # Extract the features using the cleaned mask
    filtered_features = cv2.bitwise_and(band4, band4, mask=component_mask_cleaned)

    # Save the result as a TIFF file
    cv2.imwrite(output_path, cv2.bitwise_not(filtered_features))

    # Convert the processed TIFF image to SVG using vtracer for vector representation
    vt.convert_image_to_svg_py(
        output_path, 
        "output2.svg", 
        colormode="binary",
        hierarchical="cutout",
        mode="polygon",
        filter_speckle=8,
    )

# Example usage:
extract_largest_feature_from_band4('converted_output.tif', 'image_as2yoyo.png')
