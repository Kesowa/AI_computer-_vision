import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_smaller_inner_loops_skeleton(binary_image, max_area_threshold=500):
    """
    Detect smaller inner loops (holes) in the skeletonized binary image based on area constraints.
    Only loops with area less than `max_area_threshold` are considered.
    """
    # Ensure binary image is in uint8 format
    binary_image = binary_image.astype(np.uint8)

    # Find contours with two-level hierarchy
    contours, hierarchy = cv2.findContours(binary_image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    output_image = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)  # Create color output image

    smaller_loops = []

    for i, cnt in enumerate(contours):
        parent_index = hierarchy[0][i][3]  # Check the parent index
        area = cv2.contourArea(cnt)  # Calculate area of the contour

        # Filter for inner loops (parent_index != -1) and area below the threshold
        if parent_index != -1 and area < max_area_threshold:
            smaller_loops.append(cnt)
            cv2.drawContours(output_image, [cnt], -1, (0, 0, 255), 2)  # Highlight in red

    return output_image, len(smaller_loops)


# Example usage for a skeletonized image
# skeleton_visual is your skeletonized binary image
# Ensure that it contains only values 0 and 255

# Parameters
max_area_threshold = 100  # Adjust the threshold as needed

# Apply the function to detect smaller loops
smaller_loops_skeleton_image, num_smaller_loops_skeleton = detect_smaller_inner_loops_skeleton(
    skeleton_visual, max_area_threshold=max_area_threshold
)

# Display the resulting image with smaller inner loops highlighted
plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(smaller_loops_skeleton_image, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.title(f"Smaller Inner Loops Highlighted in Red (Count: {num_smaller_loops_skeleton})")
plt.tight_layout()
plt.show()

def remove_red_highlighted_pixels(original_image, processed_image):
    """
    Remove red-highlighted pixels from the original image.
    
    Parameters:
    - original_image: The original skeleton image (binary: 0 and 255).
    - processed_image: The processed image with red-highlighted pixels.
    
    Returns:
    - modified_image: The original image with red-highlighted portions removed.
    """
    # Ensure the images are in compatible formats
    original_image = original_image.astype(np.uint8)
    processed_image = processed_image.astype(np.uint8)

    # Convert the processed image to grayscale to isolate red regions
    red_mask = cv2.inRange(processed_image, np.array([0, 0, 200]), np.array([0, 0, 255]))

    # Remove red-highlighted pixels by setting them to black in the original image
    modified_image = original_image.copy()
    modified_image[red_mask > 0] = 0

    return modified_image


# Assuming `skeleton_visual` is the original image and `smaller_loops_skeleton_image` is the processed image
modified_image = remove_red_highlighted_pixels(skeleton_visual, smaller_loops_skeleton_image)

# Display the modified image
plt.figure(figsize=(10, 10))
plt.imshow(modified_image, cmap="gray")
plt.axis("off")
plt.title("Image After Removing Red Highlighted Portions")
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
import numpy as np
import cv2

# Assuming `image_cv` is your binary image with 1s as the region of interest
binary_original = modified_image

# Label connected components in the binary image
num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_original, connectivity=8)

# Find the label of the largest connected component of 1s (excluding the background, label 0)
largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])  # Skip label 0

# Create a mask for the largest connected component
binary_processed = (labels == largest_label).astype(np.uint8)

# Convert binary images to visualization format (0-255 range)
binary_original_visual = (binary_original * 255).astype(np.uint8)
binary_processed_visual = (binary_processed * 255).astype(np.uint8)

# Compute the difference between binary original and processed images
difference_binary = binary_original - binary_processed
difference_visual = (difference_binary * 255).astype(np.uint8)

# Display the original, processed, and difference images side by side
plt.figure(figsize=(15, 10))

# Binary original image
plt.subplot(1, 3, 1)
plt.imshow(binary_original_visual, cmap='gray')
plt.axis('off')
plt.title("Binary Original Image")

# Binary processed image (largest connected component)
plt.subplot(1, 3, 2)
plt.imshow(binary_processed_visual, cmap='gray')
plt.axis('off')
plt.title("Binary Processed Image (Largest Connected Component)")

# Difference image
plt.subplot(1, 3, 3)
plt.imshow(difference_visual, cmap='gray')
plt.axis('off')
plt.title("Difference (Binary Original - Processed)")

plt.tight_layout()
plt.show()
