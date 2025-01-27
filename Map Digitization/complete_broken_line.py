import cv2
import numpy as np
from matplotlib import pyplot as plt

def complete_broken_line(image_path, kernel_size=(5, 5)):
    """
    Completes broken lines in a binary image using morphological operations.

    Args:
        image_path (str): Path to the input image.
        kernel_size (tuple): Size of the kernel for morphological operations.

    Returns:
        processed_image: Image with completed lines.
    """
    # Load the image
    image = binary_processed_visual

    # Convert to binary (0 and 255)
    _, binary = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)

    # Define a rectangular kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)

    # Perform morphological closing to complete broken lines
    processed_image = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    return binary, processed_image

# Path to the input image
image_path = binary_processed_visual

# Apply the function to complete broken lines
binary_original, completed_image = complete_broken_line(image_path, kernel_size=(2,2))

# Display the results
plt.figure(figsize=(12, 6))

# Original binary image
plt.subplot(1, 2, 1)
plt.imshow(binary_original, cmap='gray')
plt.axis('off')
plt.title("Original Binary Image")

# Image with completed lines
plt.subplot(1, 2, 2)
plt.imshow(completed_image, cmap='gray')
plt.axis('off')
plt.title("Completed Lines Image")

plt.tight_layout()
plt.show()
