import cv2
import numpy as np
import vtracer as vt

def extract_largest_feature(image_path, output_path):
    # Load the image in grayscale mode
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Ensure the image is binary (0 and 255)
    _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    
    # Invert the binary image so that features are white and the background is black
    inverted_binary_image = cv2.bitwise_not(binary_image)
    
    # Apply morphological closing to connect dotted and dashed lines
    kernel = np.ones((3, 3), np.uint8)  # 3x3 kernel to close small gaps
    closed_image = cv2.morphologyEx(inverted_binary_image, cv2.MORPH_CLOSE, kernel)
    
    # Apply dilation to ensure all parts of the feature are connected
    dilated_image = cv2.dilate(closed_image, kernel, iterations=4)
    
    # Find all contours in the dilated image
    contours, _ = cv2.findContours(dilated_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    # If no contours found, return an empty image
    if not contours:
        empty_image = np.zeros_like(image)
        cv2.imwrite(output_path, empty_image)
        return

    # Find the largest contour based on contour area
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Create a mask for the largest feature
    mask = np.zeros_like(image)
    cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
    
    # Extract the largest feature using the mask
    largest_feature = cv2.bitwise_and(image, image, mask=mask)
    
    # Save or display the result
    cv2.imwrite(output_path, cv2.bitwise_not(largest_feature))
    vt.convert_image_to_svg_py(
                               output_path, 
                               "output.svg", 
                               colormode="binary",
                               hierarchical="cutout",
                               mode="polygon",
                               filter_speckle=32,
                           )


# Example usage:
extract_largest_feature('input.tif', 'output_image.png')
