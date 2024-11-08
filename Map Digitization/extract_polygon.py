import cv2
import numpy as np
import vtracer as vt

def extract_polygons_without_text(image_path, output_path):
    # Load the image in grayscale mode
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Ensure the image is binary (0 and 255)
    _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    
    # Invert the binary image so that features are white and the background is black
    inverted_binary_image = cv2.bitwise_not(binary_image)
    
    # Apply morphological operations to clean up the image
    kernel = np.ones((3, 3), np.uint8)
    cleaned_image = cv2.morphologyEx(inverted_binary_image, cv2.MORPH_CLOSE, kernel)
    cleaned_image = cv2.dilate(cleaned_image, kernel, iterations=1)
    
    # Remove small objects (likely text) using connected components
    num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(cleaned_image, connectivity=4)
    
    # Set a minimum area threshold to filter out small components (adjust as needed)
    min_area_threshold = 800000  # You may need to adjust this value based on your image
    
    # Create a mask to keep components larger than the area threshold
    component_mask = np.zeros_like(image)
    for i in range(1, num_labels):  # Skip the background label 0
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area_threshold:
            component_mask[labels_im == i] = 255
    
    # Extract the features using the cleaned mask
    filtered_features = cv2.bitwise_and(image, image, mask=component_mask)
    
    # Save the result
    cv2.imwrite(output_path, cv2.bitwise_not(filtered_features))
    
    # Optional: Convert the output image to SVG using vtracer
    vt.convert_image_to_svg_py(
        output_path, 
        "output3.svg", 
        colormode="binary",
        hierarchical="cutout",
        mode="polygon",
        filter_speckle=8,
    )

# Example usage:
# extract_polygons_without_text('input_image.tif', 'output_image.png')
extract_polygons_without_text(r'C:\Users\FS-AI\3D Objects\02.09.24\1129042_2.tif', 'image_asb1.png')

