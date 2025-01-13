import numpy as np
import rasterio
from rasterio import Affine
from rasterio.enums import Resampling

def process_tiff(input_file, output_file, threshold):
    """
    Process a TIFF file to set pixels above a threshold to 255 and below it to 0.

    Args:
        input_file (str): Path to the input TIFF file.
        output_file (str): Path to save the processed TIFF file.
        threshold (float): The pixel value threshold.
    """
    # Open the input TIFF file
    with rasterio.open(input_file) as src:
        # Read the image data
        data = src.read(1)  # Read the first band
        
        # Apply thresholding
        processed_data = np.where(data > threshold, 255, 0).astype(np.uint8)

        # Save the updated data to a new TIFF file
        profile = src.profile
        profile.update(dtype=rasterio.uint8, count=1, compress='lzw')

        with rasterio.open(output_file, 'w', **profile) as dst:
            dst.write(processed_data, 1)

# Example usage
input_tiff = "/home/aastha/Downloads/rooftop_segmentation/sliding_window_larger_smallcrop1.tif"  # Replace with your input TIFF file path
output_tiff = "/home/aastha/Downloads/rooftop_segmentation/sliding_window_larger_smallcrop1_thresold150.tif " # Replace with your output TIFF file path
threshold_value = 150  # Replace with your desired threshold value

process_tiff(input_tiff, output_tiff, threshold_value)
