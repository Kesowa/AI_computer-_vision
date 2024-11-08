import rasterio
import numpy as np
import matplotlib.pyplot as plt

def inspect_each_band(image_path):
    # Open the TIFF file with rasterio
    with rasterio.open(image_path) as src:
        print("Metadata:", src.meta)
        
        # Iterate through each band and analyze
        for band_id in range(1, src.count + 1):  # Bands are 1-indexed in rasterio
            band = src.read(band_id)
            min_val, max_val = band.min(), band.max()
            print(f"Band {band_id} - Min pixel value: {min_val}, Max pixel value: {max_val}")

            # Plot the histogram for each band
            plt.hist(band.ravel(), bins=256, range=[0, 255])
            plt.title(f"Histogram of Band {band_id}")
            plt.show()

            # Display the band as an image to visually inspect it
            plt.imshow(band, cmap='gray')
            plt.title(f"Band {band_id}")
            plt.colorbar()
            plt.show()

# Example usage:
inspect_each_band('converted_output.tif')
