import os
import numpy as np
import tensorflow as tf
from PIL import Image
import PIL

# Increase Pillow's safety limit for maximum image pixels
PIL.Image.MAX_IMAGE_PIXELS = None

# Parameters
img_path = r'C:\Users\FS-AI\Downloads\output_3gb.tif'  # Path to your input TIFF image
window_size = 224  # Size of the sliding window
step_size = 224    # Step size for the sliding window
output_path = r'C:\Users\FS-AI\Downloads\segmentation_output.tif'  # Path to save the output image
batch_size = 12  # Number of patches to process in a batch

def process_and_save_predictions(img_path, model, window_size=224, step_size=224, output_path='output.tif', batch_size=12):
    img = Image.open(img_path)
    width, height = img.size

    # Initialize an empty array to hold the predictions
    output_mask = np.zeros((height, width), dtype=np.float32)

    patches = []
    positions = []

    # Calculate the number of steps in x and y directions
    x_steps = list(range(0, width - window_size + 1, step_size))
    y_steps = list(range(0, height - window_size + 1, step_size))

    # Add the last step if it's not already included
    if x_steps[-1] + window_size < width:
        x_steps.append(width - window_size)
    if y_steps[-1] + window_size < height:
        y_steps.append(height - window_size)

    # Iterate over the image with a sliding window
    print(y_steps)
    for y in y_steps:
        print(y)
        for x in x_steps:
            # Extract patch
            box = (x, y, x + window_size, y + window_size)
            patch = img.crop(box)

            # Convert to numpy array and normalize
            patch_array = np.array(patch) / 255.0

            # Ensure the patch has 3 channels
            if patch_array.ndim == 2:
                patch_array = np.stack([patch_array]*3, axis=-1)
            elif patch_array.shape[2] == 1:
                patch_array = np.concatenate([patch_array]*3, axis=-1)

            # Append to patches list
            patches.append(patch_array)
            positions.append((x, y))

            # Process batch
            if len(patches) == batch_size:
                patch_batch = np.array(patches)
                preds = model.predict(patch_batch)
                for i, pred in enumerate(preds):
                    pred_mask = tf.math.sigmoid(pred[:, :, 0]).numpy()
                    x_pos, y_pos = positions[i]
                    output_mask[y_pos:y_pos+window_size, x_pos:x_pos+window_size] = pred_mask
                patches = []
                positions = []

    # Process any remaining patches
    if patches:
        patch_batch = np.array(patches)
        preds = model.predict(patch_batch)
        for i, pred in enumerate(preds):
            pred_mask = tf.math.sigmoid(pred[:, :, 0]).numpy()
            x_pos, y_pos = positions[i]
            output_mask[y_pos:y_pos+window_size, x_pos:x_pos+window_size] = pred_mask
        patches = []
        positions = []

    # Save the output_mask as a TIFF image
    output_mask_uint8 = (output_mask * 255).astype(np.uint8)
    output_image = Image.fromarray(output_mask_uint8)
    output_image.save(output_path)

    print(f"Segmentation output saved to {output_path}")

# Load your trained model (ensure that 'loaded_model' is your actual model)
# For example:
# loaded_model = tf.keras.models.load_model('path_to_your_model.h5')

# Call the function with your model
process_and_save_predictions(img_path, loaded_model, window_size=window_size, step_size=step_size, output_path=output_path, batch_size=batch_size)
