import os
import sys
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.mixed_precision import set_global_policy
from PIL import Image

# Enable Mixed Precision
set_global_policy('mixed_float16')  # Allows TensorFlow to use float16 where appropriate

# Increase Pillow's safety limit for maximum image pixels
Image.MAX_IMAGE_PIXELS = None

# Parameters
img_path = '/home/aastha/Downloads/rooftop_segmentation/small_crop1.tif'
window_size = 1024
step_size = 1024
batch_size = 8  # Adjust based on GPU memory. Start lower if facing OOM issues
output_path_tf = "/home/aastha/Downloads/rooftop_segmentation/sliding_window_tf.tif"

# Ensure output directory exists
os.makedirs(os.path.dirname(output_path_tf), exist_ok=True)

# TensorFlow GPU Configuration
gpus = tf.config.list_physical_devices('GPU')
if not gpus:
    print("ERROR: No GPU found for TensorFlow. Exiting.", file=sys.stderr)
    sys.exit(1)

try:
    # Limit TensorFlow to use only the first GPU
    tf.config.set_visible_devices(gpus[0], 'GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)  # Allow memory growth
    print(f"TensorFlow is using GPU: {gpus[0]}")
except RuntimeError as e:
    print(f"ERROR setting TensorFlow GPU configuration: {e}", file=sys.stderr)
    sys.exit(1)

# Disable XLA to prevent potential overhead
tf.config.optimizer.set_jit(False)

# Import your model's get_model function
from unet import get_model  # Ensure `unet.py` is in the same directory or adjust the import path

# Load the TensorFlow model
input_shape = (window_size, window_size, 3)
output_channels = 1
output_act = 'sigmoid'
model_tf = get_model(input_shape, output_channels, output_act)
model_tf.load_weights('Final_model.h5')
model_tf.summary()  # Optional: Print model summary

# Warm-up run to initialize GPU kernels
def warm_up(model, batch_size, window_size):
    dummy_input = tf.random.uniform((batch_size, window_size, window_size, 3))
    _ = model.predict(dummy_input)
    print("Warm-up run completed.")

warm_up(model_tf, batch_size, window_size)

# Helper function to create dataset generator
def create_dataset(img, window_size, step_size):
    width, height = img.size
    x_steps = list(range(0, width - window_size + 1, step_size))
    y_steps = list(range(0, height - window_size + 1, step_size))

    # Ensure the last window covers the edge
    if x_steps[-1] + window_size < width:
        x_steps.append(width - window_size)
    if y_steps[-1] + window_size < height:
        y_steps.append(height - window_size)

    for y in y_steps:
        for x in x_steps:
            box = (x, y, x + window_size, y + window_size)
            patch = img.crop(box)
            patch_array = np.array(patch) / 255.0  # Normalize to [0, 1]

            # Handle grayscale images by converting to RGB
            if patch_array.ndim == 2:
                patch_array = np.stack([patch_array] * 3, axis=-1)
            elif patch_array.shape[2] == 1:
                patch_array = np.concatenate([patch_array] * 3, axis=-1)
            patch_array = patch_array[:, :, :3]  # Ensure 3 channels

            yield patch_array.astype(np.float32), np.array([x, y], dtype=np.int32)

# Create TensorFlow Dataset
def prepare_dataset(img_path, window_size, step_size, batch_size):
    img = Image.open(img_path)
    dataset = tf.data.Dataset.from_generator(
        lambda: create_dataset(img, window_size, step_size),
        output_signature=(
            tf.TensorSpec(shape=(window_size, window_size, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(2,), dtype=tf.int32)
        )
    )
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset, img.size

# Process and save predictions
def process_and_save_predictions(img_path, model, window_size, step_size, output_path, batch_size):
    dataset, (width, height) = prepare_dataset(img_path, window_size, step_size, batch_size)
    print(f"Input image size: Width={width}, Height={height}")

    output_mask = np.zeros((height, width), dtype=np.float32)

    start_time = time.time()
    total_batches = 0

    for batch_patches, batch_positions in dataset:
        preds = model.predict(batch_patches, batch_size=batch_size)
        # Apply sigmoid if not already applied in the model's activation
        preds = tf.math.sigmoid(preds).numpy().astype(np.float32)

        for i in range(preds.shape[0]):
            pred_mask = preds[i, :, :, 0]  # Assuming single channel output
            x_pos, y_pos = batch_positions[i].numpy()
            output_mask[y_pos:y_pos+window_size, x_pos:x_pos+window_size] = pred_mask

        total_batches += 1
        if total_batches % 10 == 0:
            elapsed = time.time() - start_time
            print(f"Processed {total_batches * batch_size} patches in {elapsed:.2f} seconds.")

    # Convert mask to uint8 and save
    output_mask_uint8 = (output_mask * 255).astype(np.uint8)
    output_image = Image.fromarray(output_mask_uint8)
    output_image.save(output_path)
    total_time = time.time() - start_time
    print(f"Segmentation output saved to {output_path}")
    print(f"Total TensorFlow inference time: {total_time:.2f} seconds")

# Run TensorFlow Inference
print("\nRunning TensorFlow Inference on GPU...")
process_and_save_predictions(
    img_path, model_tf, window_size, step_size,
    output_path_tf, batch_size
)
