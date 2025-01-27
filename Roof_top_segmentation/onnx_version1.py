# import os
# import numpy as np
# import tensorflow as tf
# from PIL import Image
# import PIL

# # Increase Pillow's safety limit for maximum image pixels
# PIL.Image.MAX_IMAGE_PIXELS = None

# # Parameters
# img_path = r'/home/aastha/Downloads/rooftop_segmentation/small_crop1.tif'  # Path to your input TIFF image
# window_size = 1024  # Size of the sliding window
# step_size = 1024    # Step size for the sliding window
# output_path = os.getcwd()  # Path to save the output image
# batch_size = 12  # Number of patches to process in a batch

# def process_and_save_predictions(img_path, model, window_size=224, step_size=224, output_path='output.tif', batch_size=12):
#     img = Image.open(img_path)
#     width, height = img.size

#     # Initialize an empty array to hold the predictions
#     output_mask = np.zeros((height, width), dtype=np.float32)

#     patches = []
#     positions = []

#     # Calculate the number of steps in x and y directions
#     x_steps = list(range(0, width - window_size + 1, step_size))
#     y_steps = list(range(0, height - window_size + 1, step_size))

#     # Add the last step if it's not already included
#     if x_steps[-1] + window_size < width:
#         x_steps.append(width - window_size)
#     if y_steps[-1] + window_size < height:
#         y_steps.append(height - window_size)

#     # Iterate over the image with a sliding window
#     print(y_steps)
#     for y in y_steps:
#         print(y)
#         for x in x_steps:
#             # Extract patch
#             box = (x, y, x + window_size, y + window_size)
#             patch = img.crop(box)

#             # Convert to numpy array and normalize
#             patch_array = np.array(patch) / 255.0

#             # Ensure the patch has 3 channels
#             if patch_array.ndim == 2:
#                 patch_array = np.stack([patch_array]*3, axis=-1)
#             elif patch_array.shape[2] == 1:
#                 patch_array = np.concatenate([patch_array]*3, axis=-1)
#             patch_array = patch_array[:, :, :3]
#             # Append to patches list
#             patches.append(patch_array)
#             positions.append((x, y))

#             # Process batch
#             if len(patches) == batch_size:
#                 patch_batch = np.array(patches)
#                 preds = model.predict(patch_batch)
#                 for i, pred in enumerate(preds):
#                     pred_mask = tf.math.sigmoid(pred[:, :, 0]).numpy()
#                     x_pos, y_pos = positions[i]
#                     output_mask[y_pos:y_pos+window_size, x_pos:x_pos+window_size] = pred_mask
#                 patches = []
#                 positions = []
#             # break
#         # break

#     # Process any remaining patches
#     if patches:
#         patch_batch = np.array(patches)
#         preds = model.predict(patch_batch)
#         for i, pred in enumerate(preds):
#             pred_mask = tf.math.sigmoid(pred[:, :, 0]).numpy()
#             x_pos, y_pos = positions[i]
#             output_mask[y_pos:y_pos+window_size, x_pos:x_pos+window_size] = pred_mask
#         patches = []
#         positions = []

#     # Save the output_mask as a TIFF image
#     output_mask_uint8 = (output_mask * 255).astype(np.uint8)
#     output_image = Image.fromarray(output_mask_uint8)
#     output_image.save(output_path)

#     print(f"Segmentation output saved to {output_path}")

# # Load your trained model (ensure that 'loaded_model' is your actual model)
# # For example:
# # loaded_model = tf.keras.models.load_model('Final_model.h5')
# from unet import get_model  # Make sure to import your `get_model` function from `unet.py`

# # Define parameters
# input_shape = (1024, 1024, 3)  # Adjust according to your image size and channels
# output_channels = 1          # For binary segmentation, use 1; for multi-class, adjust accordingly
# output_act = 'sigmoid'       # Use 'sigmoid' for binary segmentation or 'softmax' for multi-class segmentation

# # Get the model
# model = get_model(input_shape, output_channels, output_act)
# # Load the model's weights
# model.load_weights('Final_model.h5')  # Load the weights only



# import os
# # Call the function with your model
# process_and_save_predictions(img_path, model, window_size=window_size, step_size=step_size, output_path=r"/home/aastha/Downloads/rooftop_segmentation/sliding_window_larger_smallcrop1_time.tif", batch_size=batch_size)
# import os
# import numpy as np
# from PIL import Image
# import onnxruntime as ort  # For ONNX inference
# import tf2onnx  # For converting TensorFlow to ONNX

# # Increase Pillow's safety limit for maximum image pixels
# Image.MAX_IMAGE_PIXELS = None

# # Parameters
# img_path = r'/home/aastha/Downloads/rooftop_segmentation/small_crop1.tif'  # Path to your input TIFF image
# window_size = 1024  # Size of the sliding window
# step_size = 1024    # Step size for the sliding window
# output_path = os.getcwd()  # Path to save the output image
# batch_size = 12  # Number of patches to process in a batch

# # Helper function to inspect ONNX model input names
# def get_onnx_input_name(onnx_model_path):
#     ort_session = ort.InferenceSession(onnx_model_path)
#     input_name = ort_session.get_inputs()[0].name  # Get the first input name
#     print(f"ONNX Model Input Name: {input_name}")
#     return input_name

# # Function to process predictions with ONNX model
# def process_and_save_predictions_with_onnx(img_path, onnx_model_path, window_size=224, step_size=224, output_path='output.tif', batch_size=12):
#     img = Image.open(img_path)
#     width, height = img.size

#     # Initialize an empty array to hold the predictions
#     output_mask = np.zeros((height, width), dtype=np.float32)

#     patches = []
#     positions = []

#     # Calculate the number of steps in x and y directions
#     x_steps = list(range(0, width - window_size + 1, step_size))
#     y_steps = list(range(0, height - window_size + 1, step_size))

#     # Add the last step if it's not already included
#     if x_steps[-1] + window_size < width:
#         x_steps.append(width - window_size)
#     if y_steps[-1] + window_size < height:
#         y_steps.append(height - window_size)

#     # Create ONNX Runtime inference session
#     ort_session = ort.InferenceSession(onnx_model_path)
#     input_name = get_onnx_input_name(onnx_model_path)  # Get the correct input name

#     # Iterate over the image with a sliding window
#     for y in y_steps:
#         for x in x_steps:
#             # Extract patch
#             box = (x, y, x + window_size, y + window_size)
#             patch = img.crop(box)

#             # Convert to numpy array and normalize
#             patch_array = np.array(patch) / 255.0

#             # Ensure the patch has 3 channels
#             if patch_array.ndim == 2:
#                 patch_array = np.stack([patch_array]*3, axis=-1)
#             elif patch_array.shape[2] == 1:
#                 patch_array = np.concatenate([patch_array]*3, axis=-1)
#             patch_array = patch_array[:, :, :3]
#             # Append to patches list
#             patches.append(patch_array)
#             positions.append((x, y))

#             # Process batch
#             if len(patches) == batch_size:
#                 patch_batch = np.array(patches, dtype=np.float32)
#                 # Perform ONNX inference
#                 preds = ort_session.run(None, {input_name: patch_batch})[0]
#                 for i, pred in enumerate(preds):
#                     pred_mask = pred[:, :, 0]
#                     x_pos, y_pos = positions[i]
#                     output_mask[y_pos:y_pos+window_size, x_pos:x_pos+window_size] = pred_mask
#                 patches = []
#                 positions = []

#     # Process any remaining patches
#     if patches:
#         patch_batch = np.array(patches, dtype=np.float32)
#         preds = ort_session.run(None, {input_name: patch_batch})[0]
#         for i, pred in enumerate(preds):
#             pred_mask = pred[:, :, 0]
#             x_pos, y_pos = positions[i]
#             output_mask[y_pos:y_pos+window_size, x_pos:x_pos+window_size] = pred_mask
#         patches = []
#         positions = []

#     # Save the output_mask as a TIFF image
#     output_mask_uint8 = (output_mask * 255).astype(np.uint8)
#     output_image = Image.fromarray(output_mask_uint8)
#     output_image.save(output_path)

#     print(f"Segmentation output saved to {output_path}")

# # Call the function with your ONNX model
# onnx_model_path = "Final_model.onnx"
# process_and_save_predictions_with_onnx(img_path, onnx_model_path, window_size=window_size, step_size=step_size, output_path=r"/home/aastha/Downloads/rooftop_segmentation/sliding_window_larger_smallcrop1_time_onnx.tif", batch_size=batch_size)
# import os
# import time
# import numpy as np
# import tensorflow as tf
# from PIL import Image
# import onnxruntime as ort  # For ONNX inference

# # Increase Pillow's safety limit for maximum image pixels
# Image.MAX_IMAGE_PIXELS = None

# # Parameters
# img_path = r'/home/aastha/Downloads/rooftop_segmentation/small_crop1.tif'
# window_size = 1024
# step_size = 1024
# output_path = os.getcwd()
# batch_size = 12

# # Helper function to inspect ONNX model input names
# def get_onnx_input_name(onnx_model_path):
#     ort_session = ort.InferenceSession(onnx_model_path)
#     input_name = ort_session.get_inputs()[0].name
#     print(f"ONNX Model Input Name: {input_name}")
#     return input_name

# # General function to process image with sliding window
# def process_and_save_predictions(img_path, model, window_size, step_size, output_path, batch_size, mode='tensorflow'):
#     img = Image.open(img_path)
#     width, height = img.size
#     print(f"Input image size: Width={width}, Height={height}")

#     output_mask = np.zeros((height, width), dtype=np.float32)
#     patches = []
#     positions = []

#     # Sliding window coordinates
#     x_steps = list(range(0, width - window_size + 1, step_size))
#     y_steps = list(range(0, height - window_size + 1, step_size))

#     if x_steps[-1] + window_size < width:
#         x_steps.append(width - window_size)
#     if y_steps[-1] + window_size < height:
#         y_steps.append(height - window_size)

#     if mode == 'onnx':
#         ort_session = ort.InferenceSession(model)
#         input_name = get_onnx_input_name(model)

#     for y in y_steps:
#         for x in x_steps:
#             box = (x, y, x + window_size, y + window_size)
#             patch = img.crop(box)
#             patch_array = np.array(patch) / 255.0

#             if patch_array.ndim == 2:
#                 patch_array = np.stack([patch_array]*3, axis=-1)
#             elif patch_array.shape[2] == 1:
#                 patch_array = np.concatenate([patch_array]*3, axis=-1)
#             patch_array = patch_array[:, :, :3]

#             patches.append(patch_array)
#             positions.append((x, y))

#             if len(patches) == batch_size:
#                 patch_batch = np.array(patches, dtype=np.float32)
#                 if mode == 'tensorflow':
#                     preds = model.predict(patch_batch)
#                 elif mode == 'onnx':
#                     preds = ort_session.run(None, {input_name: patch_batch})[0]

#                 for i, pred in enumerate(preds):
#                     pred_mask = pred[:, :, 0] if mode == 'onnx' else tf.math.sigmoid(pred[:, :, 0]).numpy()
#                     x_pos, y_pos = positions[i]
#                     output_mask[y_pos:y_pos+window_size, x_pos:x_pos+window_size] = pred_mask

#                 patches = []
#                 positions = []

#     if patches:
#         patch_batch = np.array(patches, dtype=np.float32)
#         if mode == 'tensorflow':
#             preds = model.predict(patch_batch)
#         elif mode == 'onnx':
#             preds = ort_session.run(None, {input_name: patch_batch})[0]

#         for i, pred in enumerate(preds):
#             pred_mask = pred[:, :, 0] if mode == 'onnx' else tf.math.sigmoid(pred[:, :, 0]).numpy()
#             x_pos, y_pos = positions[i]
#             output_mask[y_pos:y_pos+window_size, x_pos:x_pos+window_size] = pred_mask

#     output_mask_uint8 = (output_mask * 255).astype(np.uint8)
#     output_image = Image.fromarray(output_mask_uint8)
#     output_image.save(output_path)
#     print(f"Segmentation output saved to {output_path}")

# # TensorFlow Inference
# from unet import get_model  # Make sure to import your `get_model` function from `unet.py`

# input_shape = (1024, 1024, 3)
# output_channels = 1
# output_act = 'sigmoid'
# model = get_model(input_shape, output_channels, output_act)
# model.load_weights('Final_model.h5')

# print("\nRunning TensorFlow Inference...")
# start_time_tf = time.time()
# process_and_save_predictions(
#     img_path, model, window_size, step_size,
#     r"/home/aastha/Downloads/rooftop_segmentation/sliding_window_tf.tif", batch_size, mode='tensorflow'
# )
# end_time_tf = time.time()
# print(f"TensorFlow inference time: {end_time_tf - start_time_tf:.2f} seconds")

# # ONNX Inference
# onnx_model_path = "Final_model.onnx"
# print("\nRunning ONNX Inference...")
# start_time_onnx = time.time()
# process_and_save_predictions(
#     img_path, onnx_model_path, window_size, step_size,
#     r"/home/aastha/Downloads/rooftop_segmentation/sliding_window_onnx.tif", batch_size, mode='onnx'
# )
# end_time_onnx = time.time()
# print(f"ONNX inference time: {end_time_onnx - start_time_onnx:.2f} seconds")

# # Print time comparison
# print("\n--- Inference Time Comparison ---")
# print(f"TensorFlow: {end_time_tf - start_time_tf:.2f} seconds")
# print(f"ONNX: {end_time_onnx - start_time_onnx:.2f} seconds")
import os
import sys
import time
import numpy as np
import tensorflow as tf
from PIL import Image
import onnxruntime as ort  # For ONNX inference
import tensorflow as tf
print("TensorFlow Version:", tf.__version__)
print("GPUs Available:", tf.config.list_physical_devices('GPU'))

# Increase Pillow's safety limit for maximum image pixels
Image.MAX_IMAGE_PIXELS = None

# Parameters
img_path = r'/home/aastha/Downloads/rooftop_segmentation/small_crop1.tif'
window_size = 1024
step_size = 1024
batch_size = 12

# Output paths
output_path_tf = r"/home/aastha/Downloads/rooftop_segmentation/sliding_window_tf.tif"
output_path_onnx = r"/home/aastha/Downloads/rooftop_segmentation/sliding_window_onnx.tif"

# Ensure output directories exist
os.makedirs(os.path.dirname(output_path_tf), exist_ok=True)
os.makedirs(os.path.dirname(output_path_onnx), exist_ok=True)

# Helper function to inspect ONNX model input names
def get_onnx_input_name(onnx_model_path):
    ort_session = ort.InferenceSession(onnx_model_path, providers=['CUDAExecutionProvider'])
    input_name = ort_session.get_inputs()[0].name
    print(f"ONNX Model Input Name: {input_name}")
    return input_name

# General function to process image with sliding window
def process_and_save_predictions(img_path, model, window_size, step_size, output_path, batch_size, mode='tensorflow', ort_session=None, input_name=None):
    img = Image.open(img_path)
    width, height = img.size
    print(f"Input image size: Width={width}, Height={height}")

    output_mask = np.zeros((height, width), dtype=np.float32)
    patches = []
    positions = []

    # Sliding window coordinates
    x_steps = list(range(0, width - window_size + 1, step_size))
    y_steps = list(range(0, height - window_size + 1, step_size))

    if x_steps[-1] + window_size < width:
        x_steps.append(width - window_size)
    if y_steps[-1] + window_size < height:
        y_steps.append(height - window_size)

    for y in y_steps:
        for x in x_steps:
            box = (x, y, x + window_size, y + window_size)
            patch = img.crop(box)
            patch_array = np.array(patch) / 255.0

            if patch_array.ndim == 2:
                patch_array = np.stack([patch_array]*3, axis=-1)
            elif patch_array.shape[2] == 1:
                patch_array = np.concatenate([patch_array]*3, axis=-1)
            patch_array = patch_array[:, :, :3]

            patches.append(patch_array)
            positions.append((x, y))

            if len(patches) == batch_size:
                patch_batch = np.array(patches, dtype=np.float32)
                if mode == 'tensorflow':
                    preds = model.predict(patch_batch)
                elif mode == 'onnx':
                    preds = ort_session.run(None, {input_name: patch_batch})[0]

                for i, pred in enumerate(preds):
                    if mode == 'onnx':
                        pred_mask = pred[:, :, 0]
                    else:
                        pred_mask = tf.math.sigmoid(pred[:, :, 0]).numpy()
                    x_pos, y_pos = positions[i]
                    output_mask[y_pos:y_pos+window_size, x_pos:x_pos+window_size] = pred_mask

                patches = []
                positions = []

    # Process any remaining patches
    if patches:
        patch_batch = np.array(patches, dtype=np.float32)
        if mode == 'tensorflow':
            preds = model.predict(patch_batch)
        elif mode == 'onnx':
            preds = ort_session.run(None, {input_name: patch_batch})[0]

        for i, pred in enumerate(preds):
            if mode == 'onnx':
                pred_mask = pred[:, :, 0]
            else:
                pred_mask = tf.math.sigmoid(pred[:, :, 0]).numpy()
            x_pos, y_pos = positions[i]
            output_mask[y_pos:y_pos+window_size, x_pos:x_pos+window_size] = pred_mask

    # Convert mask to uint8 and save
    output_mask_uint8 = (output_mask * 255).astype(np.uint8)
    output_image = Image.fromarray(output_mask_uint8)
    output_image.save(output_path)
    print(f"Segmentation output saved to {output_path}")

# ----------------------------------
# TensorFlow GPU Configuration
# ----------------------------------
gpus = tf.config.list_physical_devices('GPU')
if not gpus:
    print("ERROR: No GPU found for TensorFlow. Exiting.", file=sys.stderr)
    sys.exit(1)

try:
    # Set memory growth to prevent TensorFlow from allocating all GPU memory at once
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"TensorFlow is using GPU: {gpus}")
except RuntimeError as e:
    print(f"ERROR setting TensorFlow GPU memory growth: {e}", file=sys.stderr)
    sys.exit(1)

# TensorFlow Inference
from unet import get_model  # Make sure to import your `get_model` function from `unet.py`

input_shape = (window_size, window_size, 3)
output_channels = 1
output_act = 'sigmoid'
model_tf = get_model(input_shape, output_channels, output_act)
model_tf.load_weights('Final_model.h5')

print("\nRunning TensorFlow Inference on GPU...")
start_time_tf = time.time()
process_and_save_predictions(
    img_path, model_tf, window_size, step_size,
    output_path_tf, batch_size, mode='tensorflow'
)
end_time_tf = time.time()
print(f"TensorFlow inference time: {end_time_tf - start_time_tf:.2f} seconds")

# ----------------------------------
# ONNX Runtime GPU Configuration
# ----------------------------------

# Check if CUDA Execution Provider is available
available_providers = ort.get_available_providers()
if 'CUDAExecutionProvider' not in available_providers:
    print("ERROR: CUDA Execution Provider not available in ONNX Runtime. Exiting.", file=sys.stderr)
    sys.exit(1)

providers_onnx = ['CUDAExecutionProvider']
print("ONNX Runtime will use CUDA Execution Provider for GPU inference.")

onnx_model_path = "Final_model.onnx"

# Initialize ONNX Runtime session with GPU support
try:
    ort_session = ort.InferenceSession(onnx_model_path, providers=providers_onnx)
    input_name = get_onnx_input_name(onnx_model_path)
except Exception as e:
    print(f"ERROR initializing ONNX Runtime session: {e}", file=sys.stderr)
    sys.exit(1)

print("\nRunning ONNX Inference on GPU...")
start_time_onnx = time.time()
process_and_save_predictions(
    img_path, onnx_model_path, window_size, step_size,
    output_path_onnx, batch_size, mode='onnx', ort_session=ort_session, input_name=input_name
)
end_time_onnx = time.time()
print(f"ONNX inference time: {end_time_onnx - start_time_onnx:.2f} seconds")

# Print time comparison
print("\n--- Inference Time Comparison ---")
print(f"TensorFlow: {end_time_tf - start_time_tf:.2f} seconds")
print(f"ONNX: {end_time_onnx - start_time_onnx:.2f} seconds")
