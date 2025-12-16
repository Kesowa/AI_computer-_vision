import os
import sys
import time
import numpy as np
import onnxruntime as ort
import rasterio
from rasterio.windows import Window

# ------------------------------
# PARAMETERS
# ------------------------------
IMG_PATH = r'/home/debian/Downloads/rasters/592561_Ortho_Vakadu_COG.tif'
WINDOW_SIZE = 512
STEP_SIZE = 512
BATCH_SIZE = 1
OUTPUT_PATH = r'/home/debian/Downloads/rasters/592561_Ortho_Vakadu_COG_output.tif'

# Use the rewritten ONNX model (without ConvTranspose)
ONNX_MODEL_PATH = "/home/debian/Downloads/tree_detection/model_RS.onnx"

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

# ------------------------------
# RUN ONNX INFERENCE (GPU) - RAM OPTIMIZED
# ------------------------------
def run_onnx_inference_optimized(onnx_path, img_path, output_path, window_size, step_size, batch_size):
    print(f"\n🔹 Running RAM-optimized inference on {onnx_path}")

    # 1) Session Options for memory optimization
    sess_options = ort.SessionOptions()
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    sess_options.enable_mem_pattern = False
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED

    # 2) Force GPU only
    providers = ["CUDAExecutionProvider"]
    session = ort.InferenceSession(onnx_path, sess_options, providers=providers)
    actual_providers = session.get_providers()
    print(f"✅ ONNX Runtime using providers: {actual_providers}")

    if "CUDAExecutionProvider" not in actual_providers:
        print("⚠️ WARNING: GPU not available, inference will be slow!")

    # 3) Input/Output names
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # 4) Open input and output rasters
    with rasterio.open(img_path) as src:
        width = src.width
        height = src.height
        print(f"📌 Input image size: Width={width}, Height={height}")

        # Create output profile
        profile = {
            'driver': 'GTiff',
            'dtype': 'uint8',
            'count': 1,
            'width': width,
            'height': height,
            'crs': src.crs,
            'transform': src.transform,
            'compress': 'lzw',
            'tiled': True,
            'blockxsize': 256,
            'blockysize': 256,
            'photometric': 'MINISBLACK'
        }

        with rasterio.open(output_path, 'w', **profile) as dst:
            start_time = time.time()
            total_patches = 0

            # Calculate window positions
            x_steps = list(range(0, width - window_size + 1, step_size))
            y_steps = list(range(0, height - window_size + 1, step_size))

            # Ensure edge coverage
            if x_steps[-1] + window_size < width:
                x_steps.append(width - window_size)
            if y_steps[-1] + window_size < height:
                y_steps.append(height - window_size)

            batch_patches = []
            batch_positions = []

            # Process in sliding window fashion
            for y in y_steps:
                for x in x_steps:
                    # Read window directly from disk
                    window = Window(x, y, window_size, window_size)
                    patch = src.read(window=window)

                    # Convert to (H, W, C) format
                    if patch.shape[0] == 1:
                        patch_array = np.stack([patch[0]] * 3, axis=-1)
                    elif patch.shape[0] >= 3:
                        patch_array = np.transpose(patch[:3], (1, 2, 0))
                    else:
                        patch_array = np.transpose(patch, (1, 2, 0))
                        patch_array = np.concatenate([patch_array] * 3, axis=-1)

                    # Normalize to [0, 1]
                    patch_array = patch_array.astype(np.float32) / 255.0

                    batch_patches.append(patch_array)
                    batch_positions.append((x, y))

                    # Process when batch is full
                    if len(batch_patches) == batch_size:
                        batch_array = np.array(batch_patches)

                        # Run ONNX inference
                        preds = session.run([output_name], {input_name: batch_array})[0]

                        # Apply Sigmoid if needed (depends on model output)
                        # If model already has sigmoid, skip this
                        # preds = 1.0 / (1.0 + np.exp(-preds))

                        # Write predictions to output
                        for i in range(len(batch_patches)):
                            pred_mask = preds[i, :, :, 0]
                            x_pos, y_pos = batch_positions[i]
                            pred_mask_uint8 = (pred_mask * 255).astype(np.uint8)
                            window_out = Window(x_pos, y_pos, window_size, window_size)
                            dst.write(pred_mask_uint8, 1, window=window_out)

                        total_patches += len(batch_patches)
                        batch_patches = []
                        batch_positions = []

                        if total_patches % 100 == 0:
                            elapsed = time.time() - start_time
                            print(f"🔄 Processed {total_patches} patches in {elapsed:.2f}s")

            # Process remaining patches
            if batch_patches:
                batch_array = np.array(batch_patches)
                preds = session.run([output_name], {input_name: batch_array})[0]
                # preds = 1.0 / (1.0 + np.exp(-preds))

                for i in range(len(batch_patches)):
                    pred_mask = preds[i, :, :, 0]
                    x_pos, y_pos = batch_positions[i]
                    pred_mask_uint8 = (pred_mask * 255).astype(np.uint8)
                    window_out = Window(x_pos, y_pos, window_size, window_size)
                    dst.write(pred_mask_uint8, 1, window=window_out)

                total_patches += len(batch_patches)

            total_time = time.time() - start_time
            print(f"\n✅ Segmentation output saved => {output_path}")
            print(f"⏱️  Total ONNX inference time: {total_time:.2f} seconds")
            print(f"📊 Total patches processed: {total_patches}")

# ------------------------------
# MAIN
# ------------------------------
if __name__ == "__main__":
    # Check if ONNX model exists
    if not os.path.exists(ONNX_MODEL_PATH):
        print(f"❌ ERROR: ONNX model not found at {ONNX_MODEL_PATH}")
        print("\n📋 Available ONNX files:")
        print("  1. model_halo.onnx - Original export (may have ConvTranspose issues)")
        print("  2. model_halo_rewritten.onnx - Rewritten without ConvTranspose (RECOMMENDED)")
        print("\n💡 Share 'model_halo_rewritten.onnx' with others for best compatibility")
        sys.exit(1)

    run_onnx_inference_optimized(
        ONNX_MODEL_PATH,
        IMG_PATH,
        OUTPUT_PATH,
        WINDOW_SIZE,
        STEP_SIZE,
        BATCH_SIZE
    )

    print("\n" + "="*80)
    print("✅ INFERENCE COMPLETE!")
    print("="*80)
    print(f"📁 Output file: {OUTPUT_PATH}")
    print(f"🎯 ONNX model used: {ONNX_MODEL_PATH}")
    print("\n📤 To share with others:")
    print(f"   - ONNX Model: model_halo_rewritten.onnx")
    print(f"   - This script: {os.path.basename(__file__)}")
    print("="*80)
