import os
import time
import numpy as np
import onnx
import onnxruntime as ort
import onnx_graphsurgeon as gs
from PIL import Image

# 🚀 Increase PIL Max Image Size (Prevents Decompression Bomb Errors)
Image.MAX_IMAGE_PIXELS = None

# **System Info**
print("ONNX Runtime Version:", ort.__version__)

# **Paths and Parameters**
IMG_PATH = "/home/aastha/Downloads/rooftop_segmentation/small_crop1.tif"
WINDOW_SIZE = 1024
STEP_SIZE = 1024
BATCH_SIZE = 16  # 🚀 Increased batch size for faster inference
ORIGINAL_ONNX_MODEL_PATH = "Final_model.onnx"
FIXED_ONNX_MODEL_PATH = "Final_model_fixed.onnx"
OUTPUT_PATH_ONNX = "/home/aastha/Downloads/rooftop_segmentation/sliding_window_onnx.tif"
os.makedirs(os.path.dirname(OUTPUT_PATH_ONNX), exist_ok=True)

# **Step 1: Fix ConvTranspose (Enable Full GPU Execution)**
def fix_onnx_model(input_onnx_path, output_onnx_path):
    print("🔧 Fixing ONNX Model for Full GPU Optimization...")
    model = onnx.load(input_onnx_path)
    graph = gs.import_onnx(model)
    
    # Always fix (remove any previously fixed model)
    for node in list(graph.nodes):
        if node.op == "ConvTranspose":
            # Check if the node has a 'pads' attribute
            pads = node.attrs.get("pads", None)
            if pads is not None:
                half = len(pads) // 2
                left_pads = pads[:half]
                right_pads = pads[half:]
                # Only fix if the padding is asymmetric
                if left_pads != right_pads:
                    print(f"⚠️ Fixing ConvTranspose Node: {node.name} with asymmetric pads {pads}")

                    # Derive scale factor from strides (if available), defaulting to [2, 2]
                    strides = node.attrs.get("strides", [2, 2])
                    # For 4D input [N, C, H, W], the scales are: [1.0, 1.0, scale_h, scale_w]
                    scale = np.array([1.0, 1.0] + strides, dtype=np.float32)

                    # Create a constant node for the scales and get its output variable
                    scales_const = gs.Constant(name=node.name + "_scales", values=scale)
                    scales_var = scales_const.outputs[0]

                    # Create an empty ROI constant and get its output variable
                    empty_roi = gs.Constant(name=node.name + "_empty_roi", values=np.array([], dtype=np.float32))
                    empty_roi_var = empty_roi.outputs[0]

                    # Create a Variable for the output of the Resize node instead of using a string.
                    resize_out_var = gs.Variable(name=node.name + "_resize_out", dtype=node.inputs[0].dtype, shape=None)

                    # Create a Resize node with proper inputs (all as Variables)
                    resize_node = gs.Node(
                        op="Resize",
                        inputs=[node.inputs[0], empty_roi_var, scales_var],
                        outputs=[resize_out_var],
                        name=node.name + "_resize"
                    )

                    # Remove padding-specific attributes from the original node
                    if "pads" in node.attrs:
                        del node.attrs["pads"]
                    if "output_padding" in node.attrs:
                        del node.attrs["output_padding"]

                    # Change the original node's op from ConvTranspose to Conv.
                    # Now the upsampling is done by the Resize node, so the Conv takes the resized tensor as input.
                    node.op = "Conv"
                    node.inputs[0] = resize_out_var

                    # Append the new constant and Resize nodes to the graph
                    graph.nodes.append(scales_const)
                    graph.nodes.append(empty_roi)
                    graph.nodes.append(resize_node)
                else:
                    print(f"Node {node.name} uses symmetric padding. No fix needed.")
            else:
                print(f"Node {node.name} has no 'pads' attribute; skipping fix.")

    # Export and save the fixed model
    optimized_model = gs.export_onnx(graph)
    onnx.save(optimized_model, output_onnx_path)
    print(f"✅ Fixed ONNX Model Saved: {output_onnx_path}")

# **Always run the fix so that you’re sure the model is modified**
fix_onnx_model(ORIGINAL_ONNX_MODEL_PATH, FIXED_ONNX_MODEL_PATH)

# **Step 2: Load ONNX Model with CUDA**
def load_onnx_model(onnx_path):
    print("🚀 Initializing ONNX Runtime with CUDA...")
    providers = [
        ('CUDAExecutionProvider', {'cudnn_conv_algo_search': 'HEURISTIC'}),
        'CPUExecutionProvider'
    ]
    ort_session = ort.InferenceSession(onnx_path, providers=providers)
    input_name = ort_session.get_inputs()[0].name
    print(f"✅ ONNX Model Input Name: {input_name}")
    return ort_session, input_name

# **Load the fixed ONNX model**
ort_session, input_name = load_onnx_model(FIXED_ONNX_MODEL_PATH)

# **Step 3: Efficient Inference with Sliding Window**
def process_and_save_predictions(img_path, ort_session, input_name, window_size, step_size, output_path, batch_size):
    print("\n🏎️ Running High-Speed ONNX Inference on GPU...")
    img = Image.open(img_path)
    width, height = img.size
    print(f"📷 Input Image Size: Width={width}, Height={height}")

    output_mask = np.zeros((height, width), dtype=np.float32)
    patches, positions = [], []

    # Compute sliding window positions
    x_steps = list(range(0, width - window_size + 1, step_size))
    y_steps = list(range(0, height - window_size + 1, step_size))

    if x_steps[-1] + window_size < width:
        x_steps.append(width - window_size)
    if y_steps[-1] + window_size < height:
        y_steps.append(height - window_size)

    start_time = time.time()

    for y in y_steps:
        for x in x_steps:
            box = (x, y, x + window_size, y + window_size)
            patch = img.crop(box)
            patch_array = np.array(patch) / 255.0

            # Ensure 3-channel input
            if patch_array.ndim == 2:
                patch_array = np.stack([patch_array] * 3, axis=-1)
            elif patch_array.shape[2] == 1:
                patch_array = np.concatenate([patch_array] * 3, axis=-1)
            patch_array = patch_array[:, :, :3]

            patches.append(patch_array)
            positions.append((x, y))

            if len(patches) == batch_size:
                patch_batch = np.array(patches, dtype=np.float32)
                preds = ort_session.run(None, {input_name: patch_batch})[0]

                for i, pred in enumerate(preds):
                    pred_mask = pred[:, :, 0]  # assuming single-channel mask output
                    x_pos, y_pos = positions[i]
                    output_mask[y_pos:y_pos+window_size, x_pos:x_pos+window_size] = pred_mask

                patches, positions = [], []

    # Process any remaining patches
    if patches:
        patch_batch = np.array(patches, dtype=np.float32)
        preds = ort_session.run(None, {input_name: patch_batch})[0]

        for i, pred in enumerate(preds):
            pred_mask = pred[:, :, 0]
            x_pos, y_pos = positions[i]
            output_mask[y_pos:y_pos+window_size, x_pos:x_pos+window_size] = pred_mask

    # Convert mask to uint8 and save
    output_mask_uint8 = (output_mask * 255).astype(np.uint8)
    output_image = Image.fromarray(output_mask_uint8)
    output_image.save(output_path)

    end_time = time.time()
    print(f"✅ Segmentation Output Saved to: {output_path}")
    print(f"⏳ ONNX Inference Time: {end_time - start_time:.2f} seconds")

# **Run Inference**
process_and_save_predictions(
    IMG_PATH, ort_session, input_name, WINDOW_SIZE, STEP_SIZE,
    OUTPUT_PATH_ONNX, BATCH_SIZE
)
