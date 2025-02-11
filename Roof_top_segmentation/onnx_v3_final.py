import os
import sys
import time
import numpy as np
from PIL import Image

import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Layer, UpSampling2D, Conv2D, Conv2DTranspose

import tf2onnx
import onnx
import onnxruntime as ort

# Attempt to import onnx_graphsurgeon for rewriting leftover ConvTranspose ops
try:
    import onnx_graphsurgeon as gs
    OGS_AVAILABLE = True
except ImportError:
    OGS_AVAILABLE = False
    print("⚠ onnx_graphsurgeon not found. If leftover ConvTranspose remain, rewriting won't be possible.\n"
          "  pip install --extra-index-url https://pypi.ngc.nvidia.com onnx-graphsurgeon")


# ------------------------------
# 1) PARAMETERS
# ------------------------------
IMG_PATH = "/home/aastha/Downloads/rooftop_segmentation/small_crop1.tif"
WINDOW_SIZE = 1024
STEP_SIZE = 1024
BATCH_SIZE = 1  # Minimize memory usage
MODEL_WEIGHTS = "Final_model.h5"

ONNX_MODEL_PATH = "model_halo.onnx"
ONNX_REWRITTEN_PATH = "model_halo_rewritten.onnx"
OUTPUT_PATH = "/home/aastha/Downloads/rooftop_segmentation/sliding_window_onnx.tif"

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
Image.MAX_IMAGE_PIXELS = None  # Large images

# ------------------------------
# 2) CUSTOM UPSAMPLING LAYER (TF)
# ------------------------------
class UpSamplingConv2D(Layer):
    """
    Replaces Conv2DTranspose with UpSampling2D + Conv2D in TF.
    """
    def __init__(self, filters, kernel_size, strides=(2,2), padding='same', activation=None, **kwargs):
        super().__init__(**kwargs)
        self.upsample = UpSampling2D(size=strides, interpolation='nearest')
        self.conv = Conv2D(filters, kernel_size, padding=padding, activation=activation)

    def call(self, inputs):
        x = self.upsample(inputs)
        x = self.conv(x)
        return x

# ------------------------------
# 3) RECURSIVE REPLACEMENT
# ------------------------------
def replace_all_transpose(layer: tf.keras.layers.Layer) -> tf.keras.layers.Layer:
    # Replace each Conv2DTranspose
    if isinstance(layer, Conv2DTranspose):
        print(f"Replacing layer '{layer.name}' with UpSamplingConv2D")
        return UpSamplingConv2D(
            filters=layer.filters,
            kernel_size=layer.kernel_size,
            strides=layer.strides,
            padding=layer.padding,
            activation=layer.activation,
            name=f"{layer.name}_replaced"
        )

    # If Model or Sequential, recursively process sub-layers
    if isinstance(layer, (Model, Sequential)):
        new_layers = []
        for sublayer in layer.layers:
            replaced_sublayer = replace_all_transpose(sublayer)
            new_layers.append(replaced_sublayer)

        if isinstance(layer, Sequential):
            seq = Sequential(new_layers, name=layer.name)
            seq.build(layer.input_shape)
            return seq
        else:
            # functional sub-model
            inputs = layer.inputs if hasattr(layer, 'inputs') else None
            outputs = layer.outputs if hasattr(layer, 'outputs') else None
            new_model = Model(inputs=inputs, outputs=outputs, name=layer.name)
            new_model._layers = new_layers
            return new_model

    # Otherwise, return as-is
    return layer

# ------------------------------
# 4) BUILD & REPLACE
# ------------------------------
def build_unet_without_transpose(get_model_func):
    print("🔹 Building original TF model...")
    input_shape = (WINDOW_SIZE, WINDOW_SIZE, 3)
    model_tf = get_model_func(input_shape, output_channels=1, output_act='sigmoid')
    model_tf.load_weights(MODEL_WEIGHTS)
    model_tf.summary()

    print("🔹 Replacing all Conv2DTranspose with UpSamplingConv2D in TF model...")
    replaced = replace_all_transpose(model_tf)
    replaced.build(model_tf.input_shape)  # Attempt shape fix
    return replaced

# ------------------------------
# 5) EXPORT TO ONNX
# ------------------------------
def export_to_onnx(model_tf, onnx_path):
    print(f"\n🔹 Exporting replaced TF model => {onnx_path}")
    spec = [tf.TensorSpec((None, WINDOW_SIZE, WINDOW_SIZE, 3), tf.float32)]
    onnx_model, _ = tf2onnx.convert.from_keras(model_tf, opset=13, input_signature=spec)
    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f" ONNX model saved => {onnx_path}")

# ------------------------------
# 6) CHECK FOR ConvTranspose
# ------------------------------
def has_conv_transpose(onnx_path):
    model = onnx.load(onnx_path)
    for node in model.graph.node:
        if node.op_type == "ConvTranspose":
            return True
    return False

# ------------------------------
# 7) REWRITE ConvTranspose => Resize + Conv (No 'sizes' input!)
# ------------------------------
def rewrite_convtranspose(onnx_in, onnx_out):
    if not OGS_AVAILABLE:
        print(" onnx_graphsurgeon missing, cannot rewrite ConvTranspose.")
        return False

    import onnx_graphsurgeon as gs
    graph = gs.import_onnx(onnx.load(onnx_in))
    changed = False

    for node in graph.nodes:
        if node.op == "ConvTranspose":
            changed = True
            print(f"🛠 Rewriting {node.name or '(unnamed)'}")

            # Attributes
            strides = node.attrs.get("strides", [1,1])
            kernel_shape = node.attrs.get("kernel_shape", [3,3])
            pads = node.attrs.get("pads", [0,0,0,0])
            group = node.attrs.get("group", 1)
            dilations = node.attrs.get("dilations", [1,1])

            X = node.inputs[0]
            W = node.inputs[1]
            bias = node.inputs[2] if len(node.inputs) > 2 else None

            # A) Create Resize with float32 scales
            scale_vals = [1.0, 1.0, float(strides[0]), float(strides[1])]
            scale_const = gs.Constant(node.name+"_scale", np.array(scale_vals, dtype=np.float32))
            roi_const = gs.Constant(node.name+"_roi", np.array([], dtype=np.float32))  # empty float
            # No 'sizes' input => only 3 inputs: X, roi, scale

            out_upsample = gs.Variable(node.name+"_upsampled", dtype=np.float32)
            resize_node = gs.Node(
                op="Resize",
                name=node.name+"_resize",
                inputs=[X, roi_const, scale_const],
                outputs=[out_upsample],
                attrs={"mode":"nearest"}
            )

            # B) Optional Pad to replicate 'pads'
            pad_out = out_upsample
            if any(pads):
                top_pad = pads[2]
                left_pad = pads[3]
                # symmetrical
                bottom_pad = top_pad
                right_pad  = left_pad

                pad_vals = [0,0, top_pad, left_pad, 0,0, bottom_pad, right_pad]
                pad_vals_const = gs.Constant(node.name+"_pad_vals", np.array(pad_vals, dtype=np.int64))
                pad_fill_const = gs.Constant(node.name+"_pad_const_zero", np.array([0.0], dtype=np.float32))

                pad_op_out = gs.Variable(node.name+"_pad_out", dtype=np.float32)
                pad_op = gs.Node(
                    op="Pad",
                    name=node.name+"_pad",
                    inputs=[out_upsample, pad_vals_const, pad_fill_const],
                    outputs=[pad_op_out],
                    attrs={"mode":"constant"}
                )
                graph.nodes.append(pad_op)
                pad_out = pad_op_out

            # C) Transpose weights from [C_in,C_out,kH,kW] => [C_out,C_in,kH,kW]
            w_trans = gs.Variable(node.name+"_transposed_w", dtype=np.float32)
            transpose_node = gs.Node(
                op="Transpose",
                name=node.name+"_transpose_w",
                inputs=[W],
                outputs=[w_trans],
                attrs={"perm":[1,0,2,3]}
            )

            # D) Normal Conv
            conv_out = gs.Variable(node.name+"_conv_out", dtype=np.float32)
            conv_in = [pad_out, w_trans]
            if bias is not None:
                conv_in.append(bias)

            conv_node = gs.Node(
                op="Conv",
                name=node.name+"_conv",
                inputs=conv_in,
                outputs=[conv_out],
                attrs={
                    "kernel_shape": kernel_shape,
                    "strides": [1,1],
                    "pads": [0,0,0,0],
                    "group": group,
                    "dilations": dilations
                }
            )

            # E) Rewire old outputs => conv_out
            for old_out in node.outputs:
                for consumer in graph.nodes:
                    for i, c_inp in enumerate(consumer.inputs):
                        if c_inp == old_out:
                            consumer.inputs[i] = conv_out
            node.outputs = []

            graph.nodes.append(resize_node)
            graph.nodes.append(transpose_node)
            graph.nodes.append(conv_node)

    if changed:
        graph.cleanup().toposort()
        onnx.save(gs.export_onnx(graph), onnx_out)
        print(f" Rewritten ONNX => {onnx_out}")
    else:
        print(" No leftover ConvTranspose in ONNX.")
    return changed

# ------------------------------
# 8) CREATE DATASET
# ------------------------------
def create_dataset(img, window_size, step_size):
    width, height = img.size
    x_steps = list(range(0, width - window_size + 1, step_size))
    y_steps = list(range(0, height - window_size + 1, step_size))
    # Edge coverage
    if x_steps[-1] + window_size < width:
        x_steps.append(width - window_size)
    if y_steps[-1] + window_size < height:
        y_steps.append(height - window_size)

    for y in y_steps:
        for x in x_steps:
            box = (x, y, x+window_size, y+window_size)
            patch = img.crop(box)
            patch_array = np.array(patch, dtype=np.float32)/255.0
            if patch_array.ndim == 2:
                patch_array = np.stack([patch_array]*3, axis=-1)
            elif patch_array.shape[2] == 1:
                patch_array = np.concatenate([patch_array]*3, axis=-1)
            yield patch_array[:, :, :3], np.array([x, y], dtype=np.int32)

def prepare_dataset(img_path, window_size, step_size, batch_size):
    img = Image.open(img_path)
    def gen():
        yield from create_dataset(img, window_size, step_size)
    ds = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec((window_size, window_size, 3), tf.float32),
            tf.TensorSpec((2,), tf.int32)
        )
    )
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds, img.size

# ------------------------------
# 9) RUN INFERENCE (GPU) with Memory Workarounds
# ------------------------------
def run_onnx_inference(onnx_path):
    print(f"\n🔹 Running inference on {onnx_path} (GPU only) with memory optimizations..")

    # 1) Session Options
    sess_options = ort.SessionOptions()
    # Reduce memory usage & fragmentation:
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    sess_options.enable_mem_pattern = False
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED

    # 2) Force GPU only
    providers = ["CUDAExecutionProvider"]
    session = ort.InferenceSession(onnx_path, sess_options, providers=providers)
    actual_providers = session.get_providers()
    print(" ONNX Runtime using providers:", actual_providers)

    # 3) Input/Output names
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # 4) Prepare dataset
    dataset, (width, height) = prepare_dataset(IMG_PATH, WINDOW_SIZE, STEP_SIZE, BATCH_SIZE)
    print(f"📌 Input image size: Width={width}, Height={height}")

    # 5) Output buffer
    output_mask = np.zeros((height, width), dtype=np.float32)
    start_time = time.time()

    # 6) Sliding-window inference
    total_patches = 0
    for batch_patches, batch_positions in dataset:
        batch_patches_np = batch_patches.numpy()  # shape: (batch_size, W, H, 3)

        # Try inference
        preds = session.run([output_name], {input_name: batch_patches_np})[0]
        # Apply Sigmoid if model final activation is logistic
        preds = 1.0 / (1.0 + np.exp(-preds))

        # Place each patch
        for i in range(preds.shape[0]):
            p_mask = preds[i, :, :, 0]
            x_pos, y_pos = batch_positions[i].numpy()
            output_mask[y_pos:y_pos+WINDOW_SIZE, x_pos:x_pos+WINDOW_SIZE] = p_mask

        total_patches += preds.shape[0]
        if total_patches % 10 == 0:
            elapsed = time.time() - start_time
            print(f"🔄 Processed {total_patches} patches in {elapsed:.2f}s")

    # 7) Save final mask
    final_mask = (output_mask * 255).astype(np.uint8)
    out_img = Image.fromarray(final_mask)
    out_img.save(OUTPUT_PATH)
    print(f"\n Segmentation output saved => {OUTPUT_PATH}")
    print(f" Total ONNX inference time: {time.time() - start_time:.2f} seconds")

# ------------------------------
# 10) MAIN
# ------------------------------
if __name__ == "__main__":
    from unet import get_model

    # 1) Build TF model & remove Conv2DTranspose
    replaced_model = build_unet_without_transpose(get_model)

    # 2) Export to ONNX
    export_to_onnx(replaced_model, ONNX_MODEL_PATH)

    # 3) Check leftover ConvTranspose
    if has_conv_transpose(ONNX_MODEL_PATH):
        print(" ONNX STILL has ConvTranspose ops. Attempt rewriting..")
        if OGS_AVAILABLE:
            rewrite_convtranspose(ONNX_MODEL_PATH, ONNX_REWRITTEN_PATH)
            final_onnx = ONNX_REWRITTEN_PATH
        else:
            print(" onnx_graphsurgeon not installed, cannot rewrite at ONNX level.")
            sys.exit(1)
    else:
        print("No leftover ConvTranspose in ONNX.")
        final_onnx = ONNX_MODEL_PATH

    # 4) Run inference with memory optimizations
    run_onnx_inference(final_onnx)
