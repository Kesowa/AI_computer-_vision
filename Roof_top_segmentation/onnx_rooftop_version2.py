import os
import sys
import time
import numpy as np
from PIL import Image

import tensorflow as tf
from tensorflow.keras.layers import Layer, UpSampling2D, Conv2D, Conv2DTranspose
from tensorflow.keras import Model, Sequential

import tf2onnx
import onnx
import onnxruntime as ort

IMG_PATH = "/home/aastha/Downloads/rooftop_segmentation/small_crop1.tif"
WINDOW_SIZE = 1024
STEP_SIZE = 1024
BATCH_SIZE = 1
MODEL_WEIGHTS = "Final_model.h5"
ONNX_MODEL_PATH = "model.onnx"
OUTPUT_PATH = "/home/aastha/Downloads/rooftop_segmentation/sliding_window_onnx.tif"

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
Image.MAX_IMAGE_PIXELS = None

class UpSamplingConv2D(Layer):
    def __init__(self, filters, kernel_size, strides=(2,2), padding='same', activation=None, **kwargs):
        super().__init__(**kwargs)
        self.upsample = UpSampling2D(size=strides, interpolation='nearest')
        self.conv = Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            padding=padding,
            activation=activation
        )

    def call(self, inputs):
        x = self.upsample(inputs)
        x = self.conv(x)
        return x

def replace_all_transpose(layer: tf.keras.layers.Layer) -> tf.keras.layers.Layer:
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
    
    if isinstance(layer, Model) or isinstance(layer, Sequential):
        new_layers = []
        for sublayer in layer.layers:
            replaced_sublayer = replace_all_transpose(sublayer)
            new_layers.append(replaced_sublayer)

        if isinstance(layer, Sequential):
            new_seq = Sequential(new_layers, name=layer.name)
            new_seq.build(layer.input_shape)
            return new_seq
        else:
            inputs = layer.inputs if hasattr(layer, 'inputs') else None
            outputs = layer.outputs if hasattr(layer, 'outputs') else None
            new_model = Model(inputs=inputs, outputs=outputs, name=layer.name)
            new_model._layers = new_layers
            return new_model

    return layer

def build_unet_without_transpose(get_model_func):
    input_shape = (WINDOW_SIZE, WINDOW_SIZE, 3)
    model = get_model_func(input_shape, output_channels=1, output_act='sigmoid')
    model.load_weights(MODEL_WEIGHTS)
    model.summary()

    replaced_model = replace_all_transpose(model)
    replaced_model.build(model.input_shape)
    return model, replaced_model

def force_convert_to_onnx(replaced_model: Model):
    print("\nConverting replaced model to ONNX (overwriting any existing file)...")
    input_signature = [tf.TensorSpec(shape=(None, 1024, 1024, 3), dtype=tf.float32)]
    
    onnx_model, _ = tf2onnx.convert.from_keras(
        replaced_model,
        opset=13,
        input_signature=input_signature
    )

    with open(ONNX_MODEL_PATH, "wb") as f:
        f.write(onnx_model.SerializeToString())

    print(f"ONNX model successfully saved to {ONNX_MODEL_PATH}")

def create_dataset(img, window_size, step_size):
    width, height = img.size
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
            yield patch_array.astype(np.float32), np.array([x, y], dtype=np.int32)

def prepare_dataset(img_path, window_size, step_size, batch_size):
    img = Image.open(img_path)
    def gen():
        yield from create_dataset(img, window_size, step_size)
    dataset = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=(window_size, window_size, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(2,), dtype=tf.int32)
        )
    )
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset, img.size

def run_onnx_inference():
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    providers = ["CUDAExecutionProvider"]

    session = ort.InferenceSession(ONNX_MODEL_PATH, sess_options, providers=providers)
    print("\nONNX Runtime is running with:", session.get_providers())

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    dataset, (width, height) = prepare_dataset(IMG_PATH, WINDOW_SIZE, STEP_SIZE, BATCH_SIZE)
    print(f"Input image size: Width={width}, Height={height}")

    output_mask = np.zeros((height, width), dtype=np.float32)
    start_time = time.time()

    total_patches = 0
    for batch_patches, batch_positions in dataset:
        batch_patches_np = batch_patches.numpy()
        preds = session.run([output_name], {input_name: batch_patches_np})[0]
        preds = 1 / (1 + np.exp(-preds))

        for i in range(preds.shape[0]):
            pred_mask = preds[i, :, :, 0]
            x_pos, y_pos = batch_positions[i].numpy()
            output_mask[y_pos:y_pos+WINDOW_SIZE, x_pos:x_pos+WINDOW_SIZE] = pred_mask

        total_patches += preds.shape[0]
        if total_patches % 10 == 0:
            elapsed = time.time() - start_time
            print(f"Processed {total_patches} patches in {elapsed:.2f} s")

    out_img = Image.fromarray((output_mask * 255).astype(np.uint8))
    out_img.save(OUTPUT_PATH)
    print(f"\nSegmentation output saved to {OUTPUT_PATH}")
    print(f"Total ONNX inference time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    from unet import get_model
    original_model, replaced_model = build_unet_without_transpose(get_model)
    force_convert_to_onnx(replaced_model)
    run_onnx_inference()
