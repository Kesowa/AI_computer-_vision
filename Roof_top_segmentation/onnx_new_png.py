import os
import time
import numpy as np
import onnxruntime as ort
from PIL import Image

# ------------------------------
# CONFIG
# ------------------------------
IMG_PATH = "/home/debian/Downloads/tree_detection/rooftop.png"
ONNX_MODEL_PATH = "/home/debian/Downloads/tree_detection/unet_fp16.onnx"
OUTPUT_PATH = "/home/debian/Downloads/tree_detection/rooftop_output_onnx_only.png"

IMG_SIZE = 512

# ------------------------------
# CHECK FILES
# ------------------------------
if not os.path.exists(IMG_PATH):
    raise FileNotFoundError(IMG_PATH)

if not os.path.exists(ONNX_MODEL_PATH):
    raise FileNotFoundError(ONNX_MODEL_PATH)

# ------------------------------
# LOAD IMAGE
# ------------------------------
img = Image.open(IMG_PATH).convert("RGB")
img = img.resize((IMG_SIZE, IMG_SIZE))
img = np.asarray(img, dtype=np.float32) / 255.0
img = np.expand_dims(img, axis=0).astype(np.float16)  # FP16 input

# ------------------------------
# CREATE ONNX SESSION (CUDA → CPU FALLBACK)
# ------------------------------
print("🔹 Initializing ONNX Runtime")

sess_options = ort.SessionOptions()
sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

# Prefer CUDA, fallback to CPU automatically
providers = [
    "CUDAExecutionProvider",
    "CPUExecutionProvider"
]

session = ort.InferenceSession(
    ONNX_MODEL_PATH,
    sess_options=sess_options,
    providers=providers
)

active_providers = session.get_providers()
print("✅ Active providers:", active_providers)

if "CUDAExecutionProvider" in active_providers:
    print("🚀 Running on GPU (CUDA)")
else:
    print("⚠️ CUDA not available, running on CPU")

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# ------------------------------
# INFERENCE
# ------------------------------
print("🚀 Running ONNX inference...")
start = time.time()

pred = session.run(
    [output_name],
    {input_name: img}
)[0]

pred = pred[0, :, :, 0]  # (512, 512)

print(f"⏱️ Inference time: {time.time() - start:.3f}s")

# ------------------------------
# SAVE OUTPUT
# ------------------------------
pred_uint8 = (pred * 255).astype(np.uint8)
Image.fromarray(pred_uint8).save(OUTPUT_PATH)

print(f"✅ Output saved to: {OUTPUT_PATH}")
