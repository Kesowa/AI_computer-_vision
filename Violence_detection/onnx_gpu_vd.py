import cv2
import time
import os
import numpy as np
import tensorflow as tf
import onnx
import onnxruntime as ort

# Ensure TensorFlow is set up for GPU inference.
gpus = tf.config.list_physical_devices('GPU')
if not gpus:
    raise RuntimeError("No GPUs detected! Please ensure a GPU is available.")
else:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print("TensorFlow is using the following GPUs:")
    for gpu in gpus:
        print("  ", gpu)

IMG_SIZE = 128
IMG_PER_FILE = 30  # 30 frames per inference
MODEL_WEIGHTS_PATH = r'/home/aastha/Downloads/violence detection/ModelWeightbest_bests.h5'
VIDEO_PATH = r'/home/aastha/Downloads/violence detection/Media1.avi'
ONNX_PATH = r"__MODEL_PROTO.onnx"

def build_tf_model():
    """
    Rebuild the exact same TensorFlow model architecture used during training.
    """
    layers = tf.keras.layers
    models = tf.keras.models

    input_shapes = (IMG_SIZE, IMG_SIZE, 3)
    base_model = tf.keras.applications.vgg19.VGG19(
        include_top=False,
        weights='imagenet',
        input_shape=input_shapes
    )
    base_model.trainable = False

    cnn = models.Sequential([base_model, layers.Flatten()])

    model = models.Sequential()
    model.add(layers.TimeDistributed(cnn, input_shape=(IMG_PER_FILE, IMG_SIZE, IMG_SIZE, 3)))
    model.add(layers.LSTM(IMG_PER_FILE, return_sequences=True))
    model.add(layers.TimeDistributed(layers.Dense(90)))
    model.add(layers.Dropout(0.1))
    model.add(layers.GlobalAveragePooling1D())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(2, activation="sigmoid"))

    adam = tf.keras.optimizers.Adam(learning_rate=0.0005)
    model.compile(loss="binary_crossentropy", optimizer=adam, metrics=["accuracy"])
    return model

class ViolenceDetectorTF:
    """
    TensorFlow version of your violence detector.
    """
    def __init__(self, checkpoint_path):
        self.model = build_tf_model()
        self.model.load_weights(checkpoint_path)

    def predict_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        predictions = {'Keys': ['violence', 'nonviolence'], 'Values': []}
        count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frm = cv2.resize(rgb_img, (IMG_SIZE, IMG_SIZE))
            frm = frm / 255.0
            frames.append(frm[np.newaxis, ...])  # shape: (1, 128, 128, 3)
            count += 1

            if count == IMG_PER_FILE:
                # Prepare input: shape (1, 30, 128, 128, 3)
                X = np.array(frames)           # shape: (30, 1, 128, 128, 3)
                X = np.squeeze(X, axis=1)       # shape: (30, 128, 128, 3)
                X = np.expand_dims(X, axis=0)   # shape: (1, 30, 128, 128, 3)
                preds = self.model.predict(X)
                # Assuming output shape (1, 2)
                violence_score = float(preds[0][0])
                non_violence_score = float(preds[0][1])
                predictions['Values'].append([non_violence_score, violence_score])
                frames = []
                count = 0
        cap.release()
        return predictions

class ViolenceDetectorONNX:
    """
    ONNX version of your violence detector.
    Uses onnxruntime with CUDAExecutionProvider for GPU inference.
    """
    def __init__(self, onnx_path):
        # Check if the external data file exists (if large_model=True was used)
        external_data_file = onnx_path + ".data"
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"ONNX model file not found: {onnx_path}")
        # If the model was saved with external data, check for it.
        if os.path.exists(external_data_file):
            print(f"Found external data file: {external_data_file}")
        else:
            print("No external data file found. (If your model is large, ensure the external data file is present.)")
        
        # Try loading the model with onnx.load for an early check
        try:
            model_proto = onnx.load(onnx_path)
            onnx.checker.check_model(model_proto)
            print("ONNX model is valid!")
        except Exception as e:
            raise RuntimeError(f"ONNX model validation failed: {e}")
        
        # Create the inference session with GPU provider
        self.sess = ort.InferenceSession(onnx_path, providers=['CUDAExecutionProvider'])
        self.input_name = self.sess.get_inputs()[0].name

    def predict_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        predictions = {'Keys': ['violence', 'nonviolence'], 'Values': []}
        count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frm = cv2.resize(rgb_img, (IMG_SIZE, IMG_SIZE))
            frm = frm / 255.0
            frames.append(frm[np.newaxis, ...])  # shape: (1, 128, 128, 3)
            count += 1

            if count == IMG_PER_FILE:
                # Prepare input: shape (1, 30, 128, 128, 3)
                X = np.array(frames)             # shape: (30, 1, 128, 128, 3)
                X = np.squeeze(X, axis=1)         # shape: (30, 128, 128, 3)
                X = np.expand_dims(X, axis=0).astype(np.float32)  # shape: (1, 30, 128, 128, 3)
                preds = self.sess.run(None, {self.input_name: X})[0]
                violence_score = float(preds[0][0])
                non_violence_score = float(preds[0][1])
                predictions['Values'].append([non_violence_score, violence_score])
                frames = []
                count = 0
        cap.release()
        return predictions

def main():
    # TensorFlow Inference
    print("\n--- Running TensorFlow Inference on GPU ---")
    tf_detector = ViolenceDetectorTF(MODEL_WEIGHTS_PATH)
    start_tf = time.time()
    tf_predictions = tf_detector.predict_frames(VIDEO_PATH)
    end_tf = time.time()
    tf_inference_time = end_tf - start_tf

    print("\n=== TensorFlow Predictions ===")
    for i, pred in enumerate(tf_predictions['Values']):
        print(f"Chunk {i+1}: Violence Score: {pred[1]:.4f}, Non-Violence Score: {pred[0]:.4f}")
    print(f"TensorFlow GPU Inference Time: {tf_inference_time:.4f} seconds.")

    # ONNX Inference
    print("\n--- Running ONNX Inference on GPU ---")
    try:
        onnx_detector = ViolenceDetectorONNX(ONNX_PATH)
    except Exception as e:
        print(f"Error creating ONNX Inference Session: {e}")
        return
    
    start_onnx = time.time()
    onnx_predictions = onnx_detector.predict_frames(VIDEO_PATH)
    end_onnx = time.time()
    onnx_inference_time = end_onnx - start_onnx

    print("\n=== ONNX Predictions ===")
    for i, pred in enumerate(onnx_predictions['Values']):
        print(f"Chunk {i+1}: Violence Score: {pred[1]:.4f}, Non-Violence Score: {pred[0]:.4f}")
    print(f"ONNX GPU Inference Time: {onnx_inference_time:.4f} seconds.")

    print("\n=== Summary ===")
    print(f"TensorFlow GPU Inference Time: {tf_inference_time:.4f} s")
    print(f"ONNX GPU Inference Time:       {onnx_inference_time:.4f} s")

if __name__ == "__main__":
    main()
