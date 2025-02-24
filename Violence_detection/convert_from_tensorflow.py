import os
# Disable GPU so that TensorFlow uses CPU kernels (including for LSTM)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf

IMG_SIZE = 128
IMG_PER_FILE = 30
MODEL_WEIGHTS_PATH = r'/home/aastha/Downloads/violence detection/ModelWeightbest_bests.h5'

def build_functional_model():
    inputs = tf.keras.Input(shape=(IMG_PER_FILE, IMG_SIZE, IMG_SIZE, 3), name="model_input")
    base_model = tf.keras.applications.vgg19.VGG19(
        include_top=False,
        weights='imagenet',
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    base_model.trainable = False
    flatten_layer = tf.keras.Sequential([base_model, tf.keras.layers.Flatten()])
    x = tf.keras.layers.TimeDistributed(flatten_layer)(inputs)
    # This LSTM will now use the CPU (generic) implementation
    x = tf.keras.layers.LSTM(IMG_PER_FILE, return_sequences=True)(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(90))(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(2, activation='sigmoid', name="model_output")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="violence_lstm_functional")
    model.compile(loss="binary_crossentropy",
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
                  metrics=["accuracy"])
    return model

model = build_functional_model()
model.load_weights(MODEL_WEIGHTS_PATH)

# Export the model as a SavedModel (using Keras 3's export)
model.export("saved_model_violence_cpu")
print("SavedModel exported to 'saved_model_violence_cpu'.")
