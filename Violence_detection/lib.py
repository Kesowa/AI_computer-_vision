import cv2
import os
import numpy as np
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import  Dropout, Dense, Flatten, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.utils import plot_model
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.callbacks import Callback, ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping, ReduceLROnPlateau

class ViolenceDetector:
    IMG_SIZE = 128
    MODEL_WEIGHTS_PATH = r'C:\Users\FS-AI\Desktop\violence_dont\Mode_single_frames30.h5'
    FRAME_BUFFER_SIZE = 30
    frame_count=0
    def my_model(tf):

        # SEED
        np.random.seed(101)

        # LAYER
        layers = tf.keras.layers
        models = tf.keras.models
        losses = tf.keras.losses
        optimizers = tf.keras.optimizers
        metrics = tf.keras.metrics
        
        cnn = models.Sequential()
    
        input_shapes=(IMG_SIZE, IMG_SIZE, CHANNELS)

        VGG19_MODEL = tf.keras.applications.vgg19.VGG19

        base_model = VGG19_MODEL(include_top=False, weights='imagenet', input_shape=input_shapes)
        # Freeze the layers except the last 4 layers (we will only use the base model to extract features)
        cnn = models.Sequential()
        cnn.add(base_model)
        cnn.add(layers.Flatten())
        model = models.Sequential()
        # add cnn model
        model.add(layers.TimeDistributed(cnn, input_shape=(IMG_PER_FILE, IMG_SIZE, IMG_SIZE, CHANNELS)))
        model.add(layers.LSTM(IMG_PER_FILE , return_sequences= True))
        model.add(layers.TimeDistributed(layers.Dense(90))) 
        ## Full-connected layers
        model.add(layers.Dropout(0.1))
        model.add(layers.GlobalAveragePooling1D())
        model.add(layers.Dense(512, activation='relu'))
        # added
        # model.add(layers.Dense(512, activation='relu'))
        # model.add(layers.Dense(256, activation='relu'))


        model.add(layers.Dropout(0.3))
        model.add(layers.Dense(NUM_CLASSES, activation="sigmoid"))
        adam = optimizers.Adam(learning_rate=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        #model.load_weights(wgts)
        rms = optimizers.RMSprop()
        model.compile(loss="binary_crossentropy",
                        optimizer=adam,
                        metrics=["accuracy"])
        return model

    def __init__(self, checkpoint: str):
        self.MODEL_WEIGHTS_PATH = checkpoint
        self.model=my_model(tf)
        self.model = self._load_model()
        # self.frame_buffer = []

    def _load_model(self):
        model.load_weights(self.MODEL_WEIGHTS_PATH)
        # Load your trained model from the specified path
        # model = load_model(self.MODEL_WEIGHTS_PATH)
        return model

    def _predictor(self, preds):
        if preds[0] > 0.1:
            return 'Non-Violence'
        else:
            return 'Violence'

    def predict_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        predictions = {
            'Keys': ['violence', 'nonviolence'],
            'Values': []
        }
        count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # frame_count += 1
            # if frame_count % 5 != 0:  # skip frames that are not multiples of 5
            #     continue
            rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frm = cv2.resize(rgb_img, (self.IMG_SIZE, self.IMG_SIZE))
            frm = np.expand_dims(frm, axis=0)
            frm = frm / 255.0
            frames.append(frm)
            count += 1
            if count == self.FRAME_BUFFER_SIZE:
                X = np.array(frames)
                X = np.squeeze(X, axis=1)
                X = np.expand_dims(X, axis=0)
                preds = self.model.predict(X)
                row = []
                for pred in preds:
                    row.append([pred[1], pred[0]])
                predictions['Values'].append(row)
                frames = []
                count = 0
        cap.release()
        return predictions
