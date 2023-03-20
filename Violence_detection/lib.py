import cv2
import os
import numpy as np

class ViolenceDetector:
    IMG_SIZE = 128
    MODEL_WEIGHTS_PATH = ""
    FRAME_BUFFER_SIZE = 30
    
    def __init__(self, checkpoint: string = r'C:\Users\FS-AI\Desktop\violence detection\ModelWeightbest_bests.h5'):
        this.MODEL_WEIGHTS_PATH = checkpoint
        self.model = self._load_model()
        self.frame_buffer = []

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

    def predict_frames(self, frames):
        x=[]
        y=[]
        X=[]
        k=0
        l=0
        count=0
        for filename in os.listdir(frames):
            frame=cv2.imread(os.path.join(frames,filename))
            rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frm = cv2.resize(rgb_img, (IMG_SIZE, IMG_SIZE))
            frm = np.expand_dims(frm, axis=0)
            frm = frm / 255.0
            x.append(frm)
            count+=1
            if count==30:
                X.append(x)
                y.append(1)
                count=0
                x=[]
        X = np.array(X)
        X=np.squeeze(X,axis=2)
        # print(X.shape)
        predictions = self.model.predict(X)  
        # print(predictions)
        preds = np.apply_along_axis(self._predictor, 1, predictions)
        for i in preds :
            if(i=='Violence'):
                k+=1
            else:
                l+=1
        print(k,l)

