import cv2, os
import numpy as np
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
from vgg_face import VGG_Face, IMAGE_HEIGHT, IMAGE_WIDTH
from retinaface import RetinaFace
from utils import findCosineDistance, get_embeddings

# Hyperparams
COSINE_SIMILARITY_THRESHOLD = 0.32

# Loading models and feature engineering models
face_feature_extractor = VGG_Face()
detector = RetinaFace(quality="easy")

embeddings = pickle.load(open("embeddings/embeddings.pkl","rb"))
names = pickle.load(open("embeddings/names.pkl","rb"))

model = pickle.load(open("model/SVC.pkl","rb"))
label_encoder = pickle.load(open("model/label_encoder.pkl","rb"))

def recognise(face_array = None):
	distances_from_embeddings = []
	query_embedding = get_embeddings(face_array = face_array, model = face_feature_extractor)
	for e in embeddings:
	    embd_dist = findCosineDistance(query_embedding, e)
	    distances_from_embeddings.append(embd_dist)
	index = np.argmin(distances_from_embeddings)
	if np.min(distances_from_embeddings) < COSINE_SIMILARITY_THRESHOLD:
	    output = model.predict([query_embedding])
	    name = label_encoder.inverse_transform(output)[0]
	    return name
	else:
	    return "Not Verified!"

def live_verification(cam_id = 0):
	vid = cv2.VideoCapture(cam_id)
	while(True):
		ret, frame = vid.read()
		screen = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		face_coordinates = detector.predict(screen)
		for points in face_coordinates:
			x1, y1, x2, y2 = int(points["x1"]), int(points["y1"]), int(points["x2"]), int(points["y2"])
			face_roi = frame[y1:y2,x1:x2]
			if face_roi.shape[0] != 0:
				person = recognise(face_array = face_roi)
				if person == "Not Verified!": 
					cv2.putText(frame, person, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 4)
					cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)

				else:
					cv2.putText(frame, person, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
					cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
			cv2.imshow("Face Verification", frame)
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break	
	vid.release()
	cv2.destroyAllWindows()

live_verification(cam_id = 0)
