import cv2
import os
import numpy as np
import pickle
from tqdm import tqdm
from vgg_face import VGG_Face, IMAGE_HEIGHT, IMAGE_WIDTH
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import Normalizer

def get_embeddings(face_array, model):
	face = cv2.resize(face_array,
					(IMAGE_HEIGHT, IMAGE_WIDTH),
					interpolation = cv2.INTER_NEAREST)
	face = np.expand_dims(face,axis = 0)
	embeddings = model.predict(face)[0,:]
	return embeddings

def precompute_embeddings(database_path = None):
	feature_extractor = VGG_Face()
	image_names = os.listdir(path = database_path)
	embeddings = []
	for image in tqdm(image_names):
		image_path = os.path.join(database_path, image)
		image_array = cv2.imread(image_path)
		RGBimage = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
		embedding = get_embeddings(face_array = RGBimage, model = feature_extractor)
		embeddings.append(embedding)
	 
	names = [nm.split(" ")[0] for nm in image_names]
	with open('embeddings/names.pkl', 'wb') as f:
	    pickle.dump(names, f)
	with open('embeddings/embeddings.pkl', 'wb') as f:
	    pickle.dump(embeddings, f)
	print(f"Embedings saved!")

def findCosineDistance(source_representation, test_representation):
	# computes cosine distance
    a = np.matmul(np.transpose(source_representation), test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))

def normalize_input_variable(input_vars = None, norm = "l2", store_path = "model/"):
	# We must perform data normalization
	in_encoder = Normalizer(norm='l2')
	in_encoder.fit(input_vars)
	normalized_input_vars = in_encoder.transform(input_vars)
	# saving the normalizer
	pickle.dump(in_encoder, open(os.path.join(store_path, 'normalizer.pkl'), 'wb'))
	return normalized_input_vars

def encode_labels(output_vars = None, store_path = "model/"):
	# Performing label encoding
	out_encoder = LabelEncoder()
	out_encoder.fit(output_vars)
	encoded_output_vars = out_encoder.transform(output_vars)
	# saving the encoder
	pickle.dump(out_encoder, open(os.path.join(store_path, 'label_encoder.pkl'), 'wb'))
	return encoded_output_vars

def build_dataset(dataset_path):
	# Input: Database path
	# Returns: Person names and embedding corresponding to person names 
	feature_extractor = VGG_Face()
	embedding_list = []
	person_names = []
	for folder in os.listdir(dataset_path):
	    f_path = os.path.join(dataset_path, folder)
	    for images in os.listdir(f_path):
	        image_path = os.path.join(f_path, images)
	        # reading the image
	        image = cv2.imread(image_path)
	        # getting the embedding of the image from VGG-FACE model
	        e = get_embeddings(image, model = feature_extractor)
	        embedding_list.append(e)
	        person_names.append(images.split("_")[0])
	return np.array(embedding_list), person_names



