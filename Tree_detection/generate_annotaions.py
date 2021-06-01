import os
import pandas as pd
import numpy as np
from deepforest import deepforest
from deepforest import get_data, utilities

annotations_list = []

DATASET_DIR = "../dataset"
label_xml_files = [file for file in os.listdir(DATASET_DIR) if file.split(".")[1] == "xml"]

data = []
for annot in label_xml_files:
    try:
        annot_path = os.path.join(DATASET_DIR, annot)
        annotation = utilities.xml_to_annotations(annot_path).to_numpy()
        if len(annotation) == 1:
            data.append(annotation[0].tolist())
        else:
            for lst in annotation:
                data.append(lst.tolist())
    except:
        pass

# building a numpy array
data = np.array(data)
# building the dataframe
df = pd.DataFrame({"image_path" : data[..., 0],
                   "xmin" : data[..., 1],
                   "ymin" : data[..., 2],
                   "xmax" : data[..., 3],
                   "ymax" : data[..., 4],
                   "label" : data[..., 5]}) 

df.image_path = [os.path.join("dataset", img) for img in df.image_path]
df.label = ["tree" for i in df.image_path]

# saving the csv file using pandas
os.chdir("/home/ricky/Desktop/Kesowa/Tree Detection/DeepForest/deepforest/data")
df.to_csv("train.csv", header = False, index = False)