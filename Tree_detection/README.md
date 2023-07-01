# DeepForest Model

This code represents a Python class called `DeepForestModel` that implements a deep learning model for tree detection using the DeepForest library. The class provides methods for loading a trained model, training the model, evaluating the model on test data, and visualizing the results.

## Requirements

To run this code, you need the following dependencies installed:
- matplotlib
- torch
- geopandas
- shapely
- deepforest
- pandas

You can install these dependencies by running the following command:


pip install -r requirements.txt


## Usage

1. Import the required libraries:
python
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import csv
import torch
import time
import os
import geopandas as gpd
from shapely.geometry import Polygon
from deepforest import main
import shutil
import pandas as pd


2. Create an instance of the `DeepForestModel` class:
python
my_model = DeepForestModel()


3. Load a trained model:
python
model_path = "my_model.pt"
my_model.load_model(model_path)


4. Process an image and evaluate the model:
python
results, num_trees = my_model.process_and_evaluate()


5. Visualize the results:
python
my_model.visualize_results(results)


## Model Configuration

The `DeepForestModel` class is initialized with the following configuration:
- `source_image`: The path to the source image to be processed.
- `destination_folder`: The folder where the processed image and test file will be saved.
- `test_file`: The name of the test file in CSV format.
- `save_dir`: The directory to save the evaluation results.
- `output_shapefile_path`: The path to save the adjusted bounding boxes as a shapefile.

## Additional Information

This code assumes that you have a trained DeepForest model saved in a file named `my_model.pt`. You should replace this with the actual path to your trained model.

The code also includes methods for training the model (`train()`) and adjusting the bounding boxes of the detected trees (`adjust_bounding_boxes()`). However, these methods are not currently being used in the provided example. You can customize and extend the code based on your specific needs.

