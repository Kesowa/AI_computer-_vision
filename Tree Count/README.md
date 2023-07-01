## Tree Canopy Detection using DeepForest

The provided code snippet will allow you to process an image, evaluate the pre-trained model, and visualize the results with adjusted bounding boxes overlaid on the image.

### 1. Installation

First, make sure you have Python installed on your system. Then, install the required dependencies by running the following commands in your terminal or command prompt:

```bash
pip install torch
pip install geopandas
pip install deepforest
```

Setting up a virtual environment is recommended to avoid conflicts with other Python projects.

### 2. Code Setup

Copy the provided code snippet into a Python file, such as `deepforest_model.py`.

### 3. Importing the Class

In your main Python script or Jupyter Notebook, import the `DeepForestModel` class:

```python
from deepforest_model import DeepForestModel
```

### 4. Model Initialization and Loading

Create an instance of the `DeepForestModel` class:

```python
my_model = DeepForestModel()
```

Load the pre-trained model by providing the path to the saved model file (`.pt` file). Replace `"my_model.pt"` with the actual path to your saved model:

```python
model_path = "my_model.pt"
my_model.load_model(model_path)
```

### 5. Image Processing and Evaluation

#### 5.1 Provide the Source Image and Destination Folder

Specify the path to the source image and the destination folder where the processed image and test file will be stored:

```python
my_model.source_image = "path/to/source_image.jpg"
my_model.destination_folder = "path/to/destination_folder"
```

#### 5.2 Create Test File

Create a CSV test file with an example bounding box entry. The format should be as follows:

```python
csv_file_path = os.path.join(my_model.destination_folder, "data.csv")

data = [
    ["image_path", "xmin", "ymin", "xmax", "ymax", "label"],
    ["source_image.jpg", "xmin_value", "ymin_value", "xmax_value", "ymax_value", "Tree"]
]

with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(data)
```

#### 5.3 Process Image and Evaluate

Process the image and evaluate the model on the test file:

```python
results, num_trees = my_model.process_and_evaluate()
```

The `results` variable contains the evaluation results, and `num_trees` stores the number of trees detected.

### 6. Visualizing the Results

To visualize the results with adjusted bounding boxes overlaid on the source image, call the `visualize_results()` method:

```python
my_model.visualize_results(results)
```

This will display the source image with the adjusted bounding boxes marked in red.

### 7. Customization

The code snippet allows you to modify various paths, such as the `save_dir` for evaluation results and the `output_shapefile_path` for saving the adjusted bounding boxes as a shapefile.

Make sure to replace the placeholder paths and filenames with the actual paths and filenames relevant to your setup.

### Note

Ensure that you have permission to access the source image and that the paths specified in the `DeepForestModel` class are correct and accessible.

