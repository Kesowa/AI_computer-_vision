
import torch
import time
import os
import geopandas as gpd
from shapely.geometry import Polygon
from deepforest import main
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class DeepForestModel:
    def __init__(self):
        # Instantiate a new model object
        self.model = main.deepforest()
        self.model.use_release()
        
    def load_model(self, model_path):
        # Load the saved state dictionary into the new model object
        state_dict = torch.load(model_path)
        self.model.load_state_dict(state_dict)
    
    def train(self, train_data):
        start_time = time.time()
        self.model.trainer.fit(train_data)
        print(f"--- Training on CPU: {(time.time() - start_time):.2f} seconds ---")
        
    def evaluate(self, test_file, save_dir):
        results = self.model.evaluate(test_file, os.path.dirname(test_file), iou_threshold=0.4, savedir=save_dir)
        return results
    def save_adjusted_bounding_boxes_as_shapefile(self, adjusted_boxes, output_path):
        # Create a GeoDataFrame with the adjusted bounding boxes
        adjusted_gdf = gpd.GeoDataFrame({'geometry': adjusted_boxes})
        
        # Save the GeoDataFrame as a shapefile
        adjusted_gdf.to_file(output_path)
    def save_bounding_boxes_as_shapefile(self, results, output_path):
        # Extract the bounding box results
        bounding_boxes = results['results']

        # Create a list to store the polygons
        polygons = []

        # Iterate through each bounding box
        for index, row in bounding_boxes.iterrows():
            xmin = row['xmin']
            ymin = row['ymin']
            xmax = row['xmax']
            ymax = row['ymax']

            # Convert the bounding box to a Shapely Polygon
            polygon = Polygon([(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin)])

            # Append the polygon to the list
            polygons.append(polygon)

        # Create a GeoDataFrame
        gdf = gpd.GeoDataFrame({'geometry': polygons})

        # Save the GeoDataFrame as a shapefile
        gdf.to_file(output_path)






    def adjust_bounding_boxes(self, bounding_boxes, image_height):
        adjusted_boxes = []
        for index, row in bounding_boxes.iterrows():
            xmin = row['xmin']
            ymin = row['ymin']
            xmax = row['xmax']
            ymax = row['ymax']
            
            # Invert the y-coordinates
            ymin = image_height - ymin # flip ymin
            ymax = image_height - ymax # flip ymax

            # Convert the bounding box to a Shapely Polygon
            polygon = Polygon([(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin)])
            adjusted_boxes.append(polygon)
        return adjusted_boxes


        

# Create an instance of the class
my_model = DeepForestModel()

# Load the saved model
model_path = "my_model.pt"
my_model.load_model(model_path)

# Evaluate the model on some test data
test_file = r'C:\Users\FS-AI\Downloads\Tree canopy\Chayan_mandal\train_example.csv'
save_dir = r'C:\save_m'
results = my_model.evaluate(test_file, save_dir)

# Save the bounding boxes as a shapefile
output_path = "output_shapefile.shp"
# my_model.save_bounding_boxes_as_shapefile(results, output_path)
output_adjusted_path = "output_adjusted_shapefile.shp"
my_model.save_adjusted_bounding_boxes_as_shapefile(adjusted_boxes, output_adjusted_path)

img = mpimg.imread(r'C:\Users\FS-AI\Downloads\Tree canopy\Chayan_mandal\tree_canopy1241.jpg')
image_height, image_width, _ = img.shape

# Get the bounding boxes from the results
bounding_boxes = results['results']

# Adjust the bounding boxes
adjusted_boxes = my_model.adjust_bounding_boxes(bounding_boxes, image_height)

# Create a GeoDataFrame with the adjusted bounding boxes
adjusted_gdf = gpd.GeoDataFrame({'geometry': adjusted_boxes})

# Create a figure and axis
fig, ax = plt.subplots()

# Plot the image
ax.imshow(img, extent=[0, image_width, 0, image_height])

# Plot the adjusted shapefile on top
adjusted_gdf.plot(ax=ax, facecolor="none", edgecolor='red', linewidth=1.5)

plt.xlabel("Pixels X")
plt.ylabel("Pixels Y")
plt.title("Adjusted Polygons overlay on Image")

plt.show()

