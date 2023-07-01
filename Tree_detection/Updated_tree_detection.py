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


class DeepForestModel:
    def __init__(self):
        # Instantiate a new model object
        self.model = main.deepforest()
        self.model.use_release()
        self.num_trees = 0  # Number of trees variable

        self.source_image = r"C:\Users\FS-AI\Downloads\Tree canopy\Ramkrishna_Mahato\tree_canopy2482.JPG" # change it to accoringly


        self.destination_folder = r"C:\Users\FS-AI\Downloads\Tree canopy\Chayan_mandal"
        self.test_file = "data.csv"
        self.save_dir = r"C:\king"
        self.output_shapefile_path = r"C:\king\output_adjusted_shapefile.shp"

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
        self.num_trees = len(results['predictions'])  # Update number of trees
        return results

    def save_adjusted_bounding_boxes_as_shapefile(self, adjusted_boxes, output_path):
        # Create a GeoDataFrame with the adjusted bounding boxes
        adjusted_gdf = gpd.GeoDataFrame({'geometry': adjusted_boxes})

        # Save the GeoDataFrame as a shapefile
        adjusted_gdf.to_file(output_path)

    def adjust_bounding_boxes(self, bounding_boxes, image_height):
        adjusted_boxes = []
        for index, row in bounding_boxes.iterrows():
            xmin = row['xmin']
            ymin = row['ymin']
            xmax = row['xmax']
            ymax = row['ymax']

            # Invert the y-coordinates
            ymin = image_height - ymin  # flip ymin
            ymax = image_height - ymax  # flip ymax

            # Convert the bounding box to a Shapely Polygon
            polygon = Polygon([(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin)])
            adjusted_boxes.append(polygon)
        return adjusted_boxes

    def process_image(self):
        # Copy the source image to the destination folder
        shutil.copy(self.source_image, self.destination_folder)

        # Get the filename from the source image path
        filename = os.path.basename(self.source_image)

        # Create the test file path
        test_file_path = os.path.join(self.destination_folder, self.test_file)

        # Read the test file into a DataFrame
        a = pd.read_csv(test_file_path)

        # Set the 'image_path' column to the filename
        a['image_path'] = filename

        # Save the DataFrame back to the test file
        a.to_csv(test_file_path, index=False)

    def create_destination_folder_and_csv(self):
        if not os.path.exists(self.destination_folder):
            os.makedirs(self.destination_folder)
        else:
            print("Destination folder already exists.")

        # Specify the CSV file path
        csv_file_path = os.path.join(self.destination_folder, self.test_file)

        # Create the CSV file with header and example data
        data = [
            ["image_path", "xmin", "ymin", "xmax", "ymax", "label"],
            ["tree_canopy2482.JPG", "1", "1", "1", "1", "Tree"]
        ]

        with open(csv_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(data)

        print("CSV file created and saved successfully.")

    def process_and_evaluate(self):
        # Create the destination folder and CSV file
        self.create_destination_folder_and_csv()

        # Process the image (copy and update test file)
        self.process_image()

        # Evaluate the model on the test file and save the results
        results = self.evaluate(os.path.join(self.destination_folder, self.test_file), self.save_dir)

        # Get the number of trees from the class
        num_trees = self.num_trees

        return results, num_trees

    def visualize_results(self, results):
        img = mpimg.imread(self.source_image)
        image_height, image_width, _ = img.shape

        # Get the bounding boxes from the results
        bounding_boxes = results['predictions']

        adjusted_boxes = self.adjust_bounding_boxes(bounding_boxes, image_height)

        # Save the adjusted bounding boxes as a shapefile
        self.save_adjusted_bounding_boxes_as_shapefile(adjusted_boxes, self.output_shapefile_path)

        # Load the shapefile into a GeoDataFrame
        gdf = gpd.read_file(self.output_shapefile_path)

        # Extract the polygons from the GeoDataFrame and store them in a list
        adjusted_boxes = list(gdf.geometry)

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


# Usage example
if __name__ == '__main__':
    # Create an instance of the class
    my_model = DeepForestModel()

    # Load the saved model
    model_path = "my_model.pt"
    my_model.load_model(model_path)

    # Process the image and evaluate the model
    results, num_trees = my_model.process_and_evaluate()

    # Visualize the results
    my_model.visualize_results(results)
