import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import csv
import os
import geopandas as gpd
from shapely.geometry import Polygon
from deepforest import main
import shutil
import pandas as pd
import torch
from PIL import Image
import cv2
import numpy as np
import rasterio
from rasterio.windows import Window
class DeepForestModel:
    counter=0

    def __init__(self, source_image):
        # Instantiate a new model object
        self.model = main.deepforest()
        self.model.use_release()
        self.num_trees = 0  # Number of trees variable

        self.source_image = source_image
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")# adde
        self.test_file = "data.csv"
        # path = f"C:\\Users\\FS-AI\\Desktop\\try_kesowa\\king\\output_adjusted_shapefile{DeepForestModel.counter}.shp"

        DeepForestModel.counter += 1
        self.save_dir = r"C:\Users\FS-AI\Desktop\output_current1"# path to directory where you want to save output shape file.
        self.destination_folder = self.save_dir

        if not os.path.exists(self.save_dir):
    # Create the directory if it doesn't exist
            os.makedirs(self.save_dir)

    def process_tiff_image(self, tiff_path):
        with rasterio.open(tiff_path) as src:
            # Calculate the number of tiles in both dimensions
            n_tiles_x = int(np.ceil(src.width / 700))
            n_tiles_y = int(np.ceil(src.height / 700))

            total_trees = 0

            for i in range(n_tiles_y):
                for j in range(n_tiles_x):
                    # Define the window to read
                    window = Window(j*700, i*700, 700, 700)
                    tile = src.read(window=window)

                    # Convert the 3D array to a PIL image
                    tile_image = Image.fromarray(np.transpose(tile, (1, 2, 0)))

                    # Save tile image temporarily
                    tile_path = os.path.join(self.destination_folder, f"tile_{i}_{j}.png")
                    tile_image.save(tile_path)

                    # Set the tile as the current source image
                    self.source_image = tile_path
                    results, num_trees = self.process_and_evaluate()
                    total_trees += num_trees
                    bounding_boxes = results['predictions']

                    adjusted_boxes = self.adjust_bounding_boxes(bounding_boxes, 700)

                    # Save the adjusted bounding boxes as a shapefile
                    self.save_adjusted_bounding_boxes_as_shapefile(adjusted_boxes, self.get_shapefile_path(f"tile_{i}_{j}.png"))

                    # Remove the temporary tile image
                    os.remove(tile_path)

            return total_trees
    def get_shapefile_path(self, image_filename):
        # Extract the base name without extension
        base_name = os.path.splitext(image_filename)[0]
        # Use this base name for the shapefile
        shapefile_name = f"{base_name}.shp"
        path = os.path.join(self.save_dir, shapefile_name)
        return path
    def load_model(self, model_path):
        # Load the saved state dictionary into the new model object
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)

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
    def process_image(self):
        # Load the source TIFF image using PIL (Python Imaging Library)
        image = Image.open(self.source_image)
        resolution = self.calculate_resolution(self.source_image)
        print(f"Resolution of {self.source_image}: {resolution} meters/pixel")
        # Calculate the new dimensions while maintaining the aspect ratio
        max_dimension = self.calculate_window_size(resolution)  # Set a reasonable maximum dimension
        width, height = image.size
        if width > max_dimension or height > max_dimension:
            ratio = max_dimension / max(width, height)
            new_width = int(width * ratio)
            new_height = int(height * ratio)
            image = image.resize((new_width, new_height), Image.ANTIALIAS)

        # Copy the resized image to the destination folder
        destination_image_path = os.path.join(self.destination_folder, os.path.basename(self.source_image))
        image.save(destination_image_path)

        # Get the filename from the source image path
        filename = os.path.basename(destination_image_path)

        # Create the test file path
        test_file_path = os.path.join(self.destination_folder, self.test_file)

        # Read the test file into a DataFrame
        a = pd.read_csv(test_file_path)

        # Set the 'image_path' column to the filename
        a['image_path'] = filename

        # Save the DataFrame back to the test file
        a.to_csv(test_file_path, index=False)
    # def process_image(self):
    def calculate_window_size(self, resolution):
        if resolution <= 0.1:
            # For 0.1m data, use a linear interpolation between 400 and 800 pixels
            return int(np.interp(resolution, [0, 0.1], [400, 800]))
        elif resolution > 0.1:
            # For coarser resolution tiles, you may experiment with larger window sizes
            # Adjust this based on your specific requirements
            return int(resolution * 40000)  # Adjust some_scaling_factor
        else:
            # Handle other cases or provide default value
            return 700  # Default value

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
            [os.path.basename(self.source_image), "1", "1", "1", "1", "Tree"]
        ]

        with open(csv_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(data)

        print("CSV file created and saved successfully.")

    def process_and_evaluate(self):
        self.create_destination_folder_and_csv()

        self.process_image()
        results = self.evaluate(os.path.join(self.destination_folder, self.test_file), self.save_dir)

        num_trees = self.num_trees

        return results, num_trees

    def process_images_in_folder(self, image_folder):
        total_trees = 0

        for filename in os.listdir(image_folder):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                # Process each image
                self.source_image = os.path.join(image_folder, filename)
                results, num_trees = self.process_and_evaluate()
                total_trees += num_trees
                bounding_boxes = results['predictions']

                adjusted_boxes = self.adjust_bounding_boxes(bounding_boxes, 700)

                # Save the adjusted bounding boxes as a shapefile
                self.save_adjusted_bounding_boxes_as_shapefile(adjusted_boxes, self.get_shapefile_path(filename))

        return total_trees



    def visualize_results(self, results):
        img = mpimg.imread(self.source_image)
        image_height, image_width, _ = img.shape

        # Get the bounding boxes from the results
        bounding_boxes = results['predictions']

        adjusted_boxes = self.adjust_bounding_boxes(bounding_boxes, image_height)

        # Save the adjusted bounding boxes as a shapefile
        self.save_adjusted_bounding_boxes_as_shapefile(adjusted_boxes, self.get_shapefile_path())

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

    def calculate_resolution(self, tiff_path):
        with rasterio.open(tiff_path) as src:
            # Get the pixel size in the x and y dimensions
            pixel_x_size, pixel_y_size = src.res

            # Calculate the root-mean-square of pixel sizes
            resolution = np.sqrt(pixel_x_size**2 + pixel_y_size**2)

        return resolution
    
if __name__ == '__main__':
    # Create an instance of the class
    my_model = DeepForestModel(None)

    # Load the saved model
    model_path = "my_model.pt"
    my_model.load_model(model_path)
    
    # Path to the TIFF file
    tiff_path = r"C:\Users\FS-AI\Downloads\1673590088754_Ortho.tif"
    
    # Process the TIFF file
    total_detected_trees = my_model.process_tiff_image(tiff_path)
    print(f"Total number of trees detected: {total_detected_trees}")
