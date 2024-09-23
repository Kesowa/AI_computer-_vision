import os
import json
import torch
import rasterio
from rasterio.windows import Window
from pyproj import Proj, transform
from shapely.geometry import Polygon
from deepforest import main

class DeepForestModel:
    def __init__(self, source_image, slice_size=(700, 700)):
        # Instantiate a new model object
        self.model = main.deepforest()
        self.model.use_release()
        self.num_trees = 0  # Number of trees variable

        self.source_image = source_image
        self.slice_size = slice_size

        self.test_file = "data.csv"
        self.save_dir = r"C:\Users\FS-AI\Desktop\try_kesowa\hi"  # Path to directory where you want to save output shape file.
        self.destination_folder = self.save_dir

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.output_geojson_path = r"C:\Users\FS-AI\Desktop\try_kesowa\king\hi\output_detections_forarreycolony.geojson"
        
        # Open the source image using rasterio to get its CRS and transform matrix
        with rasterio.open(self.source_image) as src:
            self.transform_matrix = src.transform  # Get the geographic transform matrix
            self.src_crs = src.crs  # Get the CRS of the image (e.g., UTM)
            self.image_height = src.height
            self.image_width = src.width
            self.band_count = src.count  # The number of bands in the image

        # Define the WGS84 projection (lat/lon)
        self.wgs84_proj = Proj(proj='latlong', datum='WGS84')

    def load_model(self, model_path):
        # Load the saved state dictionary into the new model object
        state_dict = torch.load(model_path)
        self.model.load_state_dict(state_dict)

    def evaluate_slice(self, slice_image, save_dir):
        # Ensure the slice has 3 channels (convert RGBA to RGB if necessary)
        if slice_image.shape[2] == 4:
            slice_image = slice_image[:, :, :3]  # Strip alpha channel if it exists
        
        # Evaluate model on the image slice
        predictions = self.model.predict_image(image=slice_image, return_plot=False)

        # Ensure that predictions are not None or empty
        if predictions is None or predictions.empty:
            print(f"No predictions for the current slice.")
            return None  # Return None to avoid further processing
        
        return predictions


    def save_geojson(self, geojson_data, output_path):
        # Save the GeoJSON data to a file
        with open(output_path, "w") as geojson_file:
            json.dump(geojson_data, geojson_file, indent=2)

    def process_image_with_sliding_window(self):
        geojson_features = []

        # Open the TIFF file using rasterio
        with rasterio.open(self.source_image) as src:
            # Iterate over the image in a sliding window fashion
            for y in range(0, self.image_height, self.slice_size[1]):
                for x in range(0, self.image_width, self.slice_size[0]):
                    x1, y1 = x, y
                    x2 = min(x + self.slice_size[0], self.image_width)
                    y2 = min(y + self.slice_size[1], self.image_height)

                    # Read a window of the image (slice) using rasterio
                    window = Window(x1, y1, x2 - x1, y2 - y1)
                    slice_image = src.read(window=window)

                    # Transpose the image to match (height, width, channels)
                    slice_image = slice_image.transpose(1, 2, 0)

                    # Evaluate the model on this slice
                    predictions = self.evaluate_slice(slice_image, self.save_dir)

                    # Adjust bounding boxes for the slice
                    adjusted_boxes = self.adjust_bounding_boxes(predictions, y1, x1, slice_image.shape[0])

                    # Convert adjusted boxes to GeoJSON features
                    for box in adjusted_boxes:
                        feature = {
                            "type": "Feature",
                            "properties": {
                                "class_id": "Tree",
                                "confidence": None
                            },
                            "geometry": {
                                "type": "Polygon",
                                "coordinates": [[
                                    [box[0][0], box[0][1]],
                                    [box[1][0], box[1][1]],
                                    [box[2][0], box[2][1]],
                                    [box[3][0], box[3][1]],
                                    [box[0][0], box[0][1]]
                                ]]
                            }
                        }
                        geojson_features.append(feature)

        # Create the GeoJSON structure
        geojson_data = {
            "type": "FeatureCollection",
            "features": geojson_features
        }

        # Save the GeoJSON file
        self.save_geojson(geojson_data, self.output_geojson_path)

    def pixel_to_geo_coordinates(self, x_pixel, y_pixel):
        # Convert pixel coordinates to projected coordinates (e.g., UTM)
        x_proj, y_proj = rasterio.transform.xy(self.transform_matrix, y_pixel, x_pixel)
        
        # Convert projected coordinates to lat/lon (WGS84)
        lon, lat = transform(Proj(self.src_crs), self.wgs84_proj, x_proj, y_proj)
        return lon, lat

    def adjust_bounding_boxes(self, bounding_boxes, offset_y, offset_x, slice_height):
        if bounding_boxes is None:
            return []  # Return an empty list if there are no predictions
        
        adjusted_boxes = []
        for _, row in bounding_boxes.iterrows():
            xmin = row['xmin'] + offset_x
            ymin = row['ymin'] + offset_y
            xmax = row['xmax'] + offset_x
            ymax = row['ymax'] + offset_y

            # Convert pixel coordinates to geographic coordinates (lat/lon)
            top_left = self.pixel_to_geo_coordinates(xmin, ymin)
            bottom_right = self.pixel_to_geo_coordinates(xmax, ymax)

            # Create the bounding box coordinates as a polygon in geographic coordinates
            polygon = [top_left, 
                    self.pixel_to_geo_coordinates(xmax, ymin), 
                    bottom_right, 
                    self.pixel_to_geo_coordinates(xmin, ymax), 
                    top_left]  # Close the polygon
            adjusted_boxes.append(polygon)
        
        return adjusted_boxes


    def process_and_evaluate_with_sliding_window(self):
        # Process the image using a sliding window
        self.process_image_with_sliding_window()

        print(f"GeoJSON file saved at: {self.output_geojson_path}")

# Usage example
if __name__ == '__main__':
    # Get the source image path from the user
    source_image = r"C:\Users\FS-AI\Downloads\AareyColonyOrthomosaic_4Band_2.7cm.tif"

    # Create an instance of the class with the provided source image
    my_model = DeepForestModel(source_image)

    # Load the saved model
    model_path = "my_model.pt"
    my_model.load_model(model_path)

    # Process the image and evaluate the model using sliding window
    my_model.process_and_evaluate_with_sliding_window()
