import os
import geopandas as gpd
from shapely.geometry import Polygon
import numpy as np
from PIL import Image
import rasterio
from rasterio.windows import Window
from deepforest import main
import torch

class DeepForestModel:
    counter = 0

    def __init__(self, source_image):
        self.model = main.deepforest()
        self.model.use_release()
        self.num_trees = 0
        self.source_image = source_image
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.test_file = "data.csv"
        DeepForestModel.counter += 1
        self.save_dir = os.path.abspath("output_current1")

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def process_tiff_image(self, tiff_path):
        with rasterio.open(tiff_path) as src:
            n_tiles_x = int(np.ceil(src.width / 700))
            n_tiles_y = int(np.ceil(src.height / 700))
            total_trees = 0

            for i in range(n_tiles_y):
                for j in range(n_tiles_x):
                    window = Window(j * 700, i * 700, 700, 700)
                    tile = src.read(window=window)

                    tile_image = Image.fromarray(np.transpose(tile, (1, 2, 0)))
                    tile_path = os.path.join(self.save_dir, f"tile_{i}_{j}.png")
                    tile_image.save(tile_path)

                    self.source_image = tile_path
                    results, num_trees = self.process_and_evaluate()
                    total_trees += num_trees
                    bounding_boxes = results['predictions']

                    adjusted_boxes = self.adjust_bounding_boxes(bounding_boxes, 700)

                    self.save_adjusted_bounding_boxes_as_shapefile(adjusted_boxes,
                                                                   self.get_shapefile_path(f"tile_{i}_{j}.png"))

                    os.remove(tile_path)

            return total_trees

    def get_shapefile_path(self, image_filename):
        base_name = os.path.splitext(image_filename)[0]
        shapefile_name = f"{base_name}.shp"
        path = os.path.join(self.save_dir, shapefile_name)
        return path

    def load_model(self, model_path):
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)

    def evaluate(self, test_file, save_dir):
        results = self.model.evaluate(test_file, os.path.dirname(test_file), iou_threshold=0.4, savedir=save_dir)
        self.num_trees = len(results['predictions'])
        return results

    def save_adjusted_bounding_boxes_as_shapefile(self, adjusted_boxes, output_path):
        adjusted_gdf = gpd.GeoDataFrame({'geometry': adjusted_boxes})
        adjusted_gdf.to_file(output_path)

    def adjust_bounding_boxes(self, bounding_boxes, image_height):
        adjusted_boxes = []
        for index, row in bounding_boxes.iterrows():
            xmin, ymin, xmax, ymax = row['xmin'], row['ymin'], row['xmax'], row['ymax']
            ymin = image_height - ymin
            ymax = image_height - ymax
            polygon = Polygon([(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin)])
            adjusted_boxes.append(polygon)
        return adjusted_boxes

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
                self.source_image = os.path.join(image_folder, filename)
                results, num_trees = self.process_and_evaluate()
                total_trees += num_trees
                bounding_boxes = results['predictions']

                adjusted_boxes = self.adjust_bounding_boxes(bounding_boxes, 700)

                self.save_adjusted_bounding_boxes_as_shapefile(adjusted_boxes, self.get_shapefile_path(filename))

        return total_trees

    def calculate_resolution(self, tiff_path):
        with rasterio.open(tiff_path) as src:
            pixel_x_size, pixel_y_size = src.res
            resolution = np.sqrt(pixel_x_size ** 2 + pixel_y_size ** 2)

        return resolution


if __name__ == '__main__':
    my_model = DeepForestModel(None)
    model_path = "my_model.pt"
    my_model.load_model(model_path)

    tiff_path = "path/to/your/tiff/file.tif"  # Update with the actual path
    total_detected_trees = my_model.process_tiff_image(tiff_path)
    print(f"Total number of trees detected: {total_detected_trees}")
