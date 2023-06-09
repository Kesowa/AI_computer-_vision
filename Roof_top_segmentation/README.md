# Readme

## Image Segmentation using UNET architecture

This project provides a python class `Segmentor` for performing image segmentation tasks. It uses a trained model based on the UNET architecture to segment input images and generate polygonal annotations. These polygon annotations are saved in Shapefile format that can be directly used in GIS software like QGIS or ArcGIS.

### Requirements:

This project requires Python 3, and the dependencies include:

- keras
- rasterio
- geopandas
- cv2 (OpenCV)
- numpy
- matplotlib
- shapely
- skimage

You can install these packages using pip:
pip install keras rasterio geopandas opencv-python numpy matplotlib shapely skimage


### How to use the code:

1. Clone this repository to your local machine.

2. Ensure the necessary dependencies are installed.

3. Load the trained model weights and input the path of your image to the `Segmentor` class. Make sure to provide the correct paths.

python
segmentor = Segmentor('model_weights_60_epochs.h5', r"path\to\your\image.tif")


In this example, 'model_weights_60_epochs.h5' is the path to your trained model weights and 'path\to\your\image.tif' is the path to the image you want to segment.

4. Call the `predict` method to perform the segmentation on the image. This will return a 2D numpy array that represents the segmented image.

python
predicted_img = segmentor.predict()


5. You can visualize the original and segmented images using the `plot_prediction` method. This will display the original image alongside the predicted image segmentation.

python
original_image = cv2.imread(r"path\to\your\image.tif", 0)
segmentor.plot_prediction(original_image, predicted_img)


6. Finally, you can generate a Shapefile from the segmented image using the `vectorize` method. This will save the polygon features to a Shapefile and return a GeoPandas GeoDataFrame for further use in your Python script.

python
gdf = segmentor.vectorize(predicted_img)


This command generates a Shapefile named "inverted_output.shp" in the working directory. You can load this file into any GIS software for visualization or further analysis. The `vectorize` method also returns a GeoPandas GeoDataFrame object for additional processing or analysis within the Python environment.
