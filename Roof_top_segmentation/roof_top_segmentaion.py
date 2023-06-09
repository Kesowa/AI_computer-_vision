from rasterio.features import shapes
import geopandas as gpd
from shapely.geometry import shape, Point, Polygon, MultiPolygon
import rasterio
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import img_as_ubyte

class Segmentor:
    def _init_(self, model_path, image_path):
        self.IMG_HEIGHT = 512
        self.IMG_WIDTH  = 512
        self.IMG_CHANNELS = 1
        self.n_classes = 2
        self.model_path = model_path
        self.image_path = image_path

        # Load the model
        self.model = self.get_model()
        self.model.load_weights(model_path)

        # Load the image with rasterio to get its transform
        with rasterio.open(image_path) as src:
            self.img_transform = src.transform
            self.img_crs = src.crs
            self.original_height = src.height
            self.original_width = src.width



    def get_model(self):
        return multi_unet_model(n_classes=self.n_classes, IMG_HEIGHT=self.IMG_HEIGHT, 
                                IMG_WIDTH=self.IMG_WIDTH, IMG_CHANNELS=self.IMG_CHANNELS)
    
    def predict(self):
        # Load and preprocess the image
        img = cv2.imread(self.image_path, 0)
        img = cv2.resize(img, (self.IMG_WIDTH, self.IMG_HEIGHT))
        img = np.expand_dims(img, axis=2)
        img_input = np.expand_dims(img, 0)
        
        # Normalize the image
        img_input = normalize(img_input, axis=1)
        
        # Perform prediction
        prediction = (self.model.predict(img_input))
        predicted_img = np.argmax(prediction, axis=3)[0,:,:]
        return predicted_img


    def plot_prediction(self, original_image, predicted_img):
        # Plot the original image and the predicted image
        plt.figure(figsize=(12, 8))
        
        plt.subplot(121)
        plt.title('Original Image')
        plt.imshow(original_image, cmap='gray')

        plt.subplot(122)
        plt.title('Prediction on test image')
        plt.imshow(predicted_img, cmap='jet')
        plt.show()    

    def vectorize(self, prediction):
        # Convert prediction to uint8 and get shapes
        mask = prediction.astype('uint8')

        # Write the prediction back to a raster with the same size and transform as the original image
        with rasterio.open('prediction.tif', 'w', driver='GTiff', height=self.original_height, width=self.original_width, count=1, dtype='uint8', crs=self.img_crs, transform=self.img_transform) as dst:
            dst.write(mask, 1)

        # Read the prediction with rasterio and get the shapes
        with rasterio.open('prediction.tif') as src:
            shapes = src.read(1)
            results = (
                {'properties': {'raster_val': v}, 'geometry': s}
                for i, (s, v) 
                in enumerate(rasterio.features.shapes(shapes, transform=src.transform)) if v == 1
            )

        # Convert shapes to GeoDataFrame
        geoms = list(results)
        gdf = gpd.GeoDataFrame.from_features(geoms)
        
        # Function to invert y-coordinates
        def invert_y(geom):
            geom_type = geom.geom_type
            if geom_type == 'Polygon':
                return Polygon([(x, -y) for x, y in geom.exterior.coords])
            elif geom_type == 'Point':
                return Point(geom.x, -geom.y)
            elif geom_type == 'MultiPolygon':
                return MultiPolygon([invert_y(p) for p in geom])
            else:
                raise ValueError(f'Unsupported geometry type: {geom_type}')

        # Apply the function to the 'geometry' column
        gdf['geometry'] = gdf['geometry'].apply(invert_y)

        # Save the inverted GeoDataFrame to a new shapefile
        gdf.to_file("inverted_output.shp")
        gdf.plot()
        return gdf
