import os
import geopandas as gpd
import pandas as pd
from PIL import Image
import rasterio
from pyproj import Transformer
import matplotlib.pyplot as plt

def merge_image_tiles(image_folder):
    """
    Merge image tiles into a single image.
    
    Args:
        image_folder (str): Directory containing image tiles.
    
    Returns:
        PIL.Image: Merged image.
    """
    image_files = sorted([f for f in os.listdir(image_folder) if f.startswith('tile_') and f.endswith('.jpg')])
    with Image.open(os.path.join(image_folder, image_files[0])) as img:
        tile_width, tile_height = img.size
        n_tiles_x = max([int(file.split('_')[2].split('.')[0]) for file in image_files]) + 1
        n_tiles_y = max([int(file.split('_')[1]) for file in image_files]) + 1

    merged_image = Image.new('RGB', (tile_width * n_tiles_x, tile_height * n_tiles_y))
    for image_file in image_files:
        i, j = [int(x) for x in image_file.split('_')[1:3]]
        with Image.open(os.path.join(image_folder, image_file)) as img:
            merged_image.paste(img, (j * tile_width, i * tile_height))
    return merged_image

def merge_shapefiles_adjusted(shapefile_folder, output_shapefile, img_width, img_height, tile_size=700):
    """
    Merge and adjust coordinates of shapefiles based on image dimensions.
    
    Args:
        shapefile_folder (str): Directory containing shapefiles.
        output_shapefile (str): Path for the output merged shapefile.
        img_width (int): Width of the corresponding image.
        img_height (int): Height of the corresponding image.
        tile_size (int): Size of each tile in pixels.
    
    Returns:
        geopandas.GeoDataFrame: Merged and adjusted shapefile.
    """
    shapefiles = sorted([f for f in os.listdir(shapefile_folder) if f.startswith('tile_') and f.endswith('.shp')])
    all_gdfs = []
    for shapefile in shapefiles:
        gdf = gpd.read_file(os.path.join(shapefile_folder, shapefile))
        i, j = [int(x) for x in shapefile.split('_')[1:3]]
        gdf.geometry = gdf.geometry.translate(xoff=j * tile_size, yoff=(img_height - (i + 1) * tile_size))
        all_gdfs.append(gdf)
    combined_gdf = gpd.GeoDataFrame(pd.concat(all_gdfs, ignore_index=True))
    combined_gdf.to_file(output_shapefile)
    return combined_gdf

def get_image_bounds_in_latlon(tiff_path):
    """
    Get the geographic bounds of a TIFF image in latitude and longitude.
    
    Args:
        tiff_path (str): Path to the TIFF file.
    
    Returns:
        List: Geographic bounds [min_longitude, min_latitude, max_longitude, max_latitude].
    """
    with rasterio.open(tiff_path) as src:
        transformer = Transformer.from_crs(src.crs, 'EPSG:4326')
        return list(transformer.transform_bounds(*src.bounds))

def adjust_coordinates_to_fit_image(gdf, image_bounds):
    """
    Adjust coordinates of a GeoDataFrame to fit inside given image bounds.
    
    Args:
        gdf (geopandas.GeoDataFrame): GeoDataFrame to be adjusted.
        image_bounds (List): Bounds of the image [min_lat, min_lon, max_lat, max_lon].
    
    Returns:
        geopandas.GeoDataFrame: Adjusted GeoDataFrame.
    """
    x_min, y_min, x_max, y_max = gdf.total_bounds
    x_scale = (image_bounds[2] - image_bounds[0]) / (x_max - x_min)
    y_scale = (image_bounds[3] - image_bounds[1]) / (y_max - y_min)
    gdf.geometry = gdf.geometry.scale(xfact=x_scale, yfact=y_scale, origin=(x_min, y_min))
    gdf.geometry = gdf.geometry.translate(xoff=image_bounds[0] - x_min, yoff=image_bounds[1] - y_min)
    return gdf

# Main execution block
if __name__ == '__main__':
    image_folder = r"C:\Users\FS-AI\Desktop\try_kesowa\tiff_tiles_3gb"
    shapefile_folder = r"C:\Users\FS-AI\Desktop\try_kesowa\1gb_part_shape1"

    merged_image = merge_image_tiles(image_folder)
    img_width, img_height = merged_image.size
    merged_image.save(os.path.join(image_folder, "merged_image.jpg"))

    output_shapefile = r"C:\Users\FS-AI\Desktop\try_kesowa\1gb_part_shape1\tanmay.shp"
    merged_gdf = merge_shapefiles_adjusted(shapefile_folder, output_shapefile, img_width, img_height)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(merged_image)
    merged_gdf.plot(ax=ax, facecolor="none", edgecolor='red', linewidth=1.5)
    plt.savefig(os.path.join(image_folder, "overlay_image.jpg"), bbox_inches='tight', pad_inches=0.1)

    tiff_path = r"C:\Users\FS-AI\Downloads\Ortho_25cm.tif"
    bounds = get_image_bounds_in_latlon(tiff_path)
    image_bounds = [bounds[1], bounds[0], bounds[3], bounds[2]]

    shapefile_path = r"C:\Users\FS-AI\Desktop\try_kesowa\1gb_part_shape1\tanmay.shp"
    gdf = gpd.read_file(shapefile_path)
    adjusted_gdf = adjust_coordinates_to_fit_image(gdf, image_bounds)
    adjusted_gdf.to_file(r"C:\Users\FS-AI\Desktop\try_kesowa\1gb_part_shape1\adjusted.geojson", driver="GeoJSON")
