import rasterio
import numpy as np
import cv2
import vtracer as vt
import matplotlib.pyplot as plt
import tempfile
import os
import json
from shapely.geometry import box, mapping

def extract_bounding_boxes(
    image_path,
    output_image_path,
    svg_output_path,
    geojson_output_path
):
    # -------------------------------------------
    # 1. READ TIFF & MORPHOLOGICAL CLEANING
    # -------------------------------------------
    with rasterio.open(image_path) as src:
        data = src.read(1)   # Rooftop segmentation band
        transform = src.transform  # For pixel-to-map coordinate transformation
        raster_crs = src.crs

    # Convert to binary image
    binary_image = (data > 0).astype(np.uint8) * 255

    # Morphological cleaning
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)

    # -------------------------------------------------
    # 2. OPTIONAL: DOWNSCALE FOR VTRACER
    # -------------------------------------------------
    downscale_factor = 0.5  # Adjust as needed
    resized_image = cv2.resize(
        cleaned_image,
        None,
        fx=downscale_factor,
        fy=downscale_factor,
        interpolation=cv2.INTER_AREA
    )

    # Vectorize via vtracer (optional)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        cv2.imwrite(tmp.name, resized_image)
        tmp_path = tmp.name
    try:
        vt.convert_image_to_svg_py(
            tmp_path,
            svg_output_path,
            colormode="binary",
            hierarchical="cutout",
            mode="polygon",
            filter_speckle=8,
        )
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    # -------------------------------------------
    # 3. EXTRACT CONTOURS -> BOUNDING BOXES
    # -------------------------------------------
    contours, _ = cv2.findContours(
        cleaned_image,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # Create a list to store bounding-box geometries
    bbox_geometries = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # Filter out trivial boxes
        if w > 100 and h > 100:
            # Pixel-based bounding box corners
            row_min, col_min = y, x
            row_max, col_max = y + h, x + w

            # Convert pixel (row, col) -> map coordinates
            minx, maxy = rasterio.transform.xy(transform, row_min, col_min)
            maxx, miny = rasterio.transform.xy(transform, row_max, col_max)

            # Create shapely polygon
            geom = box(minx, miny, maxx, maxy)
            bbox_geometries.append(geom)

    # ------------------------------------------------------
    # 4. FILTER OUT LARGER BOXES THAT CONTAIN SMALLER ONES
    # ------------------------------------------------------
    # This function removes any box that fully contains one (or more) smaller boxes.
    # If you only want to remove boxes that contain >= 2 smaller boxes, change the threshold.
    def filter_containing_boxes(geoms, threshold=1):
        to_remove = set()
        n = len(geoms)

        for i in range(n):
            big_box = geoms[i]
            count_contained = 0

            for j in range(n):
                if i == j:
                    continue
                small_box = geoms[j]

                # If big_box fully contains small_box and has larger area
                if big_box.contains(small_box) and (big_box.area > small_box.area):
                    count_contained += 1

            # If big_box contains at least 'threshold' smaller boxes, remove it
            if count_contained >= threshold:
                to_remove.add(i)

        # Build the final list
        return [g for idx, g in enumerate(geoms) if idx not in to_remove]

    filtered_bboxes = filter_containing_boxes(bbox_geometries, threshold=1)

    # -------------------------------------------
    # 5. CREATE GEOJSON + VISUALIZATION
    # -------------------------------------------
    # Prepare list of features
    features = []
    for geom in filtered_bboxes:
        # Convert shapely box back to bounding box corners for output_image
        # minx, miny, maxx, maxy = geom.bounds
        # Because QGIS uses top-left (maxy) vs. bottom-right (miny),
        # you might need to invert Y for drawn bounding boxes in the PNG.
        # We'll just store it in GeoJSON. The PNG is purely for a rough check.

        feature = {
            "type": "Feature",
            "geometry": mapping(geom),
            "properties": {}
        }
        features.append(feature)

    geojson_data = {
        "type": "FeatureCollection",
        "features": features,
        "crs": {
            "type": "name",
            "properties": {
                "name": str(raster_crs)  # e.g. "EPSG:4326"
            }
        }
    }

    # Write the GeoJSON
    with open(geojson_output_path, "w") as f:
        json.dump(geojson_data, f, indent=2)

    # OPTIONAL: Draw bounding boxes on the output image for debugging.
    # We need to map from geometry -> pixel coords. Let's do that:
    debug_img = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)

    for geom in filtered_bboxes:
        minx, miny, maxx, maxy = geom.bounds
        # Convert map coords back to row, col
        # row_min, col_min
        row_min, col_min = rasterio.transform.rowcol(transform, minx, maxy)
        row_max, col_max = rasterio.transform.rowcol(transform, maxx, miny)

        # Draw it. Ensure we don't go out of image bounds, just in case
        row_min = max(row_min, 0)
        col_min = max(col_min, 0)
        row_max = min(row_max, data.shape[0] - 1)
        col_max = min(col_max, data.shape[1] - 1)

        # rectangle expects (x1,y1) -> (x2,y2) => (col_min, row_min) -> (col_max, row_max)
        cv2.rectangle(debug_img, (col_min, row_min), (col_max, row_max), (0, 255, 0), 2)

    cv2.imwrite(output_image_path, debug_img)

    # Show the debug image
    plt.figure(figsize=(8, 6))
    plt.imshow(cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB))
    plt.title("Filtered Bounding Boxes on Segmented Rooftops")
    plt.axis("off")
    plt.show()


# Example usage
if __name__ == "__main__":
    extract_bounding_boxes(
        image_path="sliding150thr.tif ",
        output_image_path="output_with_boxes.png",
        svg_output_path="output_with_boxes.svg",
        geojson_output_path="output_with_boxes.geojson"
    )
