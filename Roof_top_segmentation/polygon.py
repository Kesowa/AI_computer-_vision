import os
import cv2
import json
import numpy as np
import rasterio
from rasterio.transform import xy
from shapely.geometry import box, mapping
from shapely.ops import transform as shp_transform
from pyproj import Transformer, CRS
import matplotlib.pyplot as plt

# ------------------------------
# MAIN FUNCTION
# ------------------------------
def extract_bounding_boxes_from_png(
    png_mask_path,
    reference_tif_path,
    output_image_path,
    geojson_output_path,
    reproject_to_epsg="EPSG:4326",
    min_size_px=100,
    tile_size=4096,
    debug=True
):
    print("Starting bounding box extraction from PNG mask...")

    # ------------------------------
    # LOAD PNG MASK
    # ------------------------------
    mask = cv2.imread(png_mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Failed to read PNG: {png_mask_path}")

    H, W = mask.shape
    print(f"PNG size: {W} x {H}")

    # ------------------------------
    # LOAD GEOREFERENCE FROM TIFF
    # ------------------------------
    with rasterio.open(reference_tif_path) as ref:
        transform = ref.transform
        raster_crs = ref.crs
        if transform is None or raster_crs is None:
            raise ValueError("Reference TIFF has no valid georeferencing")

    # ------------------------------
    # TILE-BASED CONTOUR EXTRACTION
    # ------------------------------
    all_contours = []
    total_tiles = ((H + tile_size - 1) // tile_size) * ((W + tile_size - 1) // tile_size)
    processed = 0

    for y in range(0, H, tile_size):
        for x in range(0, W, tile_size):
            tile = mask[y:y+tile_size, x:x+tile_size]
            binary = (tile > 0).astype(np.uint8) * 255

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

            contours, _ = cv2.findContours(
                cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            for cnt in contours:
                cnt[:, 0, 0] += x
                cnt[:, 0, 1] += y
                all_contours.append(cnt)

            processed += 1
            if debug and processed % 10 == 0:
                print(f"Processed {processed}/{total_tiles} tiles, contours: {len(all_contours)}")

    print(f"Total contours found: {len(all_contours)}")

    # ------------------------------
    # CONVERT CONTOURS → GEO BBOXES
    # ------------------------------
    bbox_geoms = []

    for cnt in all_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < min_size_px or h < min_size_px:
            continue

        row_min, col_min = y, x
        row_max, col_max = y + h, x + w

        minx, maxy = xy(transform, row_min, col_min, offset="ul")
        maxx, miny = xy(transform, row_max - 1, col_max - 1, offset="lr")

        bbox_geoms.append(box(minx, miny, maxx, maxy))

    print(f"Bounding boxes after size filter: {len(bbox_geoms)}")

    # ------------------------------
    # REMOVE CONTAINED BOXES
    # ------------------------------
    def filter_containing_boxes(geoms):
        keep = []
        for i, g in enumerate(geoms):
            contained = False
            for j, other in enumerate(geoms):
                if i != j and other.contains(g) and other.area > g.area:
                    contained = True
                    break
            if not contained:
                keep.append(g)
        return keep

    bbox_geoms = filter_containing_boxes(bbox_geoms)
    print(f"Bounding boxes after containment filter: {len(bbox_geoms)}")

    # ------------------------------
    # REPROJECT
    # ------------------------------
    target_crs = CRS.from_user_input(reproject_to_epsg)
    if raster_crs != target_crs:
        transformer = Transformer.from_crs(raster_crs, target_crs, always_xy=True)
        bbox_geoms = [
            shp_transform(lambda x, y: transformer.transform(x, y), g)
            for g in bbox_geoms
        ]

    # ------------------------------
    # SAVE GEOJSON
    # ------------------------------
    geojson = {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "geometry": mapping(g), "properties": {}}
            for g in bbox_geoms
        ]
    }

    with open(geojson_output_path, "w") as f:
        json.dump(geojson, f, indent=2)

    print(f"GeoJSON saved to {geojson_output_path}")

    # ------------------------------
    # DEBUG VISUALIZATION
    # ------------------------------
    preview = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    for g in bbox_geoms:
        minx, miny, maxx, maxy = g.bounds
        rows, cols = rasterio.transform.rowcol(transform, [minx, maxx], [maxy, miny])
        cv2.rectangle(
            preview,
            (cols[0], rows[0]),
            (cols[1], rows[1]),
            (0, 255, 0),
            2
        )

    cv2.imwrite(output_image_path, preview)
    print(f"Debug image saved to {output_image_path}")

    plt.figure(figsize=(10, 8))
    plt.imshow(preview[:, :, ::-1])
    plt.title(f"Detected Bounding Boxes ({len(bbox_geoms)})")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    print("\nSummary:")
    print(f"- PNG mask: {png_mask_path}")
    print(f"- Boxes: {len(bbox_geoms)}")
    print(f"- GeoJSON: {geojson_output_path}")
    print(f"- Debug image: {output_image_path}")

# ------------------------------
# RUN
# ------------------------------
if __name__ == "__main__":
    extract_bounding_boxes_from_png(
        png_mask_path="/home/debian/Downloads/tree_detection/rooftop_output_onnx_thrauto.png",
        reference_tif_path="/home/debian/Downloads/tree_detection/new_test/592568_ortho_Kodiwaka_COG.tif",
        output_image_path="ortho_png_debug.png",
        geojson_output_path="ortho_png_boxes.geojson",
        reproject_to_epsg="EPSG:4326",
        min_size_px=20,
        tile_size=4096,
        debug=True
    )
