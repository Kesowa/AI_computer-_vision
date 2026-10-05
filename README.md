# Computer Vision Research

A collection of computer-vision experiments and prototypes built at Kesowa
between 2021 and 2025, mostly aimed at extracting information from drone and
aerial imagery. Each top-level directory is an independent project with its own
dependencies.

This is **research code, published for reference**. It is not a library, there
is no common interface, and the projects vary in completeness. Some were
prototypes that answered a question and stopped there.

> **Model weights in this repository are not MIT licensed.** Several derive from
> third-party pretrained models whose terms still apply, and at least one of
> those is restricted to non-commercial research. Read [NOTICE](NOTICE) before
> using any `.h5`, `.onnx`, `.pt` or `.pkl` file here.

## Projects

| Directory | What it does |
| --- | --- |
| `Roof_top_segmentation/` | U-Net segmentation of building rooftops from aerial rasters, exporting polygons as Shapefiles for QGIS/ArcGIS. The most developed project here, with both TensorFlow and ONNX inference paths. |
| `Tree_detection/` | Tree-crown detection with DeepForest, including custom training, sliding-window inference over large rasters, and Shapefile export. |
| `Tree Count/` | Lighter tree-crown counting using pretrained DeepForest against an image plus shapefile. |
| `Face_recognition/` | VGG-Face embedding and classification pipeline. **Ships no dataset** — see the note below. |
| `Violence_detection/` | Keras video classifier for violent/non-violent scenes, with a TensorFlow-to-ONNX conversion path. |
| `people_count/` | Counting people in video with YOLOv3 / YOLOv4-tiny, sampling every fourth frame with threshold-triggered snapshots. |
| `construction_material/` | YOLO-based detection of construction materials in site imagery. |
| `Map Digitization/` | Vectorising scanned maps: polygon extraction, broken-line repair, band inspection, SVG-to-GeoTIFF conversion. |
| `Bengali_text_detection/` | Bengali text detection and translation using the Google Vision OCR API. |
| `server.py`, `contract.py` | A sketch of the FastAPI serving interface these models were meant to sit behind. `contract.py` is a specification, not runnable code. |

## Running any of this

There is no top-level install. Each project has its own requirements file, and
some have their own README with more detail:

```bash
cd Roof_top_segmentation
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt     # name varies: requirement.txt, Requirement.txt
```

Expect friction. These were written against the library versions of their day,
and the requirements files are mostly unpinned. In particular:

- The TensorFlow and Keras projects predate Keras 3 and need TensorFlow 2.x.
- Several scripts contain **hardcoded absolute paths** from the machines they
  were written on, like `C:\Users\FS-AI\...`. You will need to edit these.
- `Bengali_text_detection` needs your own Google Cloud Vision API key, which
  the script reads from a placeholder, not from the environment.
- `Tree_detection/fetch_from_api.py` points at an internal Kesowa COG endpoint
  that is not publicly reachable. Substitute your own raster source.

## A note on Face_recognition

The code is here; the data is not, deliberately.

Earlier versions of this repository included sample photographs of identifiable
individuals. Those images were removed from the full git history in 2026,
because publishing people's faces under an open licence is not something Kesowa
could consent to on their behalf.

If you use this code, supply your own images and make sure you have a lawful
basis for processing them. Facial images are sensitive personal data under the
Indian DPDP Act 2023 and the GDPR, and consent for a research prototype is not
consent for public redistribution. See [NOTICE](NOTICE).

## Branches

`master` is the main line. `backup` (2023) and `map_digitizer` (2024) are older
snapshots, kept for reference.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). This is an archive of past research
rather than an actively developed project, so expect slow review. Security
reports go through [SECURITY.md](SECURITY.md).

## License

Source code written by Kesowa is released under the [MIT License](LICENSE).
**Model weights and third-party code are not covered** — read
[NOTICE](NOTICE) for the provenance of each.
