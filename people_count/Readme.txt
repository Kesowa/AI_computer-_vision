## Readme:

This script performs object detection on a video file using the YOLOv3 or YOLOv4 Tiny model. It processes every 4th frame of the video and saves frames with detected objects to an output video file. When the number of detected people exceeds certain thresholds (6, 15, or 30), it saves a snapshot of the current frame and pauses for 5 seconds. At the end, it plots the "garbage union" variable.

## Dependencies:

- Python 3.7 or higher
- OpenCV
- Numpy
- TensorFlow 2.4 or higher
- YOLOv3 or YOLOv4 Tiny model
- Corresponding weights file
- A video file to process

## Installation:

Before you run the script, make sure you have the necessary dependencies installed. If not, you can install them using pip:

```sh
pip install opencv-python
pip install numpy
pip install tensorflow
```

## YOLO Model and Weights:

You need to have the YOLOv3 or YOLOv4 Tiny model and the corresponding weights file. If you don't have them, you can download them from the official YOLO website or a trusted source. Make sure to place these files in the appropriate directories, as specified in the script (checkpoints/Darknet_53_yolov3/yolov3_custom_people_garbage for YOLOv3, checkpoints/tiny_yolov4/yolov4_tiny_custom_Tiny for YOLOv4 Tiny).

## Video File:

You need a video file to process. This should be a .mp4 file. Adjust the `VIDEO_URL` variable in the script to point to your video file. 

## Output:

The processed video will be saved to the path specified in the `output_path` variable in the script. Additionally, the script will save "snapshots" as PNG files in the current directory whenever the detected count of people crosses certain thresholds.

## Usage:

- Open the script in a text editor.
- Adjust the `USE_TINY_YOLOV4`, `VIDEO_URL`, `output_path`, `input_size`, `CLASSES`, `score_threshold`, `iou_threshold`, and `rectangle_colors` variables as needed.
- Save and close the script.
- Run the script in a terminal with `python script_name.py`.

## Notes:

This script assumes that your video file is in the .mp4 format, and that you have the necessary read/write permissions for the directories you're working with. If you encounter any errors, check that these assumptions are met and that all of the file paths in the script are correct.
