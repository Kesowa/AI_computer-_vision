Object Detection with YOLOv7
This script provides an implementation of object detection using the YOLOv7 model. It supports various functionalities like Non-Max Suppression (NMS), applying secondary classifiers, saving detections to text files, and visualizing detections.

Prerequisites-
Python 3.x
torch
cv2 (OpenCV)
numpy
How to Run
To run the object detection script with the provided pre-trained weights, use the following command:
!python C:\Users\FS-AI\Desktop\Research_Paper_Aastha\Construction_materials\yolov7\detect.py \
--weights "C:\Users\FS-AI\Desktop\Research_Paper_Aastha\Construction_materials\yolov7\best.pt" \
--img 704 --conf 0.0 \
--source "C:\Users\FS-AI\Desktop\Research_Paper_Aastha\Construction_materials\data\Val\images" \
--save-txt --iou-thres 0.0001 --save-conf
