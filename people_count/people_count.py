import cv2
import numpy as np
import tensorflow as tf
import time
from datetime import datetime
from yolov3.yolov3 import Create_Yolov3
from yolov3.utils import *
from yolov3.configs import *

# Set parameters
USE_TINY_YOLOV4 = False
VIDEO_URL = 'C:\\Users\\FS-AI\\Downloads\\processed_footage\\output_0043.mp4'
output_path = 'C:\\Algae_garbage\\detected_people_count\\detected_yolov3_0043_heheh.mp4'
input_size=YOLO_INPUT_SIZE
CLASSES=TRAIN_CLASSES
score_threshold=0.3
iou_threshold=0.45
rectangle_colors=''
save_path = "saved_frames/"

# Create YOLO model and load weights
if USE_TINY_YOLOV4:
    TRAIN_CLASSES = "./model_data/class.txt"
    YOLO_INPUT_SIZE = 480
    TRAIN_INPUT_SIZE = 480
    TEST_INPUT_SIZE = 480
    TRAIN_YOLO_TINY = True
    YOLO_TYPE = 'yolov4'
    yolo = Create_Yolo(input_size=YOLO_INPUT_SIZE, channels=3, training=False, CLASSES=TRAIN_CLASSES)
    yolo.load_weights('checkpoints/tiny_yolov4/yolov4_tiny_custom_Tiny')
else:
    Yolo = Create_Yolov3(input_size = YOLO_INPUT_SIZE, CLASSES = TRAIN_CLASSES)
    Yolo.load_weights("checkpoints/Darknet_53_yolov3/yolov3_custom_people_garbage")

# Initialize video capture and writer
cap = cv2.VideoCapture(VIDEO_URL)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
codec = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_path, codec, 10, (width, height))

if not cap.isOpened(): 
    raise IOError("Cannot connect to drone feed.")

# Initialize counters and settings
count = 0
snapshot_count = 1
last_msg = 0.0

while True:
    # Read frame
    ret, frame = cap.read()

    # If frame could not be retrieved, break loop
    if not ret:
        break

    # Only process every 4th frame
    if count % 4 == 0:
        start_time = time.time()

        # Convert image to RGB
        original_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Preprocess image
        image_data = image_preprocess(np.copy(original_image), [input_size, input_size])
        image_data = image_data[np.newaxis, ...].astype(np.float32)

        # Perform object detection
        pred_bbox = Yolo.predict(image_data)

        # Postprocess bounding boxes
        pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
        pred_bbox = tf.concat(pred_bbox, axis=0)
        bboxes = postprocess_boxes(pred_bbox, original_image, input_size, score_threshold)

        # Perform non-max suppression and draw bounding boxes
        bboxes = nms(bboxes, iou_threshold, method='nms')
        image, people_count, garbage_union = draw_bbox(original_image, bboxes, CLASSES=CLASSES, rectangle_colors=rectangle_colors)

        # Add text overlays and write frame to output video```python
        res_frame = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)
        fps = 1.0 / (time.time() - start_time)
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        cv2.putText(res_frame, "FPS: {:.1f}".format(fps), (0,20),0,1,(255,0,0),4)
        cv2.putText(res_frame, "Time: "+str(current_time), (0,50),0,1,(0,255, 0),4)
        cv2.putText(res_frame, "People count: "+str(people_count), (0,80),0,1,(255,0,0),4)
        out.write(res_frame)

        # Create snapshots based on people count
        if people_count>=30 and last_msg<30:
            last_msg = 30
            print("High crowd____________________________________________")
            cv2.putText(res_frame, "High crowd", (0,130),0,1,(0,0,255),4)
            cv2.imwrite('snapshot'+str(snapshot_count)+'.png', res_frame)
            snapshot_count+=1
            time.sleep(5)
        elif (people_count<=22 and last_msg>22) or (people_count>=15 and last_msg<15):
            last_msg = 15
            print("Moderate crowd______________________________________________")
            cv2.putText(res_frame, "Moderate crowd", (0,130),0,1,(0,0,255),4)					
            cv2.imwrite('snapshot'+str(snapshot_count)+'.png', res_frame)
            snapshot_count+=1
            time.sleep(5)
        elif people_count<=6 and last_msg>6:
            last_msg = 6
            print("Low crowd______________________________________________")
            cv2.putText(res_frame, "Low crowd", (0,130),0,1,(0,0,255),4)			
            cv2.imwrite('snapshot'+str(snapshot_count)+'.png', res_frame)
            snapshot_count+=1
            time.sleep(5)

    # Increment frame counter
    count += 1

# Release video capture and writer
cap.release()
out.release()
cv2.destroyAllWindows()

print(garbage_union.plot(color = 'red'))
plt.show()

