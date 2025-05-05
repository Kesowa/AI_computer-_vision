# bengali_ocr_to_english_pipeline.py

import cv2
import numpy as np
import base64
import requests
import json
import os
from imutils.object_detection import non_max_suppression
from transformers import MarianTokenizer
from optimum.onnxruntime import ORTModelForSeq2SeqLM

# ---------- STEP 1: Detect text regions using EAST ----------
def decode_predictions(scores, geometry, min_confidence):
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []
    for y in range(numRows):
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]
        for x in range(numCols):
            if scoresData[x] < min_confidence:
                continue
            (offsetX, offsetY) = (x * 4.0, y * 4.0)
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]
            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)
            rects.append((startX, startY, endX, endY))
            confidences.append(float(scoresData[x]))
    return (rects, confidences)


def detect_text_regions(image_path, east_model_path, conf_threshold=0.5):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    orig = image.copy()
    (H, W) = image.shape[:2]
    newW, newH = (320, 320)
    rW, rH = W / float(newW), H / float(newH)
    image = cv2.resize(image, (newW, newH))

    blob = cv2.dnn.blobFromImage(
        image, 1.0, (newW, newH), (123.68, 116.78, 103.94), swapRB=True, crop=False
    )
    net = cv2.dnn.readNet(east_model_path)
    net.setInput(blob)
    (scores, geometry) = net.forward([
        "feature_fusion/Conv_7/Sigmoid",
        "feature_fusion/concat_3"
    ])

    (rects, confidences) = decode_predictions(scores, geometry, conf_threshold)
    boxes = non_max_suppression(np.array(rects), probs=confidences)

    scaled_boxes = []
    yolo_boxes = []
    for (startX, startY, endX, endY) in boxes:
        sX = int(startX * rW)
        sY = int(startY * rH)
        eX = int(endX * rW)
        eY = int(endY * rH)
        scaled_boxes.append((sX, sY, eX, eY))

        # YOLO format conversion
        x_center = ((sX + eX) / 2) / W
        y_center = ((sY + eY) / 2) / H
        width = (eX - sX) / W
        height = (eY - sY) / H
        yolo_boxes.append((0, x_center, y_center, width, height))  # assuming class_id = 0

    return orig, scaled_boxes, yolo_boxes

# ---------- STEP 2: OCR each region with Google Vision API ----------
def detect_text_google_ocr_from_region(region, api_key):
    _, buf = cv2.imencode('.jpg', region)
    content = base64.b64encode(buf).decode('utf-8')
    url = f'https://vision.googleapis.com/v1/images:annotate?key={api_key}'
    headers = {'Content-Type': 'application/json'}
    payload = {
        "requests": [
            {
                "image": {"content": content},
                "features": [{"type": "TEXT_DETECTION"}],
                "imageContext": {"languageHints": ["bn"]}
            }
        ]
    }
    response = requests.post(url, headers=headers, data=json.dumps(payload))
    result = response.json()
    try:
        return result['responses'][0]['fullTextAnnotation']['text']
    except KeyError:
        return ""

# ---------- STEP 3: Translate using ONNX-based Hugging Face model ----------
def translate_bengali_to_english(texts, model_path="onnx_model"):
    tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-bn-en")
    model = ORTModelForSeq2SeqLM.from_pretrained(model_path)
    inputs = tokenizer(texts, return_tensors="pt", padding=True)
    outputs = model.generate(**inputs)
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)

# ---------- ENTRY POINT ----------
if __name__ == "__main__":
    image_path = "/home/tanmay/Downloads/bengali_Street.jpg"
    east_model_path = "frozen_east_text_detection.pb"
    api_key = "APIKEY"

    image, boxes, yolo_boxes = detect_text_regions(image_path, east_model_path, conf_threshold=0.5)
    print("Detected boxes (x1, y1, x2, y2):")
    for box in boxes:
        print(box)

    print("\nYOLO format bounding boxes:")
    for yolo in yolo_boxes:
        print(f"{yolo[0]} {yolo[1]:.6f} {yolo[2]:.6f} {yolo[3]:.6f} {yolo[4]:.6f}")

    results = []
    for (x1, y1, x2, y2) in boxes:
        region = image[y1:y2, x1:x2]
        text = detect_text_google_ocr_from_region(region, api_key)
        translation = ""
        if text:
            translation = translate_bengali_to_english([text])[0]
        results.append({
            'box': (x1, y1, x2, y2),
            'text': text,
            'translation': translation
        })

    print("\nFinal OCR + Translation Results:")
    for res in results:
        print(res)

    for (x1, y1, x2, y2), res in zip(boxes, results):
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = res['translation'] or res['text']
        cv2.putText(image, label, (x1, max(y1 - 10, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    output_path = "annotated_translated.jpg"
    cv2.imwrite(output_path, image)
    print(f"Annotated translation image saved to: {output_path}")
