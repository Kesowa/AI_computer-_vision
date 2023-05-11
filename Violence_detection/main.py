from typing import Tuple
from pydantic import BaseModel
from fastapi import BackgroundTasks, FastAPI
import requests
import urllib.request
from lib import ViolenceDetector
import os

violence_detector = ViolenceDetector("./ModelWeightbest_bests.h5");

app = FastAPI()


@app.post("/items/{item_id}")
def read_item(item_id: int, q: Tuple[str, int]):
    return {"item_id": item_id, "q": q}


class VideoViolence(BaseModel):
    inference: str
    target: str
    callback: str


def infer_video_violence(path: str):
    return {
        "Keys": ["violence", "nonviolence"],
        "Values": [
            [0.9, 0.1],
            [0.9, 0.1],
            [0.9, 0.1],
            [0.9, 0.1],
            [0.9, 0.1],
        ]
    }


def detect_violence(body: VideoViolence):
    print(body)
    [path, status] = urllib.request.urlretrieve(body.target)
    # if status != 200:
    #     print("unable to download video")
    #     print(status)
    #     return
    # data = infer_video_violence(path)
    data = violence_detector.predict_frames(path)
    res = requests.post(body.callback, data=data)
    os.remove(path)
    print(res.json())


@app.post("/video/violence")
def detect_violence_endpoint(
        body: VideoViolence,
        background_tasks: BackgroundTasks):
    background_tasks.add_task(detect_violence, body)
    return body
