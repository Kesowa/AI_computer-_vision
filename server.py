from enum import Enum
from pydantic import BaseModel
from fastapi import BackgroundTasks, FastAPI
import requests
import urllib.request
import os


def infer_video_violence(path: str):
    return {
        "Keys": ["violence", "nonviolence"],
        "Values": [
            [0.9, 0.1],
            [0.9, 0.1],
            [0.9, 0.1],
            [0.9, 0.1],
            [0.9, 0.1],
        ],
    }


class Dummy:
    def __init__(self, model_path="checkpoint.pth"):
        pass

    def infer(self, data_path="raster.tif"):
        return infer_video_violence(data_path)


app = FastAPI()


class Inference(Enum):
    VIOLENCE = "violence"
    DEEPFOREST = "deepforest"
    CONSTRUCTION = "construction"


class AiTask(BaseModel):
    inference: Inference
    target: str
    callback: str


def detect_violence(body: AiTask):
    print(body)
    model = Dummy()
    [data_path, status] = urllib.request.urlretrieve(body.target)
    data = model.infer(data_path)
    os.remove(data_path)
    res = requests.post(body.callback, json=data)
    print(res.json())


INFERENCES = {
    Inference.VIOLENCE: detect_violence,
}


@app.post("/")
def infer(body: AiTask, background_tasks: BackgroundTasks):
    background_tasks.add_task(INFERENCES[body.inference], body)
    return body
