from google.colab import drive
drive.mount('/content/drive')

!pip install ultralytics

from ultralytics import YOLO

model = YOLO("yolov8s.pt")

results = model.train(data="data.yaml",
                      epochs=65,
                      imgsz=640,
                      name="latest",
                      device=0,
                      save_period=5,
                      exist_ok = True
                      )

model.export(format="onnx")

import pandas as pd
pd.read_csv("latest/results.csv")

print(results)
