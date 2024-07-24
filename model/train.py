from google.colab import drive
drive.mount('/content/drive')

!pip install ultralytics

from ultralytics import YOLO

model = YOLO("yolov8s.pt")

results = model.train(data=data,
                      epochs=100,
                      imgsz=640,
                      batch=8,
                      patience=20,
                      workers=8,
                      device=0,
                      project=runs,
                      )

model.export(format="onnx")

import pandas as pd
pd.read_csv('runs/detect/train/results.csv')
print(result)
