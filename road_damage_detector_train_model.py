from google.colab import drive
drive.mount('/content/drive')

import locale
locale.getpreferredencoding = lambda: 'UTF-8'

!pip install ultralytics

from ultralytics import YOLO

modeln = YOLO('yolov8n.pt')

modeln.info()

result = modeln.train(data = '/content/drive/MyDrive/Road Damage Detection with YoloV8/Road Damages Detection/data.yaml',
                      epochs=25,
                      batch=8,
                      optimizer='Adam',
                      device='gpu',
                      patience=15,
                      workers=8,
                      project='/content/drive/MyDrive/runs/yeni',
                      )

import pandas as pd

pd.read_csv('runs/detect/train/results.csv')

print(result)

