
from google.colab import drive
drive.mount('/content/drive')

!pip install ultralytics
from ultralytics import YOLO

import cv2
import torch

!pwd

cd latest

model = YOLO("best.pt")

video_path = video_path
cap = cv2.VideoCapture(video_path)



!yolo track model=weights/best.pt source=video_path save=True tracker="bytetrack.yaml"

#latest = track4
