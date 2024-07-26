
from google.colab import drive
drive.mount('/content/drive')

!pip install ultralytics

from ultralytics import YOLO

model = YOLO("best.pt")

metrics = model.val()  # no arguments needed ;)
print(metrics.box.map)  # map50-95
print(metrics.box.map50 )
print(metrics.box.map75)
print(metrics.box.maps)
