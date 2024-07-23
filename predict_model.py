from google.colab import drive
drive.mount('/content/drive')

!pwd

!cd "/content/drive/MyDrive/Road Damage Detection with YoloV8/Road Damages Detection"

!pip install ultralytics

from ultralytics import YOLO

model = YOLO('/content/drive/MyDrive/runs/yeni/train/weights/best.pt')

res = model.val(data="/content/drive/MyDrive/Road Damage Detection with YoloV8/Road Damages Detection/data.yaml")

results = model.predict("deneme",
                        save=True)

for result in results:
    boxes = result.boxes
    masks = result.masks
    keypoints = result.keypoints
    probs = result.probs
    obb = result.obb
    result.show()
    result.save(filename="result.jpg")

