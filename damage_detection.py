from google.colab import drive
drive.mount('/content/drive')

!pip install ultralytics

from ultralytics import YOLO
import cv2
import time
import random
import numpy as np

confidence_score = 0.5

text_color_b = (0,0,0)
text_color_w = (255,255,255)
background_color = (0,255,0)

font = cv2.FONT_HERSHEY_SIMPLEX

total_fps = 0
average_fps = 0
num_of_frame = 0

model = YOLO("/content/drive/MyDrive/runs/yeni/train/weights/best.pt")

labels = model.names
colors = [[random.randint(0,255) for _ in range(0,3)] for _ in labels]

video_path = "/content/drive/MyDrive/Road Damage Detection with YoloV8/4695859-uhd_3840_2160_30fps.mp4"

cap = cv2.VideoCapture(video_path)
width = int(cap.get(3))
height = int(cap.get(4))
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# video-writer ayarları
output_path = '/content/drive/MyDrive/Processed_Video.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # veya 'XVID', 'DIVX'
out = cv2.VideoWriter(output_path, fourcc, fps, (width // 2, height // 2))

print("[INFO].. Width:", width)
print("[INFO].. Height:", height)
print("[INFO].. Total Frames:", total_frames)

frame_skip = 5  # 5 frame'de bir işlem

while True:
    start = time.time()

    for _ in range(frame_skip):
        ret, frame = cap.read()
        if not ret:
            break

    if not ret:
        break

    original_frame = frame.copy()
    frame = cv2.resize(frame, (width // 2, height // 2))

    results = model(frame, verbose=False)[0]


    boxes = np.array(results.boxes.data.tolist())

    for box in boxes:
        x1, y1, x2, y2, score, class_id = box
        x1, y1, x2, y2 = int(x1 * (width / 2 / width)), int(y1 * (height / 2 / height)), int(x2 * (width / 2 / width)), int(y2 * (height / 2 / height))
        class_id = int(class_id)

        box_color = colors[class_id]

        if score > confidence_score:
            cv2.rectangle(original_frame, (x1, y1), (x2, y2), box_color, 2)

            score = score * 100
            class_name = results.names[class_id]

            text = f"{class_name}: %{score:.2f}"

            text_loc = (x1, y1-10)

            labelSize, baseLine = cv2.getTextSize(text, font, 1, 1)
            cv2.rectangle(original_frame,
                          (x1, y1 - 10 - labelSize[1]),
                          (x1 + labelSize[0], int(y1 + baseLine-10)),
                          box_color,
                          cv2.FILLED)

            cv2.putText(original_frame, text, (x1, y1-10), font, 1, text_color_w, thickness=1)

    end = time.time()

    num_of_frame += 1
    fps = 1 / (end-start)
    total_fps = total_fps + fps

    average_fps = total_fps / num_of_frame
    avg_fps = float("{:.2f}".format(average_fps))

    cv2.rectangle(original_frame, (10,2), (280,50), background_color, -1)
    cv2.putText(original_frame, "FPS: "+str(avg_fps), (20,40), font, 1.5, text_color_b, thickness=3)

    out.write(original_frame)
    print("(%2d / %2d) frame işlendi." % (num_of_frame, total_frames))

cap.release()
out.release()
cv2.destroyAllWindows()

print("Video işleme tamamlandı ve dosya kaydedildi:", output_path)

