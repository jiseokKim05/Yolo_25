### https://img.seoul.co.kr/img/upload/2017/07/02/SSI_20170702152754_O2.jpg



```bash
# 설치 (처음 한 번만)
!pip install -q ultralytics
!pip install -q opencv-python-headless matplotlib

from PIL import Image
from IPython.display import display

import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

# 파일 업로드
from google.colab import files
uploaded = files.upload()  # 여기서 1.jpg 업로드

# 파일 이름 가져오기
img_filename = list(uploaded.keys())[0]
print("업로드된 파일:", img_filename)

# 이미지 읽기
image = cv2.imread(img_filename)
if image is None:
    raise FileNotFoundError(f"{img_filename} 파일을 읽을 수 없습니다.")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# YOLOv8 모델 로드 (사람 인식 포함된 사전학습 모델)
model = YOLO("yolov8n.pt")  # yolov8n/s/m/l/x 중 선택 가능

# 예측 수행
results = model(image_rgb)[0]

# COCO 클래스 ID: 0번 = person
for box, cls_id, conf in zip(results.boxes.xyxy.cpu().numpy(), 
                             results.boxes.cls.cpu().numpy(), 
                             results.boxes.conf.cpu().numpy()):
    if int(cls_id) == 0:  # person 클래스만
        x1, y1, x2, y2 = map(int, box)
        label = f"person {conf:.2f}"
        cv2.rectangle(image_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image_rgb, label, (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

# 결과 이미지 출력
plt.figure(figsize=(12, 8))
plt.imshow(image_rgb)
plt.axis("off")
plt.title("YOLOv8 - 사람 인식 결과")
plt.show()

img_pil = Image.fromarray(image_rgb)
display(img_pil)
```
