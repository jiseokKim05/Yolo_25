#### https://www.youtube.com/watch?v=7LrWGGJFEJo

```bash
# 1. 필요한 라이브러리 설치
!pip install ultralytics opencv-python

# 2. 필요한 모듈 임포트
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
from google.colab import files
import os

# 3. YOLOv8 모델 로드 (사전 학습된 모델 사용)
model = YOLO('yolov8n.pt')  # 필요에 따라 'yolov8s.pt', 'yolov8m.pt' 등으로 변경 가능

# 4. 동영상 업로드
uploaded = files.upload()
video_path = list(uploaded.keys())[0]  # 업로드된 동영상 파일 이름 가져오기

# 5. 출력 디렉토리 생성 (프레임별 사진 저장용)
output_dir = 'output_frames'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 6. 동영상 처리 및 객체 감지
cap = cv2.VideoCapture(video_path)
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # 동영상 끝나면 종료

    # YOLOv8으로 객체 감지
    results = model(frame)

    # 결과 렌더링 (감지된 객체에 라벨과 박스 그리기)
    annotated_frame = results[0].plot()

    # 프레임별로 화면에 출력
    plt.figure(figsize=(10, 6))
    plt.imshow(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title(f'Frame {frame_count}')
    plt.show()

    # 프레임별로 사진 파일로 저장 (선택 사항)
    output_path = f'{output_dir}/frame_{frame_count:04d}.jpg'
    cv2.imwrite(output_path, annotated_frame)

    frame_count += 1

    # 너무 많은 프레임 출력 방지를 위해 제한 설정 (예: 10프레임만 처리)
    if frame_count >= 10:
        break

# 7. 자원 해제
cap.release()
print(f"총 {frame_count} 프레임이 처리되었습니다.")
print(f"프레임별 사진이 '{output_dir}' 디렉토리에 저장되었습니다.")

# 8. 저장된 프레임 압축 및 다운로드 (선택 사항)
!zip -r output_frames.zip {output_dir}
files.download('output_frames.zip')
```

![Uploading image.png…]()

