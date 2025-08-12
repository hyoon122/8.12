import cv2
from ultralytics import YOLO

# 모델 로드 (YOLOv11n)
model = YOLO('yolov11n.pt')  # 가벼운 모델. 정확도는 중간

# 영상 경로
video_path = r"C:\Users\405\PythonProject\opencv_project\8.12\img\img_walking.avi"

# 비디오 캡처
cap = cv2.VideoCapture(video_path)

# 영상 열렸는지 확인
if not cap.isOpened():
    print("영상을 열 수 없습니다.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    