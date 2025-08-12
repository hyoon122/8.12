import cv2
from ultralytics import YOLO

# 현재 파일의 위치 기준으로 img 폴더 내 영상 파일 경로 설정
BASE_DIR = Path(__file__).resolve().parent.parent
VIDEO_PATH = BASE_DIR / 'img' / 'img_walking.avi'

# 모델 로드 (YOLOv11n)
model = YOLO('yolov11n.pt')  # 가벼운 모델. 정확도는 중간

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

    # YOLO로 객체 탐지
    results = model(frame, verbose=False)[0]

    # 사람 class만 필터링 (COCO class에서 '0'은 사람)
    people = [det for det in results.boxes.data if int(det[5]) == 0]
    person_count = len(people)

    # 박스 그리기
    for det in people:
        x1, y1, x2, y2, conf, cls = det
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, f'Person {conf:.2f}', (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # 사람 수 출력
    cv2.putText(frame, f'Total People: {person_count}', (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # 화면에 출력
    cv2.imshow("Person Detection", frame)

    # 'q'를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 자원 해제
cap.release()
cv2.destroyAllWindows()

    