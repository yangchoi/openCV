# Realtime Object Detection with OpenCV + YOLOv8

## 1) Overview / 개요

This project implements **realtime object detection** using **OpenCV** for video I/O & visualization and **YOLOv8** for detection.
OpenCV는 영상 입출력·시각화를, YOLOv8은 객체 감지를 담당

**Today’s scope / 오늘 구현 범위**

- Python venv setup / 가상환경 세팅
- Webcam smoke test / 카메라 스모크 테스트
- YOLOv8 realtime detection (CPU) / 실시간 감지
- Advanced: class filtering + MP4 saving / 클래스 필터링 + 동영상 저장

---

## 2) Environment Setup / 환경 설정

```bash
# Create & activate venv
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install deps
pip install --upgrade pip
pip install opencv-python ultralytics numpy

# (Optional) freeze
pip freeze > requirements.txt
```

## 3) Project Structure / 폴더 구조

```bash
opencv-project/
├─ venv/
├─ src/
│ ├─ cam_smoke_test.py # webcam preview / 카메라 미리보기
│ ├─ detect_cam.py # YOLOv8 realtime detection / 실시간 감지
│ └─ detect_cam_advanced.py # class filter + save / 필터+저장
├─ output/ # saved videos / 저장 영상
├─ data/ # test videos (optional) / 테스트 영상
└─ README.md
```

## 4) Quick Start

A) Webcam smoke test

```bash
python src/cam_smoke_test.py
```

B) Realtime detection / 실시간 감지

- Shows bounding boxes & labels, prints smoothed FPS.

```bash
python src/detect_cam.py
```

C) Advanced: class filter + save

```bash
WANTED_CLASSES = None  # None → show all / 전체 표시
# e.g. {"person", "car", "cell phone", "chair"}
CONF_THRESH = 0.25     # lower → more detections
RESIZE_TO = (960, 540) # None or (W,H)
SAVE_VIDEO = True
OUTPUT_PATH = "output/realtime_yolo.mp4"

# run:
python src/detect_cam_advanced.py

```

## 5) How it works

OpenCV: VideoCapture로 프레임 읽기 → imshow로 표시

YOLOv8: model.predict(frame) → boxes, classes, confidences

Visualization: bounding boxes & labels → FPS overlay

(Advanced) 원하는 클래스만 남기고 VideoWriter로 MP4 저장

## 6) Troubleshooting / 문제 해결

- macOS camera permission:
  System Settings → Privacy & Security → Camera에서 사용 앱(Visual Studio Code/Terminal) 토글 ON.

= “Import could not be resolved” (ultralytics):
VS Code의 인터프리터가 venv인지 확인 → Python: Select Interpreter.

= OpenCV not authorized / webcam open fail:
권한 초기화 tccutil reset Camera; 앱 재시작.

- Low FPS:
  해상도 축소(RESIZE_TO), 조명 밝게, 모델 yolov8n → s 비교.

## 7) Results

- CPU: resolution, FPS 표

- 스크린샷 1–2장 (전체 / 필터링)

- 저장된 샘플 영상(GIF/MP4) 링크

## 8) Roadmap

Line crossing counting (완료/미완료 표기)

Multi-object tracking (ByteTrack)

ROI counting dashboard (Streamlit)

ROS2 / LiDAR visualization
