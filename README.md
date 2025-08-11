# Realtime Object Detection with OpenCV + YOLOv8

## 1) Overview

This project implements **realtime object detection** using **OpenCV** for video I/O & visualization and **YOLOv8** for detection.

**Today’s scope (2025.08.11)**

- Python venv setup
- Webcam smoke test
- YOLOv8 realtime detection (CPU)
- Advanced: class filtering + MP4 saving

---

## 2) Environment Setup

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

## 3) Project Structure

```bash
opencv-project/
├─ venv/
├─ src/
│ ├─ cam_smoke_test.py # webcam preview
│ ├─ detect_cam.py # YOLOv8 realtime detection
│ └─ detect_cam_advanced.py # class filter + save
├─ output/ # saved videos
├─ data/ # test videos
└─ README.md
```

## 4) Quick Start

A) Webcam smoke test

```bash
python src/cam_smoke_test.py
```

B) Realtime detection

- Shows bounding boxes & labels, prints smoothed FPS.

```bash
python src/detect_cam.py
```

C) Advanced: class filter + save

```bash
WANTED_CLASSES = None  # None → show all
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

## 6) Troubleshooting

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
