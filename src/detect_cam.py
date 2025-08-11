from ultralytics import YOLO
import cv2, time

# Load YOLOv8 nano model (first time will auto-download)
model = YOLO("yolov8n.pt")

# Open webcam (0 = default camera)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Failed to open webcam")

# Variables for FPS calculation
prev, fps = time.time(), 0.0

while True:
    # Read one frame from webcam
    ok, frame = cap.read()
    if not ok:
        break

    # Optional: Reduce resolution to increase FPS
    # frame = cv2.resize(frame, (640, 360))

    # Run YOLO prediction on the frame
    r = model.predict(source=frame, conf=0.4, verbose=False)[0]

    # Draw bounding boxes and labels on the frame
    out = r.plot()

    # Calculate FPS (smoothed)
    now = time.time()
    fps = 0.9 * fps + 0.1 * (1 / (now - prev))
    prev = now

    # Put FPS text on the frame
    cv2.putText(out, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow("YOLOv8 realtime / 실시간 YOLOv8", out)

    # Exit on ESC key
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
