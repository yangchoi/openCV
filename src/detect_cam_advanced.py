from ultralytics import YOLO
import cv2, time, os

# ----------------------------
# User Config
# ----------------------------
WANTED_CLASSES = {"person", "car"}     # show only these classes
CONF_THRESH = 0.4                      # confidence threshold
RESIZE_TO = None                       # e.g., (640, 360) or None
SAVE_VIDEO = True                      # save output video
OUTPUT_PATH = "output/realtime_yolo.mp4"
# ----------------------------

def ensure_dir(path):
    # Make parent directory if not exists
    os.makedirs(os.path.dirname(path), exist_ok=True)

def main():
    # Load YOLOv8 nano model (first run auto-download)
    model = YOLO("yolov8n.pt")

    # Open webcam (0 = default)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Failed to open webcam")

    # Prepare VideoWriter after first frame size is known
    writer = None
    if SAVE_VIDEO:
        ensure_dir(OUTPUT_PATH)

    prev, fps = time.time(), 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Optional resize for speed
        if RESIZE_TO:
            frame = cv2.resize(frame, RESIZE_TO)

        # Run YOLO prediction
        result = model.predict(source=frame, conf=CONF_THRESH, verbose=False)[0]

        # Draw only wanted classes
        # Start from a copy of the original frame
        out = frame.copy()
        names = result.names

        if result.boxes is not None and len(result.boxes) > 0:
            for b in result.boxes:
                cls_id = int(b.cls[0])
                name = names.get(cls_id, str(cls_id))
                conf = float(b.conf[0])
                if name in WANTED_CLASSES and conf >= CONF_THRESH:
                    x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                    # Draw rectangle & label
                    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{name} {conf:.2f}"
                    cv2.putText(out, label, (x1, max(y1 - 8, 0)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # FPS calculation (smoothed)
        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / (now - prev))
        prev = now
        cv2.putText(out, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        # Initialize writer if needed
        if SAVE_VIDEO and writer is None:
            h, w = out.shape[:2]
            # Try to get FPS from camera; fallback to 30
            fps_out = cap.get(cv2.CAP_PROP_FPS)
            if not fps_out or fps_out <= 0:
                fps_out = 30.0
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps_out, (w, h))

        # Write frame to video
        if SAVE_VIDEO and writer is not None:
            writer.write(out)

        # Show frame
        cv2.imshow("YOLOv8 Filtered", out)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

    # Cleanup
    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
