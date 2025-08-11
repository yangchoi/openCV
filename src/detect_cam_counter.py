import cv2, time, os, math, uuid
from collections import deque
from ultralytics import YOLO

"""
# 참고
너무 흔들리면: MAX_DIST 를 조금 올리거나 RESIZE_TO 해상도를 높여 중심점 오차를 줄이기

감지가 적으면: CONF_THRESH 0.25~0.35로 내려보기, 또는 yolov8s.pt로 모델 업그레이드

라인 위치 바꾸기: line_y = h // 2 부분을 원하는 y 픽셀로 변경
"""

# ================================
# User Config
# ================================
WANTED_CLASSES = {"person", "car"}   # Count only these
CONF_THRESH = 0.35                   # Detection confidence
RESIZE_TO = (960, 540)               # Processing size or None
SAVE_VIDEO = True                    # Save annotated video
OUTPUT_PATH = "output/realtime_count.mp4"
MAX_DISAPPEAR = 10                   # Tracking tolerance (frames)
MAX_DIST = 80                        # Match threshold (pixels)

# ================================
# Simple Centroid Tracker
# ================================
class CentroidTracker:
    def __init__(self, max_disappear=10, max_dist=80):
        self.next_id = 1
        self.objects = {}           # id -> (x, y)
        self.disappeared = {}       # id -> frames
        self.paths = {}             # id -> deque of centroids
        self.max_disappear = max_disappear
        self.max_dist = max_dist

    def update(self, centroids):
        # If no detections, mark disappeared
        if len(centroids) == 0:
            to_delete = []
            for oid in list(self.disappeared.keys()):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappear:
                    to_delete.append(oid)
            for oid in to_delete:
                self.objects.pop(oid, None)
                self.disappeared.pop(oid, None)
                self.paths.pop(oid, None)
            return self.objects, {}

        # If no existing objects, register all
        if len(self.objects) == 0:
            assignments = {}
            for c in centroids:
                oid = self.next_id
                self.next_id += 1
                self.objects[oid] = c
                self.disappeared[oid] = 0
                self.paths[oid] = deque(maxlen=15)
                self.paths[oid].append(c)
                assignments[oid] = c
            return self.objects, assignments

        # Match by nearest neighbor
        obj_ids = list(self.objects.keys())
        obj_pts = [self.objects[i] for i in obj_ids]
        used_obj = set()
        used_det = set()
        assignments = {}

        for di, c in enumerate(centroids):
            # Find nearest object
            best, best_id = 1e9, None
            for oi, oid in enumerate(obj_ids):
                if oid in used_obj:
                    continue
                ox, oy = obj_pts[oi]
                dist = math.hypot(c[0]-ox, c[1]-oy)
                if dist < best:
                    best, best_id = dist, oid
            if best_id is not None and best <= self.max_dist:
                # Assign
                self.objects[best_id] = c
                self.disappeared[best_id] = 0
                self.paths[best_id].append(c)
                assignments[best_id] = c
                used_obj.add(best_id)
                used_det.add(di)

        # Unmatched detections -> register
        for di, c in enumerate(centroids):
            if di in used_det:
                continue
            oid = self.next_id
            self.next_id += 1
            self.objects[oid] = c
            self.disappeared[oid] = 0
            self.paths[oid] = deque(maxlen=15)
            self.paths[oid].append(c)
            assignments[oid] = c

        # Unmatched objects -> disappeared +1
        for oid in list(self.objects.keys()):
            if oid not in assignments:
                self.disappeared[oid] = self.disappeared.get(oid, 0) + 1
                if self.disappeared[oid] > self.max_disappear:
                    self.objects.pop(oid, None)
                    self.disappeared.pop(oid, None)
                    self.paths.pop(oid, None)

        return self.objects, assignments

def ensure_dir(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

def main():
    model = YOLO("yolov8n.pt")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Failed to open webcam")

    writer = None
    if SAVE_VIDEO:
        ensure_dir(OUTPUT_PATH)

    tracker = CentroidTracker(MAX_DISAPPEAR, MAX_DIST)

    # Counters
    up_count = 0      # line bottom -> top
    down_count = 0    # line top -> bottom

    prev_time, fps = time.time(), 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if RESIZE_TO:
            frame = cv2.resize(frame, RESIZE_TO)

        h, w = frame.shape[:2]
        line_y = h // 2  # counting line (horizontal middle)

        # Detection
        res = model.predict(source=frame, conf=CONF_THRESH, verbose=False)[0]
        names = res.names

        boxes = []
        cents = []
        labels = []

        if res.boxes is not None and len(res.boxes) > 0:
            for b in res.boxes:
                cls_id = int(b.cls[0])
                name = names.get(cls_id, str(cls_id))
                conf = float(b.conf[0])

                if WANTED_CLASSES and name not in WANTED_CLASSES:
                    continue
                if conf < CONF_THRESH:
                    continue

                x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                cx, cy = int((x1+x2)/2), int((y1+y2)/2)

                boxes.append((x1, y1, x2, y2))
                cents.append((cx, cy))
                labels.append(f"{name} {conf:.2f}")

        # Update tracker
        objs, assigns = tracker.update(cents)

        # Direction check using path crossing
        # If a path crosses line_y from above to below -> down_count, reverse -> up_count
        for oid, _ in assigns.items():
            path = tracker.paths.get(oid, None)
            if path and len(path) >= 2:
                # recent two points
                y_prev = path[-2][1]
                y_now = path[-1][1]
                if y_prev < line_y <= y_now:
                    down_count += 1
                elif y_prev > line_y >= y_now:
                    up_count += 1

        # Draw
        out = frame.copy()
        # Draw counting line
        cv2.line(out, (0, line_y), (w, line_y), (0, 200, 255), 2)
        cv2.putText(out, "COUNT LINE", (10, line_y-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,200,255), 2)

        # Draw boxes and IDs
        for (x1, y1, x2, y2), (cx, cy), lab in zip(boxes, cents, labels):
            # Find matched object id by nearest centroid
            best_id, best = None, 1e9
            for oid, (ox, oy) in objs.items():
                d = math.hypot(cx-ox, cy-oy)
                if d < best:
                    best, best_id = d, oid
            color = (0, 255, 0) if best_id else (255, 255, 255)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            tag = f"ID {best_id} | {lab}" if best_id else lab
            cv2.putText(out, tag, (x1, max(0, y1-8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            cv2.circle(out, (cx, cy), 3, (255, 0, 0), -1)

        # FPS
        now = time.time()
        fps = 0.9*fps + 0.1*(1/(now - prev_time))
        prev_time = now
        cv2.putText(out, f"FPS: {fps:.1f}", (10, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        # Counts
        cv2.putText(out, f"UP: {up_count}", (10, 56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100,255,100), 2)
        cv2.putText(out, f"DOWN: {down_count}", (10, 86),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100,255,255), 2)

        # Video writer init
        if SAVE_VIDEO and writer is None:
            fps_out = cap.get(cv2.CAP_PROP_FPS) or 30.0
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps_out, (w, h))

        if SAVE_VIDEO and writer:
            writer.write(out)

        cv2.imshow("YOLOv8 Counting", out)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
