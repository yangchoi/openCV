import cv2

cap = cv2.VideoCapture(0) # 외장카메라일 경우 1, 2로 바꾸기
if not cap.isOpened():
    raise RuntimeError("Impossible to open webcam. Check the Permission.")

while True:
    ok, frame = cap.read()
    if not ok:
        break

    cv2.imshow("Smoke Test", frame)
    if cv2.waitKey(1) & 0xFF == 27: # ESC
        break

cap.release()
cv2.destroyAllWindows()

