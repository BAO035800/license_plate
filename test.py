import cv2

cap = cv2.VideoCapture(1)  # thử số 1 nếu là camera USB

if not cap.isOpened():
    print("Không thể mở camera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow("Camera", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
