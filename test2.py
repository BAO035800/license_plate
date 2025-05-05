import cv2
import torch
import time
import os

# Tạo thư mục output nếu chưa có
os.makedirs("output", exist_ok=True)

# Load mô hình YOLOv5 đã huấn luyện
model = torch.hub.load('yolov5', 'custom', path='./yolov5/runs/train/exp18/weights/best.pt', source='local')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Dự đoán bằng mô hình
    results = model(frame)
    annotated_frame = results.render()[0]  # Ảnh có vẽ bounding box

    cv2.imshow("Camera - Detection", annotated_frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    elif key == ord('s'):  # Nhấn 's' để in ra kết quả và lưu ảnh
        print("\n[INFO] Kết quả nhận diện:")
        results.print()  # In kết quả ra console

        # Lưu ảnh vào thư mục 'output'
        filename = f"output/detected_{int(time.time())}.jpg"
        cv2.imwrite(filename, annotated_frame)
        print(f"[INFO] Đã lưu ảnh nhận diện vào {filename}")

cap.release()
cv2.destroyAllWindows()
