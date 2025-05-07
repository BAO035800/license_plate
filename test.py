import cv2
import torch
import time
import os

# Tạo thư mục output nếu chưa có
os.makedirs("output", exist_ok=True)

# Load mô hình YOLOv5 đã huấn luyện
model = torch.hub.load('yolov5', 'custom', path='./yolov5/runs/train/exp18/weights/best.pt', source='local')
# Khởi tạo webcam
cap = cv2.VideoCapture(0)

# Kích thước phóng to cho ảnh biển số (có thể điều chỉnh)
ENLARGED_SIZE = (640, 480)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Không thể đọc khung hình từ webcam")
            break

        # Dự đoán bằng mô hình
        results = model(frame)
        annotated_frame = results.render()[0]  # Ảnh có vẽ bounding box

        # Lấy các thông tin về bounding boxes
        boxes = results.xyxy[0].cpu().numpy()
        class_ids = results.names
        license_plates = []

        # Lặp qua các bounding boxes để cắt vùng biển số
        for box in boxes:
            x1, y1, x2, y2, conf, cls = box
            if class_ids[int(cls)] == 'license_plate' and conf > 0.5:
                try:
                    # Cắt vùng biển số (không tiền xử lý)
                    license_plate = frame[int(y1):int(y2), int(x1):int(x2)]
                    if license_plate.size == 0:
                        continue

                    # Phóng to ảnh biển số
                    h, w = license_plate.shape[:2]
                    aspect_ratio = w / h
                    new_w = int(ENLARGED_SIZE[1] * aspect_ratio)
                    new_h = ENLARGED_SIZE[1]
                    if new_w > ENLARGED_SIZE[0]:
                        new_w = ENLARGED_SIZE[0]
                        new_h = int(ENLARGED_SIZE[0] / aspect_ratio)
                    enlarged_plate = cv2.resize(license_plate, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

                    # Lưu vào danh sách
                    license_plates.append(enlarged_plate)

                except Exception as e:
                    print(f"[ERROR] Lỗi khi xử lý biển số: {str(e)}")
                    continue

        # Hiển thị ảnh có vẽ bounding box
        cv2.imshow("Camera - Detection", annotated_frame)

        # Hiển thị ảnh biển số đã cắt và phóng to
        for i, plate in enumerate(license_plates):
            cv2.imshow(f"License Plate {i}", plate)

        # Kiểm tra phím nhấn
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif key == ord('s'):  # Nhấn 's' để in ra kết quả và lưu ảnh
            print("\n[INFO] Kết quả nhận diện:")
            results.print()  # In kết quả ra console

            timestamp = int(time.time())
            # Lưu ảnh tổng quát (có vẽ bounding box)
            filename = f"output/detected_{timestamp}.jpg"
            if cv2.imwrite(filename, annotated_frame):
                print(f"[INFO] Đã lưu ảnh nhận diện vào {filename}")
            else:
                print(f"[ERROR] Không thể lưu ảnh: {filename}")

            # Lưu ảnh vùng biển số đã cắt và phóng to
            for i, plate in enumerate(license_plates):
                plate_filename = f"output/license_plate_{timestamp}_{i}.jpg"
                if cv2.imwrite(plate_filename, plate):
                    print(f"[INFO] Đã lưu ảnh biển số vào {plate_filename}")
                else:
                    print(f"[ERROR] Không thể lưu ảnh: {plate_filename}")

except Exception as e:
    print(f"[ERROR] Lỗi trong vòng lặp chính: {str(e)}")

finally:
    # Giải phóng tài nguyên
    print("[INFO] Đang giải phóng tài nguyên...")
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Đã thoát chương trình.")