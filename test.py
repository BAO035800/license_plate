import cv2
import torch
import time
import os
import numpy as np

# Tạo thư mục output nếu chưa có
os.makedirs("output", exist_ok=True)

# Load mô hình YOLOv5 đã huấn luyện
model = torch.hub.load('yolov5', 'custom', path='./yolov5/runs/train/exp18/weights/best.pt', source='local')

# Khởi tạo webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Không thể mở webcam")
    exit()

# Hàm resize giữ tỷ lệ và thêm padding
def resize_with_padding(img, target_size=(320, 320)):
    h, w = img.shape[:2]
    target_h, target_w = target_size

    # Tính tỷ lệ để resize
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)

    # Resize ảnh
    resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Tạo nền mới (màu đen) với kích thước target
    padded_img = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    
    # Tính vị trí để đặt ảnh vào giữa
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2

    # Đặt ảnh vào giữa nền
    padded_img[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_img

    return padded_img

# Hàm tăng cường độ sáng và độ tương phản
def adjust_brightness_contrast(img, alpha=1.5, beta=50):
    # Alpha điều chỉnh độ tương phản, beta điều chỉnh độ sáng
    adjusted_img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return adjusted_img

# Hàm tạo ảnh nền đen với ảnh đã tiền xử lý
def create_enhanced_image(preprocessed_img, target_size=(320, 320)):
    # Tạo ảnh nền đen với kích thước target
    enhanced_img = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
    
    # Lấy vùng biển số đã preprocessed và đặt vào giữa
    h, w = preprocessed_img.shape[:2]
    x_offset = (target_size[0] - w) // 2
    y_offset = (target_size[1] - h) // 2
    enhanced_img[y_offset:y_offset + h, x_offset:x_offset + w] = preprocessed_img

    return enhanced_img

# Biến lưu trữ
annotated_frame = None
license_plates = []
processed_plates = []

# Tên cửa sổ chính
main_window = "Camera - Detection"

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
        license_plates.clear()
        processed_plates.clear()

        # Lặp qua các bounding boxes để cắt và tiền xử lý biển số
        for box in boxes:
            x1, y1, x2, y2, conf, cls = box
            if class_ids[int(cls)] == 'license_plate' and conf > 0.5:
                try:
                    # Cắt vùng biển số (giữ nguyên, không xử lý)
                    license_plate = frame[int(y1):int(y2), int(x1):int(x2)]
                    if license_plate.size == 0:
                        continue

                    # Lưu ảnh biển số thô
                    license_plates.append(license_plate)

                    # Tiền xử lý riêng để tạo ảnh tăng cường
                    img_rgb = cv2.cvtColor(license_plate, cv2.COLOR_BGR2RGB)
                    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
                    
                    # Áp dụng CLAHE để tăng cường độ tương phản
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                    img_eq = clahe.apply(img_gray)
                    
                    # Chuyển lại sang RGB để tiếp tục xử lý
                    img_eq_rgb = cv2.cvtColor(img_eq, cv2.COLOR_GRAY2RGB)
                    
                    # Điều chỉnh độ sáng và độ tương phản
                    img_adjusted = adjust_brightness_contrast(img_eq_rgb, alpha=1.5, beta=50)
                    
                    # Áp dụng Gaussian Blur để giảm nhiễu
                    img_blurred = cv2.GaussianBlur(img_adjusted, (5, 5), 0)
                    
                    # Resize với padding
                    img_resized = resize_with_padding(img_blurred, target_size=(320, 320))

                    # Chuyển về định dạng BGR để hiển thị/lưu
                    img_processed = cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR)

                    # Lưu vào danh sách ảnh đã tiền xử lý
                    processed_plates.append(img_processed)

                except Exception as e:
                    print(f"[ERROR] Lỗi khi xử lý biển số: {str(e)}")
                    continue

        # Hiển thị ảnh có vẽ bounding box
        cv2.imshow(main_window, annotated_frame)

        # Hiển thị ảnh biển số thô và ảnh đã tiền xử lý
        for i, plate in enumerate(license_plates):
            cv2.imshow(f"Raw License Plate {i}", plate)
        for i, plate in enumerate(processed_plates):
            cv2.imshow(f"Processed License Plate {i}", plate)

        # Kiểm tra phím nhấn và trạng thái cửa sổ
        key = cv2.waitKey(10) & 0xFF  # Tăng thời gian chờ để xử lý sự kiện
        if key == ord('s') and annotated_frame is not None:
            timestamp = int(time.time())
            # Lưu ảnh có vẽ bounding box
            filename = f"output/detected_{timestamp}.jpg"
            if cv2.imwrite(filename, annotated_frame):
                print(f"[INFO] Đã lưu ảnh nhận diện vào {filename}")
            else:
                print(f"[ERROR] Không thể lưu ảnh: {filename}")

            # Lưu ảnh biển số thô và ảnh tăng cường
            for i, (raw_plate, processed_plate) in enumerate(zip(license_plates, processed_plates)):
                # Lưu ảnh biển số thô (không xử lý)
                plate_filename = f"output/license_plate_{timestamp}_{i}.jpg"
                if cv2.imwrite(plate_filename, raw_plate):
                    print(f"[INFO] Đã lưu ảnh biển số thô vào {plate_filename}")
                else:
                    print(f"[ERROR] Không thể lưu ảnh: {plate_filename}")

                # Tạo và lưu ảnh tăng cường trên nền đen
                enhanced_img = create_enhanced_image(processed_plate, target_size=(320, 320))
                enhanced_filename = f"output/license_plate_enhanced_{timestamp}_{i}.jpg"
                if cv2.imwrite(enhanced_filename, enhanced_img):
                    print(f"[INFO] Đã lưu ảnh tăng cường vào {enhanced_filename}")
                else:
                    print(f"[ERROR] Không thể lưu ảnh tăng cường: {enhanced_filename}")

        # Thoát khi nhấn phím 'q' hoặc đóng cửa sổ chính
        if key == ord('q'):
            break

        # Kiểm tra nếu cửa sổ chính bị đóng
        if cv2.getWindowProperty(main_window, cv2.WND_PROP_VISIBLE) < 1:
            print("[INFO] Cửa sổ chính đã đóng, thoát chương trình...")
            break

except Exception as e:
    print(f"[ERROR] Lỗi trong vòng lặp chính: {str(e)}")

finally:
    # Giải phóng tài nguyên
    print("[INFO] Đang giải phóng tài nguyên...")
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Đã thoát chương trình.")