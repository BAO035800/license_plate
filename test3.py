import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter

# Load mô hình
interpreter = Interpreter(model_path="best.tflite")
interpreter.allocate_tensors()

# Lấy thông tin input/output
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Khởi tạo camera
cap = cv2.VideoCapture(0)  # 0 là camera mặc định
if not cap.isOpened():
    print("Không thể mở camera!")
    exit()

# Kích thước ảnh đầu vào (phù hợp với huấn luyện)
input_size = 640

while True:
    # Đọc frame từ camera
    ret, frame = cap.read()
    if not ret:
        print("Không đọc được frame!")
        break

    # Chuẩn bị ảnh cho mô hình
    img = cv2.resize(frame, (input_size, input_size))
    img = np.expand_dims(img, axis=0).astype(np.uint8)

    # Đưa vào mô hình
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()

    # Lấy kết quả
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]  # Hộp giới hạn
    scores = interpreter.get_tensor(output_details[1]['index'])[0]  # Độ tin cậy
    classes = interpreter.get_tensor(output_details[2]['index'])[0]  # Lớp

    # Vẽ bounding box
    for i in range(len(scores)):
        if scores[i] > 0.5:  # Ngưỡng độ tin cậy
            ymin, xmin, ymax, xmax = boxes[i]
            (left, top, right, bottom) = (int(xmin * input_size), int(ymin * input_size),
                                         int(xmax * input_size), int(ymax * input_size))
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, f"license_plate {scores[i]:.2f}", (left, top - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Hiển thị kết quả
    cv2.imshow("License Plate Detection", frame)

    # Thoát khi nhấn 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()