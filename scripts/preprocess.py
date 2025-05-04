import cv2
import torch
from torchvision import transforms
import os
import numpy as np

image_folder = "E:/CODE/NHOM/nhandienmat/license_plate/dataset/raw_images"
processed_image_folder = "E:/CODE/NHOM/nhandienmat/license_plate/dataset/processed_images"




# # Tạo thư mục processed_images nếu chưa tồn tại
# if not os.path.exists(processed_image_folder):
#     os.makedirs(processed_image_folder)

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

# Khởi tạo transform
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
])

# Lặp qua các ảnh trong thư mục gốc
for filename in os.listdir(image_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        # Đọc ảnh bằng OpenCV
        img_path = os.path.join(image_folder, filename)
        img = cv2.imread(img_path)

        if img is None:
            continue  # Bỏ qua nếu ảnh không đọc được

        # Chuyển từ BGR (OpenCV) sang RGB (PyTorch)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Tăng độ tương phản và ánh sáng
        # Chuyển sang grayscale để cân bằng histogram
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        img_eq = cv2.equalizeHist(img_gray)
        # Chuyển lại sang RGB
        img_eq_rgb = cv2.cvtColor(img_eq, cv2.COLOR_GRAY2RGB)

        # Resize và padding
        img_resized = resize_with_padding(img_eq_rgb, target_size=(320, 320))

        # Áp dụng transform
        img_tensor = transform(img_resized)  # [3, 320, 320]

        # Đảm bảo tensor có kích thước batch [N, 3, 320, 320]
        img_tensor = img_tensor.unsqueeze(0)  # [1, 3, 320, 320]

        # Chuyển tensor về dạng ảnh để lưu
        img_processed = img_tensor.squeeze(0).permute(1, 2, 0).numpy()  # [320, 320, 3]
        img_processed = (img_processed * 255).astype('uint8')  # Chuyển về dạng uint8
        img_processed = cv2.cvtColor(img_processed, cv2.COLOR_RGB2BGR)  # Chuyển về BGR để lưu

        # Đường dẫn lưu ảnh đã xử lý
        processed_path = os.path.join(processed_image_folder, filename)
        cv2.imwrite(processed_path, img_processed)

        # In kích thước để kiểm tra
        processed_batch = torch.stack([img_tensor])
        print(f"Processed batch shape: {processed_batch.shape}")