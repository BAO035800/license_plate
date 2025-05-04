import os
import random
import shutil

# Cấu hình
IMAGE_DIR = 'dataset/raw_images'
LABEL_DIR = 'dataset/labels'

TRAIN_IMAGE_DIR = 'dataset/train/images'
TRAIN_LABEL_DIR = 'dataset/train/labels'
TEST_IMAGE_DIR = 'dataset/test/images'
TEST_LABEL_DIR = 'dataset/test/labels'

SPLIT_RATIO = 0.8  # 80% train, 20% test

# Tạo thư mục nếu chưa có
os.makedirs(TRAIN_IMAGE_DIR, exist_ok=True)
os.makedirs(TRAIN_LABEL_DIR, exist_ok=True)
os.makedirs(TEST_IMAGE_DIR, exist_ok=True)
os.makedirs(TEST_LABEL_DIR, exist_ok=True)

# Lấy danh sách file ảnh (định dạng .jpg hoặc .png)
image_files = [f for f in os.listdir(IMAGE_DIR) if f.endswith(('.jpg', '.png'))]

# Shuffle ảnh ngẫu nhiên
random.shuffle(image_files)

# Chia train/test
split_index = int(len(image_files) * SPLIT_RATIO)
train_files = image_files[:split_index]
test_files = image_files[split_index:]

# Hàm copy ảnh + nhãn
def copy_files(file_list, image_dest, label_dest):
    for img_file in file_list:
        name, _ = os.path.splitext(img_file)
        label_file = name + ".txt"

        shutil.copy(os.path.join(IMAGE_DIR, img_file), os.path.join(image_dest, img_file))
        shutil.copy(os.path.join(LABEL_DIR, label_file), os.path.join(label_dest, label_file))

# Thực hiện copy
copy_files(train_files, TRAIN_IMAGE_DIR, TRAIN_LABEL_DIR)
copy_files(test_files, TEST_IMAGE_DIR, TEST_LABEL_DIR)

print(f"Đã chia thành công: {len(train_files)} ảnh train và {len(test_files)} ảnh test.")
