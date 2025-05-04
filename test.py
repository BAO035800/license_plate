from PIL import Image
import os

def check_corrupted_images(image_dir):
    corrupted_files = []
    for filename in os.listdir(image_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            file_path = os.path.join(image_dir, filename)
            try:
                with Image.open(file_path) as img:
                    img.verify()  # Kiểm tra xem file có hợp lệ không
                print(f"{filename} OK")
            except (IOError, SyntaxError) as e:
                print(f"{filename} bị hỏng: {e}")
                corrupted_files.append(filename)
    return corrupted_files

# Ví dụ sử dụng
image_dir = r"E:\CODE\NHÓM\nhandienmat\license_plate\dataset\train\image"
corrupted = check_corrupted_images(image_dir)
if corrupted:
    print(f"Các file bị hỏng: {corrupted}")
else:
    print("Không tìm thấy file bị hỏng.")