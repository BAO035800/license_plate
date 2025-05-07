import os

# Đường dẫn thư mục chứa file labels (sửa \ thành /)
label_dir = 'E:/CODE/NHOM/nhandienmat/license_plate/dataset/labels'

# 1. Xử lý tất cả file .txt trong thư mục labels: loại bỏ xuống dòng và ghi đè
try:
    # Kiểm tra thư mục có tồn tại không
    if not os.path.exists(label_dir):
        print(f"Thư mục {label_dir} không tồn tại. Hãy tạo thư mục hoặc kiểm tra lại.")
    else:
        # Duyệt qua tất cả file trong thư mục labels
        for filename in os.listdir(label_dir):
            # Chỉ xử lý các file .txt
            if filename.endswith('.txt'):
                file_path = os.path.join(label_dir, filename)
                try:
                    # Đọc file
                    with open(file_path, 'r') as file:
                        content = file.read()

                    # Thay thế xuống dòng bằng khoảng trắng
                    content = content.replace('\n', ' ').replace('\r', '')

                    # Ghi đè lên file gốc
                    with open(file_path, 'w') as file:
                        file.write(content)

                    print(f"Xử lý hoàn tất! Kiểm tra file: {file_path}")
                except Exception as e:
                    print(f"Lỗi khi xử lý file {file_path}: {e}")

except Exception as e:
    print(f"Đã có lỗi xảy ra khi truy cập thư mục: {e}")

# 2. Xóa các file rỗng (kích thước 0 byte) trong thư mục labels
try:
    if os.path.exists(label_dir):
        for filename in os.listdir(label_dir):
            file_to_check = os.path.join(label_dir, filename)
            # Kiểm tra nếu là file và kích thước = 0
            if os.path.isfile(file_to_check) and os.path.getsize(file_to_check) == 0:
                try:
                    os.remove(file_to_check)
                    print(f"Đã xóa file rỗng: {file_to_check}")
                except Exception as e:
                    print(f"Lỗi khi xóa file {file_to_check}: {e}")
    else:
        print(f"Thư mục {label_dir} không tồn tại. Hãy kiểm tra lại.")
except Exception as e:
    print(f"Lỗi khi xử lý thư mục: {e}")