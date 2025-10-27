import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import os

# === ĐỊNH NGHĨA CÁC PHÉP BIẾN ĐỔI (TRANSFORMS) ===
# (Định nghĩa các phép biến đổi chuẩn hóa và tăng cường dữ liệu)

# Phép biến đổi cho tập huấn luyện (có Augmentation - làm giàu dữ liệu)
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),      # Đưa ảnh về 224x224
    transforms.RandomHorizontalFlip(), # Lật ngang ngẫu nhiên
    transforms.RandomRotation(10),     # Xoay ảnh ngẫu nhiên 10 độ
    transforms.ToTensor(),             # Chuyển ảnh thành Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Chuẩn hóa
                         std=[0.229, 0.224, 0.225])
])

# Phép biến đổi cho tập validation, test và web app (Không Augmentation)
test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


def create_dataloaders(data_dir, batch_size=32):
    """
    Tạo và trả về DataLoaders cho tập train, val, test.

    Hàm này giả định data_dir có cấu trúc thư mục con:
    data_dir/
        train/
        val/
        test/
    """
    train_data_path = os.path.join(data_dir, 'train')
    val_data_path = os.path.join(data_dir, 'val')
    test_data_path = os.path.join(data_dir, 'test')

    # Tạo datasets
    train_dataset = ImageFolder(root=train_data_path, transform=train_transforms)
    val_dataset = ImageFolder(root=val_data_path, transform=test_transforms)
    test_dataset = ImageFolder(root=test_data_path, transform=test_transforms)

    # Tạo DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Lấy tên các lớp từ thư mục
    class_names = train_dataset.classes 

    return train_loader, val_loader, test_loader, class_names
