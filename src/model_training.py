import torch
import torch.nn as nn
from torchvision import models
import time

# --- ĐỊNH NGHĨA CÁC MÔ HÌNH ---
# Module này chứa các hàm khởi tạo 3 kiến trúc mô hình đã được so sánh.

def create_resnet50_model(num_classes=2, pretrained=True):
    """
    Hàm tạo mô hình ResNet-50 (Kiến trúc ResNet-50 transfer learning).
    """
    if pretrained:
        weights = models.ResNet50_Weights.IMAGENET1K_V1
    else:
        weights = None
        
    model = models.resnet50(weights=weights)
    
    # Đóng băng các trọng số cũ nếu dùng pre-trained
    if pretrained:
        for param in model.parameters():
            param.requires_grad = False
    
    # Thay thế lớp phân loại (fully connected) cuối cùng
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model

def create_mobilenet_v2_model(num_classes=2, pretrained=True):
    """
    Hàm tạo mô hình MobileNet-V2 (Kiến trúc MobileNet-V2 transfer learning).
    """
    if pretrained:
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V1
    else:
        weights = None
        
    model = models.mobilenet_v2(weights=weights)
    
    if pretrained:
        for param in model.parameters():
            param.requires_grad = False
    
    # Thay thế lớp phân loại cuối cùng
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    
    return model

def create_efficientnet_b0_model(num_classes=2, pretrained=True):
    """
    Hàm tạo mô hình EfficientNet-B0 (Kiến trúc EfficientNet-B0 transfer learning).
    """
    if pretrained:
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
    else:
        weights = None
            
    model = models.efficientnet_b0(weights=weights)
    
    if pretrained:
        for param in model.parameters():
            param.requires_grad = False
    
    # Thay thế lớp phân loại cuối cùng
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    
    return model


# --- HÀM HUẤN LUYỆN (Training Loop) ---
# Cung cấp một vòng lặp huấn luyện tiêu chuẩn

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device='cpu'):
    """
    Hàm thực hiện vòng lặp huấn luyện và đánh giá mô hình.
    """
    print(f"Bắt đầu huấn luyện trên {device}...")
    model.to(device)
    
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }

    for epoch in range(num_epochs):
        start_time = time.time()
        
        # --- Training ---
        model.train()
        running_loss = 0.0
        correct_preds = 0
        
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct_preds += torch.sum(preds == labels.data)
            
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct_preds.double() / len(train_loader.dataset)
        
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc.item())

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_corrects += torch.sum(preds == labels.data)

        val_epoch_loss = val_loss / len(val_loader.dataset)
        val_epoch_acc = val_corrects.double() / len(val_loader.dataset)
        
        history['val_loss'].append(val_epoch_loss)
        history['val_acc'].append(val_epoch_acc.item())
        
        end_time = time.time()
        
        print(f'Epoch {epoch+1}/{num_epochs} [Thời gian: {end_time - start_time:.2f}s]')
        print(f'  Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        print(f'  Val   Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.4f}')
        
    print("Huấn luyện hoàn tất!")
    return model, history
