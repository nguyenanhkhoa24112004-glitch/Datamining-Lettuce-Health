import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import torch

def plot_history(history):
    """
    Hàm này vẽ đồ thị Training/Validation Loss và Accuracy.
    (Vẽ đồ thị lịch sử huấn luyện)
    """
    
    # Lấy accuracy và loss
    # Chuyển đổi tensor (nếu có) sang giá trị số
    train_acc = [a.item() if isinstance(a, torch.Tensor) else a for a in history['train_acc']]
    val_acc = [a.item() if isinstance(a, torch.Tensor) else a for a in history['val_acc']]
    train_loss = history['train_loss']
    val_loss = history['val_loss']
    
    epochs = range(1, len(train_acc) + 1)

    # Vẽ đồ thị Accuracy
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_acc, 'bo-', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'ro-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Vẽ đồ thị Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_loss, 'bo-', label='Training Loss')
    plt.plot(epochs, val_loss, 'ro-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show() # Hiển thị đồ thị

def evaluate_on_test_set(model, test_loader, device='cpu'):
    """
    Hàm này lấy các dự đoán (preds) và nhãn thật (labels) từ tập test.
    """
    model.to(device)
    model.eval() # Chuyển sang chế độ đánh giá
    
    all_labels = []
    all_preds = []
    
    with torch.no_grad(): # Không cần tính gradient
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            
    return all_labels, all_preds

def plot_confusion_matrix_and_report(y_true, y_pred, class_names):
    """
    Vẽ ma trận nhầm lẫn (confusion matrix) và in báo cáo phân loại.
    """
    # In Báo cáo
    print("--- Classification Report ---")
    print(classification_report(y_true, y_pred, target_names=class_names))
    print("-------------------------------")

    # Vẽ Ma trận nhầm lẫn
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, 
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show() # Hiển thị đồ thị
