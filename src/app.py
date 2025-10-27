import streamlit as st
import torch
from PIL import Image
import os

# Import các hàm tiền xử lý từ module cục bộ
from preprocessing import test_transforms 

# --- CẤU HÌNH VÀ TẢI MODEL ---

# Đường dẫn đến model đã huấn luyện
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'ResnetModel_PlantHealthy (1).pth')

# Xác định thiết bị
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Tên 2 lớp
class_names = ['khỏe mạnh', 'không khỏe mạnh'] 

@st.cache_resource
def load_model():
    """
    Tải model đã huấn luyện từ file .pth (dạng TorchScript).
    """
    # Tải toàn bộ model đã lưu (đã sửa lỗi TypeError)
    model = torch.jit.load(MODEL_PATH, map_location=device)
    model.to(device)
    model.eval() # Chuyển sang chế độ đánh giá
    return model

def predict(image):
    """
    Hàm nhận ảnh PIL, tiền xử lý và trả về dự đoán
    """
    # Tiền xử lý ảnh (dùng hàm import từ preprocessing.py)
    img_tensor = test_transforms(image).unsqueeze(0).to(device)
    
    # Dự đoán
    with torch.no_grad():
        outputs = model(img_tensor)
        _, preds = torch.max(outputs, 1)
        label_idx = preds.item()
        label = class_names[label_idx]
    return label

# Tải model
model = load_model()

# --- GIAO DIỆN WEB STREAMLIT ---

st.set_page_config(page_title="Lettuce Knight", page_icon="🥬", layout="wide")

# CSS tùy chỉnh để cải thiện giao diện
st.markdown("""
    <style>
    .main {
        background-color: #f5f5ff; /* Màu nền xám nhạt */
    }
    /* Style cho khung tải ảnh */
    [data-testid="stFileUploader"] {
        border: 2px dashed #4CAF50;
        border-radius: 10px;
        padding: 25px;
        background-color: #FAFAFA;
    }
    [data-testid="stFileUploader"] > label {
        color: #4CAF50;
        font-size: 1.2em;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🥬 Lettuce Knight - Hiệp sĩ Xà lách")
st.write("---")
st.markdown("**Chào mừng!** Tải lên một hình ảnh của cây rau xà lách để kiểm tra tình trạng sức khỏe của cây.")

# --- PHẦN TẢI ẢNH LÊN ---
uploaded_file = st.file_uploader("Chọn một hình ảnh (JPG, PNG, JPEG)...", type=["jpg", "png", "jpeg"], label_visibility="collapsed")

st.write("---")

# Hiển thị kết quả
if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert('RGB')
        
        # Chia 2 cột cho kết quả
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption='Ảnh đã tải lên.', use_column_width=True)

        with col2:
            with st.spinner('Đang phân tích...'):
                label = predict(image)
    
            st.subheader("Kết quả phân tích:")
            
            # Hiển thị kết quả và gợi ý giải pháp
            if label == "không khỏe mạnh":
                st.warning(f"⚠️ **Cảnh báo:** Rau có dấu hiệu bị stress!")
                st.markdown(
                    """
                    **Gợi ý giải pháp:**
                    - **Kiểm tra sâu bệnh:** Tìm kiếm dấu hiệu của sâu, rệp, hoặc nấm.
                    - **Tưới nước:** Đảm bảo cây được cung cấp đủ nước, tránh úng hoặc khô hạn.
                    - **Dinh dưỡng:** Bổ sung phân bón hữu cơ hoặc vi lượng.
                    - **Ánh sáng:** Đảm bảo cây nhận đủ ánh sáng mặt trời.
                    """
                )
            else:
                st.balloons()
                st.success(f"✅ **Tuyệt vời!** Rau của bạn đang phát triển khỏe mạnh. Hãy tiếp tục chăm sóc tốt nhé!")

    except Exception as e:
        st.error(f"Đã có lỗi xảy ra khi xử lý ảnh: {e}")
