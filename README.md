# Đồ án: Nhận diện sức khỏe cây xà lách (Lettuce Knight)

Đây là đồ án môn học Khai phá Dữ liệu. Project này thực hiện **so sánh 3 mô hình Deep Learning (ResNet50, MobileNetV2, EfficientNet-B0)** để tìm ra mô hình tốt nhất cho bài toán nhận diện cây xà lách khỏe mạnh và không khỏe mạnh.

**Thành viên nhóm:**
* Lê Hữu Phát - 2251068228
* Nguyễn Chí Nguyên - 2251068219
* Nguyễn Anh Khoa - 2251068202

---

## Cấu trúc thư mục

```bash
Datamining-Lettuce-Health/
├── README.md           # File hướng dẫn này
├── report.pdf          # Báo cáo chi tiết (so sánh 3 mô hình)
├── requirements.txt    # Các thư viện cần thiết
├── data/
│   └── download.txt    # Hướng dẫn tải dữ liệu
├── models/
│   └── ResnetModel_PlantHealthy (1).pth  # Model ResNet50 (tốt nhất)
└── src/
    ├── __pycache__/      # (Thư mục Python tự tạo)
    ├── app.py            # Script chạy web app (dùng model tốt nhất)
    ├── preprocessing.py  # Script tiền xử lý, tạo dataloader
    ├── model_training.py # Script định nghĩa CẢ 3 MODEL
    └── evaluation.py     # Script đánh giá model
```
## Hướng dẫn cài đặt và chạy
1. Clone project:
git clone [https://github.com/nguyenanhkhoa24112004-glitch/Datamining-Lettuce-Health.git](https://github.com/nguyenanhkhoa24112004-glitch/Datamining-Lettuce-Health.git)
cd Datamining-Lettuce-Health

2. Tải dữ liệu: Vui lòng truy cập link trong file data/download.txt để tải bộ dữ liệu (plant-health.zip).

Sau đó, giải nén và đặt các thư mục train, val, test vào bên trong thư mục data/.

3. Cài đặt thư viện:
pip3 install -r requirements.txt

4. Chạy Web App Demo: (Web app này sử dụng mô hình ResNet50 là mô hình tốt nhất sau khi so sánh)
streamlit run src/app.py
Sau đó mở trình duyệt và truy cập http://localhost:8501.

Link Demo Online:
Nhóm đã triển khai một phiên bản demo (sử dụng ResNet50) trên Hugging Face Spaces: https://huggingface.co/spaces/main00100/PlantHealth-ResNet50-App

Kết quả
Sau khi so sánh 3 mô hình, mô hình tốt nhất là ResNet50 đạt độ chính xác 97-98%

