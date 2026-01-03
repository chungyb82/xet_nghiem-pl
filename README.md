# Tổng hợp sổ xét nghiệm bằng Streamlit

Ứng dụng Streamlit đọc các file Excel trong cùng thư mục và tổng hợp theo:
- Tháng/năm, Nơi gửi
- Đối tượng bảo hiểm, thu phí (tính tổng lượt xét nghiệm của dòng thuộc BH/VP)
- Số lượt xét nghiệm theo từng loại (Vi sinh, Sinh hóa, Huyết học, Nước tiểu)
- TAT trung bình chung và TAT theo từng loại (phút)
- Lọc bỏ dòng không có “Người thực hiện”

## Chạy nhanh
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Sử dụng
1) Đặt các file Excel cần xử lý cùng thư mục với `app.py`.
2) Mở trang Streamlit (tự mở trình duyệt) và chọn:
   - Chế độ xử lý: tất cả file hoặc chọn file cụ thể.
   - Bấm “Xử lý dữ liệu”.
3) Xem bảng tổng hợp và tải Excel bằng nút “Tải Excel tổng hợp”.

## Ghi chú xử lý dữ liệu
- Header dữ liệu ở dòng 5–6 (0-index 4–5). Dòng dữ liệu bắt đầu sau đó và chỉ giữ dòng có STT.
- Chỉ giữ dòng có “Người thực hiện” không trống.
- Nơi gửi: chuỗi chứa “phòng khám” hoặc “phòng lưu khoa khám bệnh” được gộp thành “Khoa khám bệnh”; chuỗi rỗng thành “Không rõ”.
- BH/VP: nếu cột BH/VP có dữ liệu, số xét nghiệm của dòng sẽ được cộng vào BH/VP tương ứng; ngược lại là 0.
- TAT: `Thời gian valid (3)` trừ `Thời gian nhận mẫu (2)`, đơn vị phút; trung bình khi gộp.

## Cấu trúc cột đầu ra
`Nơi gửi`, `Tháng/năm`, `Đối tượng bảo hiểm`, `Đối tượng thu phí`,  
`XN vi sinh`, `XN sinh hóa`, `XN huyết học`, `XN nước tiểu`,  
`TAT vi sinh (phút)`, `TAT sinh hóa (phút)`, `TAT huyết học (phút)`, `TAT nước tiểu (phút)`,  
`Thời gian trung bình (phút)`.

## Yêu cầu môi trường
- Python 3.10+ đã cài đặt.
- Thư viện trong `requirements.txt`.


