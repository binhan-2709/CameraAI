# CamAI Frontend

Dashboard tách riêng khỏi backend FastAPI.

## Cấu trúc

```text
frontend/
├── index.html
└── assets/
    ├── css/main.css
    └── js/app.js
```

## Chạy frontend

Từ thư mục gốc project:

```powershell
python -m http.server 5500 --directory frontend
```

Mở:

```text
http://localhost:5500
```

Backend cần chạy ở:

```text
http://localhost:8000
```

Nếu backend chạy ở IP/port khác, sửa ô `Backend API` trên sidebar rồi bấm `Lưu`.

## Chức năng

- Xem thống kê attendance theo ngày.
- Lọc nhân viên có mặt, đi muộn, vắng mặt, chưa checkout.
- Check-in/check-out thủ công.
- Xuất Excel.
- Xem danh sách nhân viên và trạng thái hệ thống.
