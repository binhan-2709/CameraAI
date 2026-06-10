# CamAI Attendance

Hệ thống chấm công bằng nhận diện khuôn mặt qua camera, gồm pipeline thu dữ liệu, tạo embedding, lưu FAISS database, API FastAPI và dashboard báo cáo.

## Kiến trúc

```text
data/raw/              ảnh gốc theo nhân viên
data/augmented/        ảnh sau augmentation
models/                model weights và face_db.pkl
src/config.py          cấu hình tập trung từ .env
src/detection/         phát hiện mặt và tạo embedding
src/recognition/       nhận diện bằng cosine similarity / FAISS
src/api/               FastAPI, ORM, logic chấm công
src/pipeline/          realtime camera preview
src/data/              thu ảnh và augmentation
frontend/              dashboard web, chay rieng voi backend API
frontend/assets/       CSS va JavaScript cua dashboard
static/                dashboard cu, khong con duoc backend serve mac dinh
tests/                 unit tests nhẹ
```

## Cài đặt

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Nếu cài PyTorch CPU riêng trên Windows:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## Quy trình sử dụng

1. Thu ảnh nhân viên:

```bash
python src/data/collect_faces.py
```

2. Tăng dữ liệu:

```bash
python src/data/augment.py
```

3. Tạo database nhận diện:

```bash
python src/recognition/build_database.py
```

4. Chạy backend API:

```bash
uvicorn src.api.main:app --reload --port 8000
```

API docs: `http://localhost:8000/docs`

5. Chạy frontend dashboard:

```bash
python -m http.server 5500 --directory frontend
```

Dashboard: `http://localhost:5500`

Frontend mac dinh goi backend tai `http://localhost:8000`. Co the doi o o "Backend API" tren giao dien.

Dashboard co san:

- Tong quan cham cong theo ngay.
- Loc co mat, di muon, vang mat, chua checkout.
- Check-in/check-out thu cong cho truong hop can xu ly tai quay.
- Xuat bao cao Excel.
- Danh sach nhan vien va trang thai he thong.

6. Chạy realtime preview:

```bash
python src/pipeline/realtime_pipeline.py
```

## Cấu hình `.env`

Các biến quan trọng:

```env
CAMERA_SOURCE=0
FACE_BACKEND=auto
FACE_THRESHOLD=0.5
CHECKIN_COOLDOWN_MINUTES=5
LATE_THRESHOLD=08:30
DATABASE_URL=sqlite:///data/attendance.db
FACE_DB_PATH=models/face_db.pkl
DATA_RAW_DIR=data/raw
DATA_AUG_DIR=data/augmented
```

`FACE_BACKEND=auto` sẽ ưu tiên YOLO nếu có `models/yolov8n-face.pt`, sau đó MTCNN, cuối cùng fallback Haar Cascade để app không chết khi thiếu model.

## API chính

- `GET /health`: trạng thái hệ thống.
- `POST /api/employees/register`: upload ảnh và đăng ký nhân viên.
- `GET /api/employees`: danh sách nhân viên active.
- `DELETE /api/employees/{employee_id}`: vô hiệu hóa nhân viên và xóa embedding.
- `POST /api/recognize`: nhận diện từ một frame ảnh và tự ghi chấm công.
- `GET /api/attendance/report`: báo cáo ngày, hỗ trợ `format=excel`.
- `POST /api/attendance/manual`: ghi check-in/check-out thủ công.

## Test

```bash
pytest tests/ -v
```

Các test hiện tại tránh tải model nặng để có thể chạy nhanh trong môi trường dev/CI.
