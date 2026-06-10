"""FastAPI application for CamAI Attendance."""
from __future__ import annotations

import io
import platform
import re
import time
from collections import deque
from contextlib import asynccontextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session, sessionmaker

try:
    from api.attendance_service import AttendanceService, Employee, get_engine
    from config import CAMERA_HEIGHT, CAMERA_SOURCE, CAMERA_WIDTH, DATA_RAW_DIR
    from detection.face_detector import FaceDetector
    from recognition.recognizer import FaceRecognizer
except ImportError:  # pragma: no cover
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from api.attendance_service import AttendanceService, Employee, get_engine
    from config import CAMERA_HEIGHT, CAMERA_SOURCE, CAMERA_WIDTH, DATA_RAW_DIR
    from detection.face_detector import FaceDetector
    from recognition.recognizer import FaceRecognizer


EMPLOYEE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{2,32}$")
ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MAX_UPLOAD_BYTES = 8 * 1024 * 1024

engine = None
SessionLocal = None
detector: FaceDetector | None = None
recognizer: FaceRecognizer | None = None


class ManualAttendanceRequest(BaseModel):
    employee_id: str = Field(..., min_length=2, max_length=32)
    event_type: Literal["check_in", "check_out"]
    camera_id: str = "MANUAL"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine, SessionLocal, detector, recognizer
    engine = get_engine()
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    detector = FaceDetector()
    recognizer = FaceRecognizer()
    print(
        "[CamAI] ready: "
        f"detector={detector.active_detector}, embeddings={detector.embedder is not None}, "
        f"face_db={recognizer.total}"
    )
    yield


app = FastAPI(
    title="CamAI Attendance API",
    description="Realtime face recognition attendance service",
    version="1.1.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_db():
    if SessionLocal is None:
        raise HTTPException(status_code=503, detail="Database is not initialized")
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _require_models() -> tuple[FaceDetector, FaceRecognizer]:
    if detector is None or recognizer is None:
        raise HTTPException(status_code=503, detail="AI models are not initialized")
    return detector, recognizer


def _validate_employee_id(employee_id: str) -> str:
    employee_id = employee_id.strip()
    if not EMPLOYEE_ID_RE.fullmatch(employee_id):
        raise HTTPException(
            status_code=400,
            detail="employee_id must be 2-32 chars: letters, numbers, underscore or dash",
        )
    return employee_id


async def _read_image_upload(file: UploadFile) -> np.ndarray:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_IMAGE_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Only JPG, PNG, or WEBP images are accepted")

    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image is too large")

    arr = np.frombuffer(data, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Cannot decode image")
    return frame


def _open_camera_capture():
    if isinstance(CAMERA_SOURCE, int):
        candidates = [CAMERA_SOURCE, 1, 2] if CAMERA_SOURCE == 0 else [CAMERA_SOURCE]
        backends = [cv2.CAP_ANY]
        if platform.system() == "Windows":
            backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    else:
        candidates = [CAMERA_SOURCE]
        backends = [cv2.CAP_FFMPEG, cv2.CAP_ANY]

    for source in candidates:
        for backend in backends:
            cap = cv2.VideoCapture(source, backend)
            if not cap.isOpened():
                cap.release()
                continue
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

            for _ in range(5):
                cap.read()
            ok, frame = cap.read()
            if ok and frame is not None:
                return cap
            cap.release()
    return None


def _draw_face_label(frame: np.ndarray, bbox: list[int], label: str, confidence: float, known: bool) -> None:
    color = (0, 220, 80) if known else (0, 80, 220)
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

    text = f"{label} {confidence:.0%}" if known else "unknown"
    scale = 0.7
    thickness = 2
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)

    label_bg_padding = 6
    label_y = max(0, y1 - th - label_bg_padding * 2)
    cv2.rectangle(frame, (x1 - label_bg_padding, label_y), (x1 + tw + label_bg_padding, y1), color, -1)
    cv2.putText(
        frame,
        text,
        (x1, y1 - label_bg_padding),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def _draw_camera_header(frame: np.ndarray) -> None:
    timestamp = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
    cv2.putText(
        frame,
        timestamp,
        (max(10, frame.shape[1] - 450), 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def _draw_detection_history(frame: np.ndarray, history: deque) -> None:
    """Draw detection history panel on the left side of frame."""
    h, w = frame.shape[:2]
    panel_width = 290

    cv2.rectangle(frame, (0, 0), (panel_width, h), (25, 25, 35), -1)

    cv2.putText(
        frame,
        "Detected Persons",
        (15, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.line(frame, (10, 50), (panel_width - 10, 50), (80, 80, 100), 2)

    y_pos = 75
    row_height = 50

    for detection in list(history)[:6]:
        if y_pos + row_height > h - 20:
            break

        display_name = detection["display_name"]
        timestamp = detection["timestamp"].strftime("%H:%M:%S")
        confidence = int(detection["confidence"] * 100)

        cv2.putText(
            frame,
            display_name,
            (20, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 220, 80),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            f"{timestamp}",
            (20, y_pos + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (150, 150, 180),
            1,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            f"Conf: {confidence}%",
            (20, y_pos + 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (100, 200, 255),
            1,
        )

        cv2.line(frame, (12, y_pos + 45), (panel_width - 12, y_pos + 45), (50, 50, 70), 1)
        y_pos += row_height


def _camera_frame_generator(record_attendance: bool = True, every_n_frames: int = 2):
    det, rec_model = _require_models()
    if SessionLocal is None:
        raise HTTPException(status_code=503, detail="Database is not initialized")

    cap = _open_camera_capture()
    if cap is None:
        raise HTTPException(status_code=503, detail="Cannot open camera source")

    db = SessionLocal()
    svc = AttendanceService(db)
    frame_index = 0
    last_results: list[dict] = []
    detection_history: deque = deque(maxlen=15)

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.05)
                continue

            frame_index += 1
            if frame_index % max(1, every_n_frames) == 0:
                faces = det.detect(frame)
                last_results = []
                for face in faces:
                    rec = rec_model.recognize(face.embedding)
                    identity = rec["identity"]
                    known = identity != "unknown"
                    display_name = identity
                    attendance = None

                    if known:
                        employee = db.get(Employee, identity)
                        display_name = employee.name if employee else identity
                        if record_attendance:
                            attendance = svc.record(
                                employee_id=identity,
                                confidence=rec["confidence"],
                                camera_id="LIVE_CAMERA",
                                is_real=True,
                            )

                        detection_history.append({
                            "identity": identity,
                            "display_name": display_name,
                            "confidence": rec["confidence"],
                            "timestamp": datetime.now(),
                        })

                    last_results.append(
                        {
                            "bbox": [int(value) for value in face.bbox.tolist()],
                            "identity": identity,
                            "display_name": display_name,
                            "confidence": rec["confidence"],
                            "known": known,
                            "attendance": attendance,
                        }
                    )

            display = frame.copy()
            h, w = display.shape[:2]

            _draw_camera_header(display)
            for result in last_results:
                _draw_face_label(
                    display,
                    result["bbox"],
                    result["display_name"],
                    result["confidence"],
                    result["known"],
                )

            _draw_detection_history(display, detection_history)

            ok, encoded = cv2.imencode(".jpg", display, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
            if not ok:
                continue
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + encoded.tobytes()
                + b"\r\n"
            )
    finally:
        db.close()
        cap.release()


@app.get("/health", tags=["System"])
def health():
    return {
        "status": "ok",
        "detector": detector.active_detector if detector else None,
        "embeddings_enabled": bool(detector and detector.embedder is not None),
        "db_size": recognizer.total if recognizer else 0,
        "people": recognizer.people if recognizer else [],
    }


@app.post("/api/recognize", tags=["Recognition"])
async def recognize(
    file: UploadFile = File(..., description="Camera frame as JPG/PNG/WEBP"),
    camera_id: str = "CAM_01",
    db: Session = Depends(get_db),
):
    det, rec_model = _require_models()
    frame = await _read_image_upload(file)
    faces = det.detect(frame)
    if not faces:
        return {"status": "no_face", "face_count": 0, "results": []}

    svc = AttendanceService(db)
    results = []
    for face in faces:
        rec = rec_model.recognize(face.embedding)
        attendance = None
        if rec["identity"] != "unknown":
            attendance = svc.record(
                employee_id=rec["identity"],
                confidence=rec["confidence"],
                camera_id=camera_id,
                is_real=True,
            )
        results.append({**rec, "attendance": attendance, "bbox": face.bbox.tolist()})

    return {"status": "ok", "face_count": len(faces), "results": results}


@app.post("/api/employees/register", tags=["Employees"])
async def register(
    employee_id: str = Form(...),
    name: str = Form(...),
    department: str = Form(""),
    files: list[UploadFile] = File(...),
    db: Session = Depends(get_db),
):
    det, rec_model = _require_models()
    employee_id = _validate_employee_id(employee_id)
    if len(files) < 3:
        raise HTTPException(status_code=400, detail="Upload at least 3 clear face images")

    save_dir = Path(DATA_RAW_DIR) / employee_id
    save_dir.mkdir(parents=True, exist_ok=True)

    new_embeddings = []
    saved = 0
    for idx, upload in enumerate(files):
        frame = await _read_image_upload(upload)
        detected = det.detect(frame)
        if not detected or detected[0].embedding is None:
            continue

        suffix = Path(upload.filename or ".jpg").suffix.lower()
        path = save_dir / f"{employee_id}_{idx:03d}{suffix}"
        _, encoded = cv2.imencode(suffix if suffix != ".jpg" else ".jpg", frame)
        with open(path, "wb") as file:
            file.write(encoded.tobytes())

        new_embeddings.append(detected[0].embedding)
        saved += 1

    if not new_embeddings:
        raise HTTPException(status_code=400, detail="No usable face embeddings found in upload")

    added = rec_model.add_person(employee_id, new_embeddings)
    emp = Employee(id=employee_id, name=name.strip(), department=department.strip(), active=True)
    db.merge(emp)
    db.commit()

    return {
        "status": "success",
        "employee_id": employee_id,
        "images_saved": saved,
        "embeddings_added": added,
        "total_in_db": rec_model.total,
    }


@app.delete("/api/employees/{employee_id}", tags=["Employees"])
def remove_employee(employee_id: str, db: Session = Depends(get_db)):
    _, rec_model = _require_models()
    employee_id = _validate_employee_id(employee_id)
    removed = rec_model.remove_person(employee_id)
    emp = db.get(Employee, employee_id)
    if emp:
        emp.active = False
        db.commit()
    return {"status": "removed", "employee_id": employee_id, "embeddings_removed": removed}


@app.get("/api/employees", tags=["Employees"])
def list_employees(db: Session = Depends(get_db)):
    employees = db.query(Employee).filter(Employee.active == True).order_by(Employee.id).all()  # noqa: E712
    return [
        {"id": employee.id, "name": employee.name, "department": employee.department}
        for employee in employees
    ]


@app.get("/api/attendance/report", tags=["Attendance"])
def attendance_report(
    target_date: str | None = None,
    format: str = "json",
    db: Session = Depends(get_db),
):
    try:
        report_date = date.fromisoformat(target_date) if target_date else date.today()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="target_date must use YYYY-MM-DD") from exc

    report = AttendanceService(db).get_daily_report(report_date)
    if format != "excel":
        return JSONResponse({"date": str(report_date), "data": report})

    import openpyxl
    from openpyxl.styles import Alignment, Font, PatternFill

    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = f"Attendance_{report_date}"
    headers = [
        "Employee ID",
        "Name",
        "Department",
        "Check In",
        "Check Out",
        "Hours",
        "Late",
        "Status",
    ]
    sheet.append(headers)
    for cell in sheet[1]:
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="1D4ED8")
        cell.alignment = Alignment(horizontal="center")

    for row in report:
        sheet.append(
            [
                row["employee_id"],
                row["name"],
                row.get("department", ""),
                row["check_in"] or "-",
                row["check_out"] or "-",
                row["work_hours"] or "-",
                "Yes" if row["late"] else "No",
                row["status"],
            ]
        )

    for column in sheet.columns:
        width = max(len(str(cell.value or "")) for cell in column) + 4
        sheet.column_dimensions[column[0].column_letter].width = width

    buffer = io.BytesIO()
    workbook.save(buffer)
    buffer.seek(0)
    return StreamingResponse(
        buffer,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f"attachment; filename=attendance_{report_date}.xlsx"},
    )


@app.post("/api/attendance/manual", tags=["Attendance"])
def manual_attendance(payload: ManualAttendanceRequest, db: Session = Depends(get_db)):
    employee_id = _validate_employee_id(payload.employee_id)
    result = AttendanceService(db).record_event(
        employee_id=employee_id,
        event_type=payload.event_type,
        confidence=1.0,
        camera_id=payload.camera_id,
        is_real=True,
    )
    if not result.get("logged"):
        return JSONResponse(status_code=409, content=result)
    return result
