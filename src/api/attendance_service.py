"""Attendance persistence and business rules."""
from __future__ import annotations

import uuid
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path

from sqlalchemy import Boolean, Column, DateTime, Float, String, create_engine
from sqlalchemy.orm import Session, declarative_base

try:
    from config import BASE_DIR, DATABASE_URL, get_runtime_settings
except ImportError:  # pragma: no cover
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from config import BASE_DIR, DATABASE_URL, get_runtime_settings


Base = declarative_base()


class Employee(Base):
    __tablename__ = "employees"

    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    department = Column(String, default="")
    active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.now)


class AttendanceLog(Base):
    __tablename__ = "attendance_logs"

    id = Column(String, primary_key=True)
    employee_id = Column(String, nullable=False, index=True)
    timestamp = Column(DateTime, default=datetime.now, index=True)
    event_type = Column(String, nullable=False)  # check_in | check_out
    confidence = Column(Float, default=0.0)
    camera_id = Column(String, default="CAM_01")
    is_real_face = Column(Boolean, default=True)
    liveness_score = Column(Float, default=1.0)
    emotion = Column(String, default="neutral")


def _normalize_database_url(url: str) -> str:
    if url == "sqlite:///:memory:":
        return url
    if not url.startswith("sqlite:///") or url.startswith("sqlite:////"):
        return url

    relative_path = url.replace("sqlite:///", "", 1)
    if Path(relative_path).is_absolute():
        return url
    return f"sqlite:///{(BASE_DIR / relative_path).as_posix()}"


def get_engine(url: str | None = None):
    db_url = _normalize_database_url(url or DATABASE_URL)
    if db_url.startswith("sqlite:///"):
        db_path = Path(db_url.replace("sqlite:///", "", 1))
        db_path.parent.mkdir(parents=True, exist_ok=True)
        engine = create_engine(db_url, echo=False, connect_args={"check_same_thread": False})
    else:
        engine = create_engine(db_url, echo=False)
    Base.metadata.create_all(engine)

    # Run manual migration for new columns if they are missing
    from sqlalchemy import text
    with engine.begin() as conn:
        for column_name, sql_type in [
            ("liveness_score", "FLOAT DEFAULT 1.0"),
            ("is_real_face", "BOOLEAN DEFAULT 1"),
            ("emotion", "VARCHAR DEFAULT 'neutral'"),
        ]:
            try:
                conn.execute(text(f"ALTER TABLE attendance_logs ADD COLUMN {column_name} {sql_type}"))
            except Exception:
                # Column might already exist or table is not created yet
                pass

    return engine


class AttendanceService:
    def __init__(self, db: Session):
        self.db = db

    def record_event(
        self,
        employee_id: str,
        event_type: str,
        confidence: float = 1.0,
        camera_id: str = "MANUAL",
        is_real: bool = True,
        liveness_score: float = 1.0,
        emotion: str = "neutral",
    ) -> dict:
        if event_type not in {"check_in", "check_out"}:
            return {"logged": False, "reason": "invalid_event_type"}

        now = datetime.now()
        today_start = datetime.combine(date.today(), datetime.min.time())
        employee = self.db.get(Employee, employee_id)
        if employee is None:
            return {"logged": False, "reason": "employee_not_found"}
        if not employee.active:
            return {"logged": False, "reason": "inactive_employee"}

        today_logs = (
            self.db.query(AttendanceLog)
            .filter(
                AttendanceLog.employee_id == employee_id,
                AttendanceLog.timestamp >= today_start,
            )
            .order_by(AttendanceLog.timestamp)
            .all()
        )
        has_check_in = any(log.event_type == "check_in" for log in today_logs)
        has_check_out = any(log.event_type == "check_out" for log in today_logs)

        if event_type == "check_in" and has_check_in:
            return {"logged": False, "reason": "already_checked_in"}
        if event_type == "check_out" and not has_check_in:
            if camera_id == "REALTIME_CAMERA":
                check_in_time = now - timedelta(seconds=1)
                check_in_log = AttendanceLog(
                    id=str(uuid.uuid4()),
                    employee_id=employee_id,
                    timestamp=check_in_time,
                    event_type="check_in",
                    confidence=float(confidence),
                    camera_id=camera_id,
                    is_real_face=is_real,
                    liveness_score=float(liveness_score),
                    emotion=str(emotion),
                )
                self.db.add(check_in_log)
                has_check_in = True
            else:
                return {"logged": False, "reason": "missing_check_in"}
        if event_type == "check_out" and has_check_out:
            return {"logged": False, "reason": "already_checked_out"}

        log = AttendanceLog(
            id=str(uuid.uuid4()),
            employee_id=employee_id,
            timestamp=now,
            event_type=event_type,
            confidence=float(confidence),
            camera_id=camera_id,
            is_real_face=is_real,
            liveness_score=float(liveness_score),
            emotion=str(emotion),
        )
        self.db.add(log)
        self.db.commit()

        return {
            "logged": True,
            "event_type": event_type,
            "employee_id": employee_id,
            "timestamp": now.isoformat(),
            "confidence": round(float(confidence), 4),
            "is_real_face": is_real,
            "liveness_score": round(float(liveness_score), 4),
            "emotion": str(emotion),
        }

    def record(
        self,
        employee_id: str,
        confidence: float,
        camera_id: str = "CAM_01",
        is_real: bool = True,
        liveness_score: float = 1.0,
        emotion: str = "neutral",
    ) -> dict:
        now = datetime.now()
        today_start = datetime.combine(date.today(), datetime.min.time())

        employee = self.db.get(Employee, employee_id)
        if employee is not None and not employee.active:
            return {"logged": False, "reason": "inactive_employee"}

        runtime_cooldown = get_runtime_settings()["cooldown_minutes"]
        recent = (
            self.db.query(AttendanceLog)
            .filter(
                AttendanceLog.employee_id == employee_id,
                AttendanceLog.timestamp >= now - timedelta(minutes=runtime_cooldown),
            )
            .order_by(AttendanceLog.timestamp.desc())
            .first()
        )
        if recent:
            next_ok = recent.timestamp + timedelta(minutes=runtime_cooldown)
            return {
                "logged": False,
                "reason": "cooldown",
                "next_allowed": next_ok.isoformat(),
            }

        has_check_in = (
            self.db.query(AttendanceLog)
            .filter(
                AttendanceLog.employee_id == employee_id,
                AttendanceLog.timestamp >= today_start,
                AttendanceLog.event_type == "check_in",
            )
            .first()
            is not None
        )
        event_type = "check_out" if has_check_in else "check_in"

        log = AttendanceLog(
            id=str(uuid.uuid4()),
            employee_id=employee_id,
            timestamp=now,
            event_type=event_type,
            confidence=float(confidence),
            camera_id=camera_id,
            is_real_face=is_real,
            liveness_score=float(liveness_score),
            emotion=str(emotion),
        )
        self.db.add(log)
        self.db.commit()

        return {
            "logged": True,
            "event_type": event_type,
            "employee_id": employee_id,
            "timestamp": now.isoformat(),
            "confidence": round(float(confidence), 4),
            "is_real_face": is_real,
            "liveness_score": round(float(liveness_score), 4),
            "emotion": str(emotion),
        }

    def get_daily_report(self, target_date: date | None = None) -> list[dict]:
        d = target_date or date.today()
        start = datetime.combine(d, datetime.min.time())
        end = datetime.combine(d, datetime.max.time())
        runtime_late_threshold = get_runtime_settings()["late_threshold"]
        late_time = datetime.strptime(runtime_late_threshold, "%H:%M").time()

        logs = (
            self.db.query(AttendanceLog)
            .filter(AttendanceLog.timestamp.between(start, end))
            .order_by(AttendanceLog.timestamp)
            .all()
        )
        grouped: dict[str, list[AttendanceLog]] = defaultdict(list)
        for log in logs:
            grouped[log.employee_id].append(log)

        employees = self.db.query(Employee).filter(Employee.active == True).all()  # noqa: E712
        employee_ids = {employee.id for employee in employees} | set(grouped.keys())
        employees_by_id = {employee.id: employee for employee in employees}

        result: list[dict] = []
        for emp_id in sorted(employee_ids):
            emp_logs = grouped.get(emp_id, [])
            emp = employees_by_id.get(emp_id) or self.db.get(Employee, emp_id)
            check_ins = [log for log in emp_logs if log.event_type == "check_in"]
            check_outs = [log for log in emp_logs if log.event_type == "check_out"]
            first_in = check_ins[0].timestamp if check_ins else None
            last_out = check_outs[-1].timestamp if check_outs else None
            work_hours = (
                round((last_out - first_in).total_seconds() / 3600, 2)
                if first_in and last_out and last_out >= first_in
                else None
            )

            result.append(
                {
                    "employee_id": emp_id,
                    "name": emp.name if emp else emp_id,
                    "department": emp.department if emp else "",
                    "check_in": first_in.strftime("%H:%M:%S") if first_in else None,
                    "check_out": last_out.strftime("%H:%M:%S") if last_out else None,
                    "work_hours": work_hours,
                    "late": bool(first_in and first_in.time() > late_time),
                    "status": "present" if first_in else "absent",
                }
            )
        return result

    def get_recent_logs(self, limit: int = 20) -> list[dict]:
        limit = max(1, min(limit, 100))
        logs = (
            self.db.query(AttendanceLog)
            .order_by(AttendanceLog.timestamp.desc())
            .limit(limit)
            .all()
        )

        result = []
        for log in logs:
            employee = self.db.get(Employee, log.employee_id)
            result.append(
                {
                    "employee_id": log.employee_id,
                    "name": employee.name if employee else log.employee_id,
                    "department": employee.department if employee else "",
                    "event_type": log.event_type,
                    "timestamp": log.timestamp.isoformat(),
                    "time": log.timestamp.strftime("%H:%M:%S"),
                    "confidence": round(float(log.confidence or 0.0), 4),
                    "camera_id": log.camera_id,
                }
            )
        return result
