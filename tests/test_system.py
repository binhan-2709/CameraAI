from __future__ import annotations

from datetime import date, timedelta
from uuid import uuid4

import numpy as np
import pytest
from sqlalchemy.orm import sessionmaker


def _test_db_path() -> str:
    return f"tests/assets/face_db_{uuid4().hex}.pkl"


def test_recognizer_add_and_match():
    from recognition.recognizer import FaceRecognizer

    db_path = _test_db_path()
    recognizer = FaceRecognizer(db_path=db_path, threshold=0.7)
    recognizer.save = lambda: None
    emb = np.ones(512, dtype="float32")

    assert recognizer.add_person("NV001", [emb]) == 1

    result = recognizer.recognize(emb)
    assert result["identity"] == "NV001"
    assert result["status"] == "recognized"
    assert result["confidence"] >= 0.99


def test_recognizer_handles_missing_embedding():
    from recognition.recognizer import FaceRecognizer

    db_path = _test_db_path()
    recognizer = FaceRecognizer(db_path=db_path)
    result = recognizer.recognize(None)

    assert result == {"identity": "unknown", "confidence": 0.0, "status": "no_embedding"}


def test_daily_report_includes_absent_active_employee():
    from api.attendance_service import AttendanceService, Employee, get_engine

    engine = get_engine("sqlite:///:memory:")
    Session = sessionmaker(bind=engine)

    with Session() as db:
        db.add(Employee(id="NV001", name="An", department="AI", active=True))
        db.commit()

        report = AttendanceService(db).get_daily_report(date.today())

    assert report == [
        {
            "employee_id": "NV001",
            "name": "An",
            "department": "AI",
            "check_in": None,
            "check_out": None,
            "work_hours": None,
            "late": False,
            "status": "absent",
        }
    ]


def test_attendance_record_checkin_then_cooldown():
    from api.attendance_service import AttendanceService, Employee, get_engine

    engine = get_engine("sqlite:///:memory:")
    Session = sessionmaker(bind=engine)

    with Session() as db:
        db.add(Employee(id="NV001", name="An", department="AI", active=True))
        db.commit()

        service = AttendanceService(db)
        first = service.record("NV001", 0.95)
        second = service.record("NV001", 0.96)

    assert first["logged"] is True
    assert first["event_type"] == "check_in"
    assert second["logged"] is False
    assert second["reason"] == "cooldown"


def test_attendance_checkout_after_cooldown():
    from api.attendance_service import AttendanceLog, AttendanceService, Employee, get_engine

    engine = get_engine("sqlite:///:memory:")
    Session = sessionmaker(bind=engine)

    with Session() as db:
        db.add(Employee(id="NV001", name="An", department="AI", active=True))
        db.commit()

        service = AttendanceService(db)
        first = service.record("NV001", 0.95)
        log = db.query(AttendanceLog).filter_by(employee_id="NV001").one()
        log.timestamp = log.timestamp - timedelta(minutes=10)
        db.commit()

        second = service.record("NV001", 0.96)

    assert first["event_type"] == "check_in"
    assert second["logged"] is True
    assert second["event_type"] == "check_out"


def test_manual_checkout_requires_checkin():
    from api.attendance_service import AttendanceService, Employee, get_engine

    engine = get_engine("sqlite:///:memory:")
    Session = sessionmaker(bind=engine)

    with Session() as db:
        db.add(Employee(id="NV001", name="An", department="AI", active=True))
        db.commit()

        result = AttendanceService(db).record_event("NV001", "check_out")

    assert result["logged"] is False
    assert result["reason"] == "missing_check_in"


def test_manual_checkin_checkout_sequence():
    from api.attendance_service import AttendanceService, Employee, get_engine

    engine = get_engine("sqlite:///:memory:")
    Session = sessionmaker(bind=engine)

    with Session() as db:
        db.add(Employee(id="NV001", name="An", department="AI", active=True))
        db.commit()

        service = AttendanceService(db)
        checkin = service.record_event("NV001", "check_in")
        checkout = service.record_event("NV001", "check_out")
        duplicate = service.record_event("NV001", "check_out")

    assert checkin["logged"] is True
    assert checkin["event_type"] == "check_in"
    assert checkout["logged"] is True
    assert checkout["event_type"] == "check_out"
    assert duplicate["logged"] is False
    assert duplicate["reason"] == "already_checked_out"
