"""Collect face images for one employee from a camera."""
from __future__ import annotations

import re
import platform
import time
from pathlib import Path

import cv2

try:
    from config import CAMERA_SOURCE, DATA_RAW_DIR
except ImportError:  # pragma: no cover
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from config import CAMERA_SOURCE, DATA_RAW_DIR


EMPLOYEE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{2,32}$")


def _open_camera(source):
    """Open camera with Windows-friendly fallbacks."""
    candidates = [source]
    if source == 0:
        candidates.extend([1, 2])

    backends = [cv2.CAP_ANY]
    if platform.system() == "Windows":
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]

    for candidate in candidates:
        for backend in backends:
            cap = cv2.VideoCapture(candidate, backend)
            if not cap.isOpened():
                cap.release()
                continue

            # Warm up and verify the camera actually returns frames.
            for _ in range(5):
                cap.read()
            ok, frame = cap.read()
            if ok and frame is not None:
                print(f"Using camera source={candidate}, backend={backend}")
                return cap
            cap.release()

    return None


def collect(employee_id: str, n_samples: int = 50, save_dir: str = DATA_RAW_DIR) -> int:
    employee_id = employee_id.strip()
    if not EMPLOYEE_ID_RE.fullmatch(employee_id):
        raise ValueError("employee_id must be 2-32 chars: letters, numbers, underscore or dash")

    save_path = Path(save_dir) / employee_id
    save_path.mkdir(parents=True, exist_ok=True)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    cap = _open_camera(CAMERA_SOURCE)
    if cap is None:
        print("[!] Cannot open camera or read frames")
        return 0

    count = 0
    print(f"Collecting images for {employee_id}")
    print("Press SPACE to capture, Q to quit.")

    while count < n_samples:
        ok, frame = cap.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))
        display = frame.copy()
        for x, y, w, h in faces:
            cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)

        status = "face detected" if len(faces) else "no face"
        color = (0, 255, 0) if len(faces) else (0, 0, 255)
        cv2.putText(display, f"{employee_id} | {count}/{n_samples}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(display, status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.imshow("CamAI data collection", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" ") and len(faces):
            x, y, w, h = faces[0]
            pad = 20
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(frame.shape[1], x + w + pad)
            y2 = min(frame.shape[0], y + h + pad)
            face_img = cv2.resize(frame[y1:y2, x1:x2], (160, 160))
            filename = save_path / f"{employee_id}_{count:03d}.jpg"
            cv2.imwrite(str(filename), face_img)
            count += 1
            print(f"[{count:02d}/{n_samples}] saved {filename}")
            time.sleep(0.15)

    cap.release()
    cv2.destroyAllWindows()
    print(f"Done. Saved {count} images to {save_path}")
    return count


if __name__ == "__main__":
    emp_id = input("Employee ID (example NV001): ").strip()
    raw_n = input("Number of images [50]: ").strip()
    collect(emp_id, n_samples=int(raw_n) if raw_n.isdigit() else 50)
