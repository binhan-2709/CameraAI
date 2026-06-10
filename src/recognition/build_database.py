"""Build the face embedding database from data/augmented."""
from __future__ import annotations

from pathlib import Path

import cv2
from tqdm import tqdm

try:
    from config import DATA_AUG_DIR, FACE_DB_PATH
    from detection.face_detector import FaceDetector
    from recognition.recognizer import FaceRecognizer
except ImportError:  # pragma: no cover
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from config import DATA_AUG_DIR, FACE_DB_PATH
    from detection.face_detector import FaceDetector
    from recognition.recognizer import FaceRecognizer


def build(data_dir: str = DATA_AUG_DIR, db_path: str = FACE_DB_PATH) -> None:
    detector = FaceDetector()
    recognizer = FaceRecognizer(db_path=db_path)
    recognizer.embeddings = []
    recognizer.labels = []
    recognizer._rebuild_index()

    root = Path(data_dir)
    if not root.exists():
        print(f"[!] Data directory does not exist: {root}")
        return

    person_dirs = sorted(path for path in root.iterdir() if path.is_dir())
    if not person_dirs:
        print(f"[!] No employee folders found in {root}")
        print(f"    Expected: {root}/<employee_id>/*.jpg")
        return

    total_embeddings = 0
    for person_dir in person_dirs:
        employee_id = person_dir.name
        images = list(person_dir.glob("*.jpg")) + list(person_dir.glob("*.jpeg")) + list(person_dir.glob("*.png"))
        embeddings = []

        for image_path in tqdm(images, desc=f"  {employee_id}", leave=False):
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            faces = detector.detect(image)
            if not faces or faces[0].embedding is None:
                continue
            embeddings.append(faces[0].embedding)

        added = recognizer.add_person(employee_id, embeddings)
        total_embeddings += added
        print(f"  {employee_id}: {added} embeddings")

    if total_embeddings == 0:
        print("[!] No embeddings were created. Check image quality and embedding backend.")
        return

    print(f"[OK] Database built: {total_embeddings} embeddings")
    print(f"     Saved to: {db_path}")


if __name__ == "__main__":
    build()
