"""Face detection and embedding extraction.

The detector is intentionally defensive: optional ML backends are loaded lazily
so the API can still start, run health checks, and execute tests when model
weights or heavy dependencies are not installed yet.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image

try:
    from config import BASE_DIR, FACE_BACKEND
except ImportError:  # pragma: no cover - direct module execution fallback
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from config import BASE_DIR, FACE_BACKEND


@dataclass
class FaceObject:
    bbox: np.ndarray
    det_score: float
    embedding: Optional[np.ndarray] = None


class FaceDetector:
    """Detect faces and return InsightFace-like face objects."""

    def __init__(self, backend: str = FACE_BACKEND, model_dir: str | Path | None = None):
        self.backend = backend.lower()
        self.model_dir = Path(model_dir) if model_dir else BASE_DIR / "models"
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.yolo = None
        self.mtcnn = None
        self.embedder = None
        self.transform = None
        self.haar = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        self.active_detector = "haar"
        self._load_detector_backend()
        self._load_embedder()

    def detect(self, frame: np.ndarray) -> list[FaceObject]:
        if frame is None or frame.size == 0:
            return []

        if self.active_detector == "yolo" and self.yolo is not None:
            return self._detect_yolo(frame)
        if self.active_detector == "mtcnn" and self.mtcnn is not None:
            return self._detect_mtcnn(frame)
        return self._detect_haar(frame)

    def draw(
        self,
        frame: np.ndarray,
        faces: list[FaceObject],
        labels: dict[int, str] | None = None,
    ) -> np.ndarray:
        for i, face in enumerate(faces):
            box = face.bbox.astype(int)
            score = face.det_score
            name = (labels or {}).get(i, "")
            known = bool(name and name != "unknown")
            color = (0, 220, 80) if known else (0, 80, 220)

            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 3)

            label = f"{name} ({score:.0%})" if name else f"face ({score:.0%})"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)

            label_bg_padding = 8
            y1 = max(0, box[1] - th - label_bg_padding * 2)
            cv2.rectangle(frame, (box[0] - label_bg_padding, y1), (box[0] + tw + label_bg_padding, box[1]), color, -1)
            cv2.putText(
                frame,
                label,
                (box[0], box[1] - label_bg_padding),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        return frame

    def _load_detector_backend(self) -> None:
        if self.backend in {"auto", "yolo"}:
            model_path = self.model_dir / "yolov8n-face.pt"
            if model_path.exists():
                try:
                    from ultralytics import YOLO

                    self.yolo = YOLO(str(model_path))
                    self.active_detector = "yolo"
                    return
                except Exception as exc:
                    print(f"[FaceDetector] YOLO unavailable: {exc}")
            elif self.backend == "yolo":
                print(f"[FaceDetector] YOLO model not found: {model_path}")

        if self.backend in {"auto", "mtcnn"}:
            try:
                from facenet_pytorch import MTCNN

                self.mtcnn = MTCNN(
                    image_size=160,
                    margin=20,
                    keep_all=True,
                    post_process=False,
                    min_face_size=40,
                    device="cpu",
                )
                self.active_detector = "mtcnn"
                return
            except Exception as exc:
                print(f"[FaceDetector] MTCNN unavailable: {exc}")

        self.active_detector = "haar"

    def _load_embedder(self) -> None:
        try:
            import torch
            from facenet_pytorch import InceptionResnetV1
            from torchvision import transforms

            self.torch = torch
            self.embedder = InceptionResnetV1(pretrained="vggface2").eval()
            self.transform = transforms.Compose(
                [
                    transforms.Resize((160, 160)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ]
            )
        except Exception as exc:
            self.torch = None
            self.embedder = None
            self.transform = None
            print(f"[FaceDetector] Embeddings disabled: {exc}")

    def _detect_yolo(self, frame: np.ndarray) -> list[FaceObject]:
        try:
            faces: list[FaceObject] = []
            h, w = frame.shape[:2]
            results = self.yolo(frame, verbose=False)
            for result in results:
                if result.boxes is None:
                    continue
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    score = float(box.conf[0])
                    face = self._build_face(frame, x1, y1, x2, y2, score, w, h)
                    if face:
                        faces.append(face)
            return faces
        except Exception as exc:
            print(f"[FaceDetector] YOLO detection failed, falling back to Haar: {exc}")
            self.active_detector = "haar"
            return self._detect_haar(frame)

    def _detect_mtcnn(self, frame: np.ndarray) -> list[FaceObject]:
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            h, w = frame.shape[:2]
            boxes, probs = self.mtcnn.detect(pil)
            if boxes is None:
                return []

            faces: list[FaceObject] = []
            for box, prob in zip(boxes, probs):
                if prob is None or prob < 0.8:
                    continue
                face = self._build_face(frame, box[0], box[1], box[2], box[3], float(prob), w, h)
                if face:
                    faces.append(face)
            return faces
        except Exception as exc:
            print(f"[FaceDetector] MTCNN detection failed, falling back to Haar: {exc}")
            self.active_detector = "haar"
            return self._detect_haar(frame)

    def _detect_haar(self, frame: np.ndarray) -> list[FaceObject]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        boxes = self.haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
        faces: list[FaceObject] = []
        h, w = frame.shape[:2]
        for x, y, bw, bh in boxes:
            face = self._build_face(frame, x, y, x + bw, y + bh, 0.75, w, h)
            if face:
                faces.append(face)
        return faces

    def _build_face(
        self,
        frame: np.ndarray,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        score: float,
        frame_w: int,
        frame_h: int,
    ) -> FaceObject | None:
        ix1 = max(0, int(x1))
        iy1 = max(0, int(y1))
        ix2 = min(frame_w, int(x2))
        iy2 = min(frame_h, int(y2))
        if ix2 <= ix1 or iy2 <= iy1:
            return None

        face_crop = frame[iy1:iy2, ix1:ix2]
        embedding = self._get_embedding(face_crop)
        return FaceObject(
            bbox=np.array([ix1, iy1, ix2, iy2], dtype=float),
            det_score=score,
            embedding=embedding,
        )

    def _get_embedding(self, face_bgr: np.ndarray) -> np.ndarray | None:
        if self.embedder is None or self.transform is None or face_bgr is None or face_bgr.size == 0:
            return None
        try:
            rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            tensor = self.transform(pil).unsqueeze(0)
            with self.torch.no_grad():
                emb = self.embedder(tensor)
            return emb.squeeze().numpy().astype("float32")
        except Exception as exc:
            print(f"[FaceDetector] Embedding failed: {exc}")
            return None


if __name__ == "__main__":
    detector = FaceDetector()
    print(f"Detector backend: {detector.active_detector}")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera is not available.")
        raise SystemExit(0)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        faces = detector.detect(frame)
        detector.draw(frame, faces)
        cv2.putText(
            frame,
            f"Faces: {len(faces)} | Backend: {detector.active_detector}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2,
        )
        cv2.imshow("CamAI Face Detector", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
