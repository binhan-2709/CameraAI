"""Realtime camera preview pipeline."""
from __future__ import annotations

import queue
import threading
import time
from pathlib import Path

import cv2

try:
    from config import CAMERA_HEIGHT, CAMERA_SOURCE, CAMERA_WIDTH
    from detection.face_detector import FaceDetector
    from recognition.recognizer import FaceRecognizer
except ImportError:  # pragma: no cover
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from config import CAMERA_HEIGHT, CAMERA_SOURCE, CAMERA_WIDTH
    from detection.face_detector import FaceDetector
    from recognition.recognizer import FaceRecognizer


class RealtimePipeline:
    def __init__(self, camera_source=None):
        source = CAMERA_SOURCE if camera_source is None else camera_source
        self.cap = cv2.VideoCapture(source)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

        self.frame_q: queue.Queue = queue.Queue(maxsize=2)
        self.result_q: queue.Queue = queue.Queue(maxsize=2)
        self.running = False
        self.latencies_ms: list[float] = []

        self.detector = FaceDetector()
        self.recognizer = FaceRecognizer()

    def _reader(self) -> None:
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.02)
                continue
            if self.frame_q.full():
                try:
                    self.frame_q.get_nowait()
                except queue.Empty:
                    pass
            self.frame_q.put(frame.copy())

    def _worker(self) -> None:
        while self.running:
            try:
                frame = self.frame_q.get(timeout=0.1)
            except queue.Empty:
                continue

            started = time.perf_counter()
            faces = self.detector.detect(frame)
            results = []
            for face in faces:
                rec = self.recognizer.recognize(face.embedding)
                rec["bbox"] = face.bbox.tolist()
                results.append(rec)

            latency_ms = (time.perf_counter() - started) * 1000
            if self.result_q.full():
                try:
                    self.result_q.get_nowait()
                except queue.Empty:
                    pass
            self.result_q.put((frame, results, latency_ms))

    def run(self) -> None:
        if not self.cap.isOpened():
            print("[!] Cannot open camera")
            return

        self.running = True
        threading.Thread(target=self._reader, daemon=True).start()
        threading.Thread(target=self._worker, daemon=True).start()

        print("Realtime pipeline is running. Press Q to quit.")
        last_frame = None
        last_results = []

        while True:
            try:
                last_frame, last_results, latency = self.result_q.get(timeout=0.03)
                self.latencies_ms.append(latency)
                self.latencies_ms = self.latencies_ms[-30:]
            except queue.Empty:
                pass

            if last_frame is None:
                continue

            display = last_frame.copy()
            for result in last_results:
                box = [int(value) for value in result["bbox"]]
                identity = result["identity"]
                confidence = result["confidence"]
                color = (0, 220, 80) if identity != "unknown" else (0, 80, 220)
                cv2.rectangle(display, (box[0], box[1]), (box[2], box[3]), color, 2)
                cv2.putText(
                    display,
                    f"{identity} {confidence:.0%}",
                    (box[0], max(20, box[1] - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2,
                )

            if self.latencies_ms:
                avg_ms = sum(self.latencies_ms) / len(self.latencies_ms)
                fps = 1000 / avg_ms if avg_ms > 0 else 0
                cv2.putText(
                    display,
                    f"FPS: {fps:.1f} | Latency: {avg_ms:.0f} ms",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 0),
                    2,
                )

            cv2.imshow("CamAI Attendance", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        self.running = False
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    RealtimePipeline().run()
