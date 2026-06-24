"""Realtime camera preview pipeline."""
from __future__ import annotations

import queue
import threading
import time
from collections import deque, Counter
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

import requests

try:
    from config import CAMERA_HEIGHT, CAMERA_SOURCE, CAMERA_WIDTH, PORT
    from detection.face_detector import FaceDetector
    from detection.emotion_detector import EmotionDetector, EMOTION_COLOR_BGR, EMOTION_SHORT
    from recognition.recognizer import FaceRecognizer
except ImportError:  # pragma: no cover
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from config import CAMERA_HEIGHT, CAMERA_SOURCE, CAMERA_WIDTH, PORT
    from detection.face_detector import FaceDetector
    from detection.emotion_detector import EmotionDetector, EMOTION_COLOR_BGR, EMOTION_SHORT
    from recognition.recognizer import FaceRecognizer

try:
    from api.attendance_service import AttendanceService, get_engine
    from sqlalchemy.orm import sessionmaker
except ImportError:
    AttendanceService = None
    get_engine = None
    sessionmaker = None


class RealtimePipeline:
    def __init__(self, camera_source=None, history_size=20, record_attendance=True):
        source = CAMERA_SOURCE if camera_source is None else camera_source
        self.cap = cv2.VideoCapture(source)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

        self.frame_q: queue.Queue = queue.Queue(maxsize=2)
        self.result_q: queue.Queue = queue.Queue(maxsize=2)
        self.running = False
        self.latencies_ms: list[float] = []

        self.detection_history: deque = deque(maxlen=history_size)
        self.displayed_identities: dict[str, datetime] = {}
        self.record_attendance = record_attendance

        self.detector = FaceDetector()
        self.recognizer = FaceRecognizer()
        self.emotion_detector = EmotionDetector()
        print(f"[EmotionDetector] Ready: {self.emotion_detector.is_ready}")
        
        self.api_url = f"http://127.0.0.1:{PORT}/api/attendance/record"
        print(f"[API] Pipeline configured to log events to: {self.api_url}")
        
        # Real-time Tracking & Performance Metrics state
        self.active_trackers: list[dict] = []
        self.avg_det_ms = 0.0
        self.avg_track_ms = 0.0
        self.fps = 0.0
        self._load_existing_history()

    def _load_existing_history(self) -> None:
        """Pre-populate self.detection_history with today's logs from the database."""
        try:
            base_logs_url = self.api_url.replace("/record", "/logs")
            resp = requests.get(f"{base_logs_url}?limit=20", timeout=1.5)
            if resp.status_code == 200:
                logs = resp.json()
                today_str = datetime.now().strftime("%Y-%m-%d")
                for log in reversed(logs):
                    if log.get("date") == today_str:
                        try:
                            dt = datetime.fromisoformat(log["timestamp"])
                        except ValueError:
                            dt = datetime.now()
                        self.detection_history.append({
                            "identity": log["employee_id"],
                            "timestamp": dt,
                            "event_type": log["event_type"]
                        })
                print(f"[Pipeline] Pre-populated {len(self.detection_history)} existing logs from today.")
        except Exception as e:
            print(f"[Pipeline] Could not pre-populate history: {e}")

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
        frame_counter = 0
        h_frame, w_frame = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        if h_frame <= 0 or w_frame <= 0:
            h_frame, w_frame = 720, 1280

        while self.running:
            try:
                frame = self.frame_q.get(timeout=0.1)
            except queue.Empty:
                continue

            started = time.perf_counter()
            results = []
            det_ms = 0.0
            track_ms = 0.0

            # Run full detection every 8 frames, or if no active trackers exist
            run_detection = (frame_counter % 8 == 0) or (not self.active_trackers)
            frame_counter += 1

            if run_detection:
                det_start = time.perf_counter()
                faces = self.detector.detect(frame)
                det_ms = (time.perf_counter() - det_start) * 1000
                self.avg_det_ms = 0.8 * self.avg_det_ms + 0.2 * det_ms

                new_trackers = []
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                for face in faces:
                    rec = self.recognizer.recognize(face.embedding)
                    x1, y1, x2, y2 = [int(v) for v in face.bbox]
                    
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w_frame, x2), min(h_frame, y2)
                    if x2 <= x1 or y2 <= y1:
                        continue

                    face_crop = frame[y1:y2, x1:x2]
                    face_template = frame_gray[y1:y2, x1:x2].copy()

                    # Run emotion detection on the face crop
                    em_result = self.emotion_detector.predict(face_crop)

                    # Simple overlap check to carry over gesture tracking state
                    matching_old = None
                    for old_t in self.active_trackers:
                        ox1, oy1, ox2, oy2 = old_t["bbox"]
                        overlap_x = max(0, min(x2, ox2) - max(x1, ox1))
                        overlap_y = max(0, min(y2, oy2) - max(y1, oy1))
                        if overlap_x * overlap_y > 0.40 * (x2 - x1) * (y2 - y1):
                            matching_old = old_t
                            break

                    if matching_old:
                        tracker = matching_old
                        tracker["bbox"] = [x1, y1, x2, y2]
                        tracker["template"] = face_template
                        tracker["liveness_score"] = face.liveness_score
                        tracker["confidence"] = rec["confidence"]
                        tracker["identity"] = rec["identity"]
                        tracker["emotion"] = em_result["emotion"]
                        tracker["emotion_score"] = em_result["score"]
                        tracker["emotion_short"] = em_result["short"]
                        tracker["emotion_color"] = em_result["color_bgr"]
                        # Temporal smoothing: majority vote over last 5 frames
                        emo_window = tracker.setdefault("emotion_window", [])
                        emo_window.append(em_result["emotion"])
                        # Keep only last 5 predictions
                        if len(emo_window) > 5:
                            emo_window.pop(0)
                        # Pick the emotion that appears most often in window
                        vote = Counter(emo_window).most_common(1)[0][0]
                        tracker["emotion"] = vote
                        tracker["emotion_short"] = EMOTION_SHORT.get(vote, vote)
                        tracker["emotion_color"] = EMOTION_COLOR_BGR.get(vote, (180, 180, 180))
                    else:
                        tracker = {
                            "bbox": [x1, y1, x2, y2],
                            "template": face_template,
                            "identity": rec["identity"],
                            "confidence": rec["confidence"],
                            "liveness_score": face.liveness_score,
                            "emotion": em_result["emotion"],
                            "emotion_score": em_result["score"],
                            "emotion_short": em_result["short"],
                            "emotion_color": em_result["color_bgr"],
                            "emotion_history": {emo: sc * 0.3 for emo, sc in em_result["all_scores"].items()},
                            "motion_history": deque(maxlen=15),
                            "last_left_roi": None,
                            "last_right_roi": None,
                            "gesture": "none",
                            "start_time": datetime.now(),
                            "api_logged": False
                        }
                    new_trackers.append(tracker)
                self.active_trackers = new_trackers
            else:
                # Track in intermediate frames
                track_start = time.perf_counter()
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                valid_trackers = []
                for tracker in self.active_trackers:
                    x1, y1, x2, y2 = tracker["bbox"]
                    tw, th = x2 - x1, y2 - y1

                    # Expand search region slightly
                    x1_s = max(0, x1 - 30)
                    y1_s = max(0, y1 - 30)
                    x2_s = min(w_frame, x2 + 30)
                    y2_s = min(h_frame, y2 + 30)

                    if (x2_s - x1_s) < tw or (y2_s - y1_s) < th:
                        continue

                    search_crop = frame_gray[y1_s:y2_s, x1_s:x2_s]
                    res = cv2.matchTemplate(search_crop, tracker["template"], cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, max_loc = cv2.minMaxLoc(res)

                    if max_val > 0.55:
                        new_x1 = x1_s + max_loc[0]
                        new_y1 = y1_s + max_loc[1]
                        new_x2 = new_x1 + tw
                        new_y2 = new_y1 + th
                        tracker["bbox"] = [new_x1, new_y1, new_x2, new_y2]
                        tracker["template"] = frame_gray[new_y1:new_y2, new_x1:new_x2].copy()
                        valid_trackers.append(tracker)
                self.active_trackers = valid_trackers
                track_ms = (time.perf_counter() - track_start) * 1000
                self.avg_track_ms = 0.8 * self.avg_track_ms + 0.2 * track_ms

            # Analyze hand waving gesture in side margins
            for tracker in self.active_trackers:
                x1, y1, x2, y2 = tracker["bbox"]
                w, h = x2 - x1, y2 - y1

                left_x1 = max(0, x1 - w)
                left_x2 = x1
                right_x1 = x2
                right_x2 = min(w_frame, x2 + w)

                curr_left = frame_gray[y1:y2, left_x1:left_x2]
                curr_right = frame_gray[y1:y2, right_x1:right_x2]

                motion_val = 0.0
                if tracker["last_left_roi"] is not None and curr_left.shape == tracker["last_left_roi"].shape:
                    diff_l = cv2.absdiff(curr_left, tracker["last_left_roi"])
                    _, motion_l = cv2.threshold(diff_l, 15, 255, cv2.THRESH_BINARY)
                    motion_val = max(motion_val, motion_l.sum() / 255.0)
                tracker["last_left_roi"] = curr_left.copy() if curr_left.size > 0 else None

                if tracker["last_right_roi"] is not None and curr_right.shape == tracker["last_right_roi"].shape:
                    diff_r = cv2.absdiff(curr_right, tracker["last_right_roi"])
                    _, motion_r = cv2.threshold(diff_r, 15, 255, cv2.THRESH_BINARY)
                    motion_val = max(motion_val, motion_r.sum() / 255.0)
                tracker["last_right_roi"] = curr_right.copy() if curr_right.size > 0 else None

                tracker["motion_history"].append(motion_val)

                # --- Waving gesture detection (strict swing-based) ---
                # Strategy: require alternating peaks (left->right->left or vice versa)
                # to avoid false positives from background motion or tilted body position.
                motion_list = list(tracker["motion_history"])
                if len(motion_list) >= 12:
                    roi_area = w * h
                    # High threshold: 15% of ROI area must be moving
                    HIGH_THRESH = roi_area * 0.15
                    LOW_THRESH = roi_area * 0.04

                    # Count direction swings: motion_val must exceed HIGH_THRESH at least 3 times
                    # with drops below LOW_THRESH between peaks (real wave pattern)
                    peaks = 0
                    in_peak = False
                    for mv in motion_list:
                        if mv > HIGH_THRESH and not in_peak:
                            peaks += 1
                            in_peak = True
                        elif mv < LOW_THRESH:
                            in_peak = False

                    if peaks >= 3:
                        tracker["gesture"] = "waving"
                    else:
                        # Reset gesture if motion is consistently low
                        avg_m = sum(motion_list) / len(motion_list)
                        if avg_m < LOW_THRESH:
                            tracker["gesture"] = "none"

                # Process API Log
                identity = tracker["identity"]
                if identity != "unknown":
                    is_real = tracker["liveness_score"] >= 0.5
                    elapsed_sec = (datetime.now() - tracker["start_time"]).total_seconds()

                    # Query check-in status from backend once when tracking starts
                    if "is_checked_in_today" not in tracker:
                        tracker["is_checked_in_today"] = False  # default
                        try:
                            base_logs_url = self.api_url.replace("/record", "/logs")
                            resp = requests.get(f"{base_logs_url}?employee_id={identity}&limit=10", timeout=1.0)
                            if resp.status_code == 200:
                                logs = resp.json()
                                today_str = datetime.now().strftime("%Y-%m-%d")
                                for l in logs:
                                    if l.get("date") == today_str and l.get("event_type") == "check_in":
                                        tracker["is_checked_in_today"] = True
                                        break
                        except Exception as err:
                            print(f"[Pipeline] Error checking logs for {identity}: {err}")

                    should_log = False
                    forced_event = None

                    # --- NEW RULE: Check-in with happy smile, Check-out with waving hand ---
                    if not tracker["is_checked_in_today"]:
                        # Rule 1: Smile to Check-In
                        if tracker.get("emotion") == "happiness" and not tracker.get("checkin_logged") and elapsed_sec > 0.6:
                            if identity not in self.displayed_identities:
                                should_log = True
                                forced_event = "check_in"
                                tracker["checkin_logged"] = True
                    else:
                        # Rule 2: Wave to Check-Out (bypasses displayed_identities to allow immediate testing checkout)
                        if tracker["gesture"] == "waving" and not tracker.get("checkout_logged"):
                            should_log = True
                            forced_event = "check_out"
                            tracker["checkout_logged"] = True

                    if should_log:
                        logged_successfully = False
                        actual_event = None

                        if self.record_attendance:
                            try:
                                resp = requests.post(
                                    self.api_url,
                                    json={
                                        "employee_id": identity,
                                        "confidence": float(tracker["confidence"]),
                                        "camera_id": "REALTIME_CAMERA",
                                        "is_real": is_real,
                                        "liveness_score": float(tracker["liveness_score"]),
                                        "event_type": forced_event,
                                        "emotion": tracker.get("emotion", "neutral"),
                                    },
                                    timeout=2.0
                                )
                                if resp.status_code == 200:
                                    res_json = resp.json()
                                    if res_json.get("logged"):
                                        logged_successfully = True
                                        actual_event = res_json.get("event_type")
                                        if actual_event == "check_in":
                                            tracker["is_checked_in_today"] = True
                                        print(f"[API] Logged attendance for {identity} ({actual_event}): status=200")
                                    else:
                                        print(f"[API] Logging skipped: {res_json.get('reason')}")
                                else:
                                    print(f"[API] Server returned status={resp.status_code}")
                            except Exception as e:
                                print(f"[API] Attendance logging failed: {e}")
                        else:
                            logged_successfully = True
                            actual_event = forced_event or "check_in"
                            if actual_event == "check_in":
                                tracker["is_checked_in_today"] = True

                        if logged_successfully:
                            # Register display identity to limit rapid duplicates (cleared after 60s)
                            self.displayed_identities[identity] = datetime.now()
                            # Append a new log to left-hand camera display panel
                            self.detection_history.append({
                                "identity": identity,
                                "timestamp": datetime.now(),
                                "event_type": actual_event
                            })

                results.append({
                    "bbox": tracker["bbox"],
                    "identity": tracker["identity"],
                    "confidence": tracker["confidence"],
                    "liveness_score": tracker["liveness_score"],
                    "gesture": tracker["gesture"],
                    "emotion": tracker.get("emotion", "neutral"),
                    "emotion_short": tracker.get("emotion_short", "Neutral"),
                    "emotion_color": tracker.get("emotion_color", (180, 180, 180)),
                })

            # Clean up old displayed identities after 60 seconds to allow recheck
            now_dt = datetime.now()
            expired = [ident for ident, t in self.displayed_identities.items() if (now_dt - t).total_seconds() > 60.0]
            for ident in expired:
                del self.displayed_identities[ident]

            latency_ms = (time.perf_counter() - started) * 1000
            self.fps = 0.8 * self.fps + 0.2 * (1000.0 / latency_ms if latency_ms > 0 else 30.0)

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

        print("Realtime pipeline is running. Press Q to quit, F for fullscreen.")
        last_frame = None
        last_results = []
        fullscreen = False

        cv2.namedWindow("CamAI Attendance", cv2.WINDOW_NORMAL)

        while True:
            try:
                last_frame, last_results, latency = self.result_q.get(timeout=0.03)
                self.latencies_ms.append(latency)
                self.latencies_ms = self.latencies_ms[-30:]
            except queue.Empty:
                pass

            if last_frame is None:
                continue

            h, w = last_frame.shape[:2]
            panel_width = 200

            output = np.zeros((h, w + panel_width, 3), dtype=np.uint8)
            output[:, :panel_width] = [25, 25, 35]
            output[:, panel_width:] = last_frame

            for result in last_results:
                box = [int(v) for v in result["bbox"]]
                identity = result["identity"]
                confidence = result["confidence"]
                liveness = result.get("liveness_score", 0.0)
                gesture = result.get("gesture", "none")
                
                is_real = liveness >= 0.5
                if identity != "unknown" and is_real:
                    color = (0, 220, 80)  # Green
                elif identity != "unknown" and not is_real:
                    color = (0, 140, 255)  # Orange (Recognized but spoof alert)
                else:
                    color = (0, 80, 220)  # Red/Amber

                adjusted_box = [box[0] + panel_width, box[1], box[2] + panel_width, box[3]]
                cv2.rectangle(output, (adjusted_box[0], adjusted_box[1]), (adjusted_box[2], adjusted_box[3]), color, 2)
                
                if identity == "unknown":
                    label = "unknown"
                else:
                    gesture_suffix = " [CHECK-OUT]" if gesture == "waving" else ""
                    emotion_short = result.get("emotion_short", "Neutral")
                    label = f"{identity} ({confidence:.0%}) | {emotion_short} | Live:{liveness:.0%}{gesture_suffix}"

                # Determine label color based on emotion when known
                if identity != "unknown":
                    em_color = result.get("emotion_color", color)
                    label_color = em_color if is_real else (0, 140, 255)
                else:
                    label_color = color
                
                cv2.putText(
                    output,
                    label,
                    (adjusted_box[0], max(20, adjusted_box[1] - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    label_color,
                    2,
                )

            if self.latencies_ms:
                cv2.putText(
                    output,
                    f"FPS: {self.fps:.1f} | Det: {self.avg_det_ms:.0f}ms | Track: {self.avg_track_ms:.0f}ms",
                    (max(panel_width + 10, w + panel_width - 280), h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (200, 200, 200),
                    1,
                )

            self._draw_detection_history_panel(output, panel_width, h)

            cv2.imshow("CamAI Attendance", output)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("f"):
                fullscreen = not fullscreen
                if fullscreen:
                    cv2.setWindowProperty("CamAI Attendance", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                else:
                    cv2.setWindowProperty("CamAI Attendance", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

        self.running = False
        self.cap.release()
        cv2.destroyAllWindows()
        if self.db:
            self.db.close()

    def _draw_detection_history_panel(self, frame: np.ndarray, panel_width: int, h: int) -> None:
        """Draw detection history on left panel (split into CHECK-IN and CHECK-OUT sections)."""
        date_str = datetime.now().strftime("%Y/%m/%d")
        cv2.putText(
            frame,
            date_str,
            (15, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (200, 200, 220),
            1,
            cv2.LINE_AA
        )

        cv2.line(frame, (10, 34), (panel_width - 10, 34), (80, 80, 100), 1)

        # Split history list
        hist = list(self.detection_history)
        check_ins = [d for d in hist if d.get("event_type") == "check_in"]
        check_outs = [d for d in hist if d.get("event_type") == "check_out"]

        # --- 1. CHECK-IN SECTION ---
        cv2.putText(
            frame,
            "CHECK-IN",
            (15, 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 220, 80),   # Green
            1,
            cv2.LINE_AA
        )
        cv2.line(frame, (10, 58), (panel_width - 10, 58), (50, 50, 70), 1)

        y_pos = 70
        row_height = 36
        for detection in check_ins[:4]:
            if y_pos + row_height > h - 15:
                break
            identity = detection["identity"]
            timestamp = detection["timestamp"].strftime("%H:%M:%S")

            # Small green circle
            cv2.circle(frame, (18, y_pos + 10), 3, (0, 220, 80), -1)
            # Employee ID
            cv2.putText(
                frame,
                identity,
                (28, y_pos + 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )
            # Time
            cv2.putText(
                frame,
                timestamp,
                (28, y_pos + 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.32,
                (160, 160, 160),
                1,
                cv2.LINE_AA
            )
            cv2.line(frame, (12, y_pos + 30), (panel_width - 12, y_pos + 30), (40, 40, 50), 1)
            y_pos += row_height

        # --- 2. CHECK-OUT SECTION ---
        y_checkout_section = 230
        cv2.line(frame, (10, y_checkout_section - 15), (panel_width - 10, y_checkout_section - 15), (80, 80, 100), 1)
        
        cv2.putText(
            frame,
            "CHECK-OUT",
            (15, y_checkout_section + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 140, 255),  # Orange/Amber
            1,
            cv2.LINE_AA
        )
        cv2.line(frame, (10, y_checkout_section + 11), (panel_width - 10, y_checkout_section + 11), (50, 50, 70), 1)

        y_pos = y_checkout_section + 24
        for detection in check_outs[:4]:
            if y_pos + row_height > h - 15:
                break
            identity = detection["identity"]
            timestamp = detection["timestamp"].strftime("%H:%M:%S")

            # Small orange circle
            cv2.circle(frame, (18, y_pos + 10), 3, (0, 140, 255), -1)
            # Employee ID
            cv2.putText(
                frame,
                identity,
                (28, y_pos + 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )
            # Time
            cv2.putText(
                frame,
                timestamp,
                (28, y_pos + 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.32,
                (160, 160, 160),
                1,
                cv2.LINE_AA
            )
            cv2.line(frame, (12, y_pos + 30), (panel_width - 12, y_pos + 30), (40, 40, 50), 1)
            y_pos += row_height


if __name__ == "__main__":
    RealtimePipeline().run()
