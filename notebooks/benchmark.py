"""
Đo benchmark hệ thống: FPS, latency, accuracy, so sánh model
Chạy: python notebooks/benchmark.py
"""
import sys, os, time, numpy as np, cv2
sys.path.insert(0, "src")


def measure_fps(detector, recognizer,
                source=0, n_frames=100):
    print(f"
[FPS Test] {n_frames} frames...")
    cap = cv2.VideoCapture(source)
    latencies = []

    for i in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            break
        t0    = time.perf_counter()
        faces = detector.detect(frame)
        if faces:
            recognizer.recognize(faces[0].embedding)
        latencies.append((time.perf_counter() - t0) * 1000)

    cap.release()
    if not latencies:
        print("  [!] Không đọc được frame")
        return {}

    return {
        "frames_tested": len(latencies),
        "avg_ms":   round(np.mean(latencies), 2),
        "p50_ms":   round(np.percentile(latencies, 50), 2),
        "p95_ms":   round(np.percentile(latencies, 95), 2),
        "p99_ms":   round(np.percentile(latencies, 99), 2),
        "min_ms":   round(np.min(latencies), 2),
        "max_ms":   round(np.max(latencies), 2),
        "avg_fps":  round(1000 / np.mean(latencies), 1),
    }


def measure_accuracy(detector, recognizer, test_dir="data/test"):
    from pathlib import Path
    print(f"
[Accuracy Test] dir: {test_dir}")
    if not os.path.isdir(test_dir):
        print(f"  [!] Không tìm thấy {test_dir}")
        return {}

    tp = fp = fn = 0
    for person_dir in sorted(Path(test_dir).iterdir()):
        if not person_dir.is_dir():
            continue
        true_id = person_dir.name
        imgs    = list(person_dir.glob("*.jpg")) +                   list(person_dir.glob("*.png"))
        for img_path in imgs:
            img   = cv2.imread(str(img_path))
            if img is None:
                continue
            faces = detector.detect(img)
            if not faces:
                fn += 1
                continue
            r = recognizer.recognize(faces[0].embedding)
            if r["identity"] == true_id:
                tp += 1
            elif r["identity"] == "unknown":
                fn += 1
            else:
                fp += 1

    total = tp + fp + fn
    if total == 0:
        return {"error": "Không có ảnh test"}

    return {
        "total_tested": total,
        "TP": tp, "FP": fp, "FN": fn,
        "accuracy": round(tp / total, 4),
        "FAR":      round(fp / max(fp + tp, 1), 4),
        "FRR":      round(fn / max(fn + tp, 1), 4),
    }


def compare_models():
    import onnxruntime as ort
    print("
[Model Comparison]")
    models = {
        "Baseline (FP32)":  "models/face_model.onnx",
        "Quantized (INT8)": "models/face_model_int8.onnx",
    }
    dummy   = np.random.randn(1, 3, 112, 112).astype("float32")
    results = {}
    for name, path in models.items():
        if not os.path.exists(path):
            results[name] = {"status": "file not found"}
            continue
        sess = ort.InferenceSession(path)
        for _ in range(20):
            sess.run(None, {"input": dummy})
        times = []
        for _ in range(100):
            t = time.perf_counter()
            sess.run(None, {"input": dummy})
            times.append((time.perf_counter() - t) * 1000)
        results[name] = {
            "size_mb":  round(os.path.getsize(path) / 1e6, 1),
            "avg_ms":   round(np.mean(times), 2),
            "fps_est":  round(1000 / np.mean(times), 1),
        }
    return results


def print_table(title: str, data: dict):
    print(f"
{'─'*50}")
    print(f"  {title}")
    print(f"{'─'*50}")
    for k, v in data.items():
        print(f"  {k:<20} {v}")


if __name__ == "__main__":
    from detection.face_detector import FaceDetector
    from recognition.recognizer  import FaceRecognizer

    det = FaceDetector()
    rec = FaceRecognizer()

    fps_res = measure_fps(det, rec, n_frames=50)
    if fps_res:
        print_table("FPS & LATENCY", fps_res)

    acc_res = measure_accuracy(det, rec)
    if acc_res:
        print_table("ACCURACY", acc_res)

    cmp_res = compare_models()
    for name, res in cmp_res.items():
        print_table(f"MODEL: {name}", res)

    print(f"
{'═'*50}")
    print("  Benchmark hoàn thành!")
    print("  Copy kết quả vào bảng trong báo cáo đồ án.")
    print(f"{'═'*50}
")
