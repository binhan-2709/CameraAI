"""Lightweight facial expression recognition using ONNX FER+ model via cv2.dnn.

The model is automatically downloaded on first use to models/emotion_ferplus.onnx.
Inference takes ~1-3ms on CPU using OpenCV DNN module (no extra dependencies).

Emotion classes (FER+ based):
    neutral, happiness, surprise, sadness, anger, disgust, contempt, fear
"""
from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EMOTION_LABELS: list[str] = [
    "neutral", "happiness", "surprise", "sadness",
    "anger", "disgust", "contempt", "fear",
]

EMOTION_VI: dict[str, str] = {
    "neutral":   "Binh thuong",
    "happiness": "Vui ve",
    "surprise":  "Ngac nhien",
    "sadness":   "Buon ba",
    "anger":     "Tuc gian",
    "disgust":   "Kho chiu",
    "contempt":  "Kinh thuong",
    "fear":      "So hai",
}

EMOTION_SHORT: dict[str, str] = {
    "neutral":   "Neutral",
    "happiness": "Happy",
    "surprise":  "Surprised",
    "sadness":   "Sad",
    "anger":     "Angry",
    "disgust":   "Disgust",
    "contempt":  "Contempt",
    "fear":      "Fear",
}

EMOTION_EMOJI: dict[str, str] = {
    "neutral":   ":)",
    "happiness": ":D",
    "surprise":  ":O",
    "sadness":   ":(",
    "anger":     ">:(",
    "disgust":   ":/",
    "contempt":  ":|",
    "fear":      "D:",
}

EMOTION_COLOR_BGR: dict[str, tuple] = {
    "neutral":   (180, 180, 180),
    "happiness": (0, 220, 80),
    "surprise":  (0, 200, 255),
    "sadness":   (200, 100, 50),
    "anger":     (30, 30, 220),
    "disgust":   (30, 160, 90),
    "contempt":  (130, 50, 180),
    "fear":      (60, 60, 200),
}

EMOTION_CSS: dict[str, dict] = {
    "neutral":   {"bg": "#f1f5f9", "color": "#475569"},
    "happiness": {"bg": "#dcfce7", "color": "#166534"},
    "surprise":  {"bg": "#e0f2fe", "color": "#0369a1"},
    "sadness":   {"bg": "#eff6ff", "color": "#1d4ed8"},
    "anger":     {"bg": "#fee2e2", "color": "#991b1b"},
    "disgust":   {"bg": "#f0fdf4", "color": "#15803d"},
    "contempt":  {"bg": "#faf5ff", "color": "#7e22ce"},
    "fear":      {"bg": "#fef3c7", "color": "#92400e"},
}

_MODEL_URLS: list[str] = [
    "https://storage.googleapis.com/ailia-models/emotion_ferplus/emotion-ferplus.onnx",
    "https://github.com/onnx/models/raw/main/validated/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx",
]
_MODEL_FILENAME = "emotion_ferplus.onnx"


class EmotionDetector:
    """Predict facial emotion from a BGR face crop."""

    def __init__(self, model_dir=None) -> None:
        base_dir = Path(__file__).resolve().parents[2]
        self.model_dir = Path(model_dir) if model_dir else base_dir / "models"
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.model_path = self.model_dir / _MODEL_FILENAME
        self.net = None
        self._load_model()

    def predict(self, face_bgr: np.ndarray) -> dict:
        """Predict emotion. Returns dict with emotion, score, vi, emoji, color_bgr, css."""
        if self.net is None or face_bgr is None or face_bgr.size == 0:
            return self._make_result("neutral", 0.0, {})
        try:
            gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
            blob = cv2.dnn.blobFromImage(gray, scalefactor=1.0, size=(64, 64), swapRB=False)
            self.net.setInput(blob)
            raw = self.net.forward()
            scores = raw[0].astype(np.float64)

            # --- Temperature scaling (T=1.8) ---
            # Dividing logits by T > 1 flattens the distribution so that
            # minority classes (fear, disgust, surprise) can compete fairly.
            T = 1.8
            scores_t = scores / T
            exp_s = np.exp(scores_t - scores_t.max())
            softmax = exp_s / exp_s.sum()

            # --- Prior correction (inverse class frequency on FER-2013) ---
            # FER-2013 class frequencies (approximate): neutral=0.35, happiness=0.20,
            # surprise=0.12, sadness=0.12, anger=0.08, disgust=0.03, contempt=0.05, fear=0.05
            # We boost rare classes so the model isn't always biased toward neutral/happy.
            PRIOR_WEIGHTS = np.array([
                0.35,  # neutral
                0.20,  # happiness
                0.12,  # surprise
                0.12,  # sadness
                0.08,  # anger
                0.03,  # disgust
                0.05,  # contempt
                0.05,  # fear
            ], dtype=np.float64)
            # Multiply by inverse prior, then re-normalize
            corrected = softmax / PRIOR_WEIGHTS
            corrected = corrected / corrected.sum()

            idx = int(np.argmax(corrected))
            dominant = EMOTION_LABELS[idx]
            confidence = float(corrected[idx])
            all_scores = {EMOTION_LABELS[i]: float(corrected[i]) for i in range(len(EMOTION_LABELS))}
            return self._make_result(dominant, confidence, all_scores)
        except Exception as exc:
            print(f"[EmotionDetector] predict() failed: {exc}")
            return self._make_result("neutral", 0.0, {})

    @property
    def is_ready(self) -> bool:
        return self.net is not None

    def _make_result(self, emotion: str, score: float, all_scores: dict) -> dict:
        return {
            "emotion":    emotion,
            "score":      score,
            "all_scores": all_scores,
            "vi":         EMOTION_VI.get(emotion, emotion),
            "emoji":      EMOTION_EMOJI.get(emotion, ":)"),
            "short":      EMOTION_SHORT.get(emotion, emotion),
            "color_bgr":  EMOTION_COLOR_BGR.get(emotion, (180, 180, 180)),
            "css":        EMOTION_CSS.get(emotion, {"bg": "#f1f5f9", "color": "#475569"}),
        }

    def _load_model(self) -> None:
        if not self.model_path.exists():
            self._download_model()
        if self.model_path.exists():
            try:
                self.net = cv2.dnn.readNetFromONNX(str(self.model_path))
                print(f"[EmotionDetector] Loaded ({self.model_path.stat().st_size // 1024} KB)")
            except Exception as exc:
                print(f"[EmotionDetector] Load failed: {exc}")
                self.net = None
        else:
            print("[EmotionDetector] Model unavailable - emotion detection disabled.")

    def _download_model(self) -> None:
        for url in _MODEL_URLS:
            try:
                print(f"[EmotionDetector] Downloading model...")
                urllib.request.urlretrieve(url, str(self.model_path))
                print(f"[EmotionDetector] Download OK ({self.model_path.stat().st_size // 1024} KB)")
                return
            except Exception as exc:
                print(f"[EmotionDetector] Download failed ({url}): {exc}")
                if self.model_path.exists():
                    self.model_path.unlink(missing_ok=True)
        print("[EmotionDetector] All downloads failed. Place emotion_ferplus.onnx in models/ manually.")
