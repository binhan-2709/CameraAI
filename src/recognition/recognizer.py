"""Face recognition with FAISS, plus a NumPy fallback for tests/dev."""
from __future__ import annotations

import pickle
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover - depends on local environment
    faiss = None

try:
    from config import FACE_DB_PATH, FACE_THRESHOLD
except ImportError:  # pragma: no cover
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from config import FACE_DB_PATH, FACE_THRESHOLD


EMBEDDING_DIM = 512


class FaceRecognizer:
    def __init__(self, db_path: str = FACE_DB_PATH, threshold: float = FACE_THRESHOLD):
        self.db_path = str(db_path)
        self.threshold = threshold
        self.embeddings: list[np.ndarray] = []
        self.labels: list[str] = []
        self.index: Any = None

        if Path(self.db_path).exists():
            self.load()
        else:
            self._rebuild_index()

    def recognize(self, embedding: np.ndarray | None) -> dict:
        if embedding is None:
            return {"identity": "unknown", "confidence": 0.0, "status": "no_embedding"}
        if not self.labels:
            return {"identity": "unknown", "confidence": 0.0, "status": "no_database"}

        emb = self._normalize(embedding)
        if emb.shape[0] != EMBEDDING_DIM:
            return {
                "identity": "unknown",
                "confidence": 0.0,
                "status": f"bad_embedding_dim:{emb.shape[0]}",
            }

        sims, idxs = self._search(emb, k=min(5, len(self.labels)))
        best_sim = float(sims[0])
        if best_sim < self.threshold:
            return {"identity": "unknown", "confidence": best_sim, "status": "unknown"}

        top_labels = [self.labels[i] for i in idxs if i >= 0]
        identity = Counter(top_labels).most_common(1)[0][0]
        return {"identity": identity, "confidence": best_sim, "status": "recognized"}

    def add_person(self, employee_id: str, embeddings: list[np.ndarray]) -> int:
        new_embs = [self._normalize(e) for e in embeddings if e is not None]
        new_embs = [e for e in new_embs if e.shape[0] == EMBEDDING_DIM]
        if not new_embs:
            return 0

        self.embeddings.extend(new_embs)
        self.labels.extend([employee_id] * len(new_embs))
        self._rebuild_index()
        self.save()
        return len(new_embs)

    def remove_person(self, employee_id: str) -> int:
        before = len(self.labels)
        pairs = [(e, l) for e, l in zip(self.embeddings, self.labels) if l != employee_id]
        if pairs:
            self.embeddings, self.labels = map(list, zip(*pairs))
        else:
            self.embeddings, self.labels = [], []
        self._rebuild_index()
        self.save()
        return before - len(self.labels)

    def save(self) -> None:
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "version": 2,
            "embeddings": self.embeddings,
            "labels": self.labels,
            "backend": "faiss" if faiss is not None else "numpy",
            "index": None,
        }
        if faiss is not None and self.index is not None:
            payload["index"] = faiss.serialize_index(self.index)
        with open(self.db_path, "wb") as file:
            pickle.dump(payload, file)

    def load(self) -> None:
        with open(self.db_path, "rb") as file:
            data = pickle.load(file)

        self.embeddings = [self._normalize(np.asarray(e)) for e in data.get("embeddings", [])]
        self.labels = list(data.get("labels", []))
        if len(self.embeddings) != len(self.labels):
            raise ValueError("Face database is corrupted: embeddings/labels length mismatch")

        if faiss is not None and data.get("index") is not None:
            try:
                self.index = faiss.deserialize_index(data["index"])
                return
            except Exception:
                pass
        self._rebuild_index()

    def _search(self, emb: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        if faiss is not None and self.index is not None:
            sims, idxs = self.index.search(emb.reshape(1, -1).astype("float32"), k)
            return sims[0], idxs[0]

        mat = np.asarray(self.embeddings, dtype="float32")
        sims = mat @ emb.astype("float32")
        order = np.argsort(-sims)[:k]
        return sims[order], order

    def _rebuild_index(self) -> None:
        if faiss is None:
            self.index = None
            return
        self.index = faiss.IndexFlatIP(EMBEDDING_DIM)
        if self.embeddings:
            mat = np.asarray(self.embeddings, dtype="float32")
            self.index.add(mat)

    @staticmethod
    def _normalize(emb: np.ndarray) -> np.ndarray:
        arr = np.asarray(emb, dtype="float32").reshape(-1)
        norm = np.linalg.norm(arr)
        return (arr / norm).astype("float32") if norm > 0 else arr

    @property
    def total(self) -> int:
        return len(self.labels)

    @property
    def people(self) -> list[str]:
        return sorted(set(self.labels))
