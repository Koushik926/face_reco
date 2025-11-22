"""Recognition logic using cosine similarity.

Provides matching against the pre-built DB. Threshold is adjustable.
"""
from typing import Tuple, Dict
import sys
from pathlib import Path

# Ensure local src imports work when running this file directly
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two 1D numpy vectors."""
    if a is None or b is None:
        return -1.0
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return -1.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


class Recognizer:
    def __init__(self, db: Dict, threshold: float = 0.6):
        """db: mapping person_id -> {name, embedding, imgs}
        threshold: cosine similarity threshold (higher = more strict)
        """
        self.db = db or {}
        self.threshold = threshold

    def match(self, emb: np.ndarray) -> Tuple[str, float]:
        """Match a single embedding to the DB. Returns (label, score)."""
        best_id = None
        best_score = -1.0
        for pid, rec in self.db.items():
            score = cosine_similarity(emb, rec['embedding'])
            if score > best_score:
                best_score = score
                best_id = pid

        if best_score >= self.threshold and best_id is not None:
            return self.db[best_id]['name'], float(best_score)
        return 'Unknown', float(best_score)

    def match_batch(self, embs: np.ndarray):
        """Match a batch of embeddings. Returns list of (label, score)."""
        results = []
        for e in embs:
            results.append(self.match(e))
        return results
