"""Utilities for image conversion, drawing and simple metrics."""
from typing import Tuple, List
import sys
from pathlib import Path

# Ensure local src imports work when running this file directly
sys.path.insert(0, str(Path(__file__).resolve().parent))
import cv2
import numpy as np


def draw_box_label(frame: np.ndarray, box: Tuple[int, int, int, int], label: str, score: float = None, color=(0, 255, 0)) -> None:
    x1, y1, x2, y2 = box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    text = label
    if score is not None:
        text = f"{label} {score:.2f}"
    # draw background
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.rectangle(frame, (x1, y1 - 20), (x1 + w, y1), color, -1)
    cv2.putText(frame, text, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)


def resize_keep_aspect(frame: np.ndarray, width: int = 640) -> Tuple[np.ndarray, float]:
    h, w = frame.shape[:2]
    if w == width:
        return frame, 1.0
    scale = width / float(w)
    new_h = int(h * scale)
    resized = cv2.resize(frame, (width, new_h))
    return resized, scale


def box_scale_back(box: List[int], scale: float) -> List[int]:
    return [int(round(b / scale)) for b in box]
