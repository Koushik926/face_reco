"""Simple centroid/IOU tracker to persist identities between detection frames.

This tracker keeps short-term tracks for objects (faces). It matches new boxes
to existing tracks using IOU and assigns stable track IDs.
"""
from typing import List, Tuple, Dict
import sys
from pathlib import Path

# Ensure local src imports work when running this file directly
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np


def iou(boxA, boxB):
    # boxes: [x1,y1,x2,y2]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    boxAArea = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    boxBArea = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])
    if boxAArea + boxBArea - interArea == 0:
        return 0.0
    return interArea / float(boxAArea + boxBArea - interArea)


class SimpleTracker:
    """A very small tracker.

    Keeps tracks as dict id -> {box, missed} and assigns integer IDs.
    """

    def __init__(self, iou_threshold: float = 0.3, max_missed: int = 5, smoothing_alpha: float = 0.6, stable_k: int = 3, stable_min_score: float = 0.65):
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self.next_id = 1
        # tracks: id -> {'box': box, 'smoothed': box, 'missed': 0, 'hits': 0, 'label': str, 'score': float,
        #                 'history': deque([(label, score), ...]), 'stable_label': str, 'stable_score': float}
        self.tracks = {}
        # smoothing factor (alpha): new_smoothed = alpha*new + (1-alpha)*old
        self.smoothing_alpha = float(smoothing_alpha)
        # stable label parameters
        from collections import deque
        self._deque_cls = deque
        self.stable_k = int(stable_k)
        self.stable_min_score = float(stable_min_score)

    def update(self, boxes: List[List[int]], labels: List[str] = None, scores: List[float] = None) -> Dict[int, List[int]]:
        """Update tracks with new detection boxes.

        Returns mapping track_id -> box
        """
        updated = {}
        used = set()

        # Match existing tracks to new boxes by IOU
        for tid, t in list(self.tracks.items()):
            best_iou = 0.0
            best_j = -1
            for j, b in enumerate(boxes):
                if j in used:
                    continue
                v = iou(t['box'], b)
                if v > best_iou:
                    best_iou = v
                    best_j = j

            if best_j >= 0 and best_iou >= self.iou_threshold:
                # update track
                new_box = boxes[best_j]
                old_box = self.tracks[tid]['box']
                # store raw box and update smoothed box
                self.tracks[tid]['box'] = new_box
                self.tracks[tid]['missed'] = 0
                # increment consecutive hits for liveness
                self.tracks[tid]['hits'] = int(self.tracks[tid].get('hits', 0)) + 1
                # ultra-stable: use average of old and new box (no EMA)
                sm = [int(round((nb + ob) / 2.0)) for nb, ob in zip(new_box, old_box)]
                self.tracks[tid]['smoothed'] = sm
                # update recognition history if provided
                if labels is not None and best_j < len(labels):
                    lab = labels[best_j]
                    sc = None
                    if scores is not None and best_j < len(scores):
                        try:
                            sc = float(scores[best_j])
                        except Exception:
                            sc = None
                    hist = self.tracks[tid].get('history')
                    if hist is not None:
                        hist.append((lab, sc))
                        # recompute stable label by majority vote over last stable_k frames with min score threshold
                        if len(hist) >= self.stable_k:
                            # count occurrences only for entries meeting score threshold (if score available)
                            counts = {}
                            avg_scores = {}
                            n_scores = {}
                            for (lb, scv) in list(hist)[-self.stable_k:]:
                                if scv is None or scv >= self.stable_min_score:
                                    counts[lb] = counts.get(lb, 0) + 1
                                    if scv is not None:
                                        avg_scores[lb] = avg_scores.get(lb, 0.0) + scv
                                        n_scores[lb] = n_scores.get(lb, 0) + 1
                            if counts:
                                stable = max(counts.items(), key=lambda kv: kv[1])[0]
                                # compute avg score for stable label if available
                                if stable in avg_scores and stable in n_scores and n_scores[stable] > 0:
                                    self.tracks[tid]['stable_score'] = avg_scores[stable] / n_scores[stable]
                                else:
                                    self.tracks[tid]['stable_score'] = None
                                self.tracks[tid]['stable_label'] = stable
                updated[tid] = sm
                used.add(best_j)
            else:
                # no match
                self.tracks[tid]['missed'] += 1
                # decay hits on miss
                self.tracks[tid]['hits'] = max(0, int(self.tracks[tid].get('hits', 0)) - 1)
                # Do not delete tracks; keep last known box for maximum persistence

        # Create tracks for unmatched boxes
        for j, b in enumerate(boxes):
            if j in used:
                continue
            tid = self.next_id
            self.next_id += 1
            # initialize smoothed to the raw box
            entry = {'box': b, 'smoothed': b, 'missed': 0, 'hits': 1}
            if labels is not None and j < len(labels):
                entry['label'] = labels[j]
            else:
                entry['label'] = 'Unknown'
            if scores is not None and j < len(scores):
                entry['score'] = float(scores[j])
            else:
                entry['score'] = None
            # init history deque
            entry['history'] = self._deque_cls(maxlen=max(5, self.stable_k))
            entry['stable_label'] = entry['label']
            entry['stable_score'] = entry['score']
            if entry['label'] is not None:
                entry['history'].append((entry['label'], entry['score']))
            self.tracks[tid] = entry
            updated[tid] = b

        return updated
