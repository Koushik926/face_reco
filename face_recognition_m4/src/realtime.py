"""Real-time face recognition demo optimized for Apple Silicon (MPS).

Run as:
    python src/realtime.py --db ../face_db.pkl
"""
import argparse
import time
import os
import csv
from pathlib import Path
from collections import deque
import sys

import cv2
import numpy as np
import torch
import subprocess

# Ensure local src imports work when running this file directly
sys.path.insert(0, str(Path(__file__).resolve().parent))
from detector import FaceDetector
from embedder import Embedder
from recognizer import Recognizer
from tracker import SimpleTracker, iou
from utils import draw_box_label, resize_keep_aspect, box_scale_back


def get_device(prefer: str = None):
    if prefer == 'mps' and torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def load_db(path: str):
    import pickle
    with open(path, 'rb') as f:
        return pickle.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--db', default=os.path.join(os.path.dirname(__file__), '..', 'face_db.pkl'))
    parser.add_argument('--cam', type=int, default=0)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--device', choices=['mps', 'cpu'], default='mps')
    parser.add_argument('--detection-device', choices=['mps', 'cpu'], default=None, help='Device for running MTCNN detection (if not set, uses --device)')
    parser.add_argument('--threshold', type=float, default=0.6)
    parser.add_argument('--detect-every', type=int, default=3, help='Run MTCNN every N frames')
    parser.add_argument('--embedder', choices=['resnet', 'mobileface'], default='resnet', help='Embedding model to use')
    parser.add_argument('--pad-mult', type=int, default=64, help='Padding multiple used by detector when running on MPS (higher may avoid MPS adaptive pool errors)')
    parser.add_argument('--smoothing-alpha', type=float, default=0.6, help='Tracker smoothing alpha (0..1) where higher=more responsive')
    parser.add_argument('--log-csv', default=os.path.join(os.path.dirname(__file__), '..', 'recognitions.csv'))
    parser.add_argument('--greet', action='store_true', help='Enable TTS greeting for recognized people (macOS say)')
    parser.add_argument('--greet-interval', type=float, default=6.0, help='Minimum seconds between greetings for the same person')
    parser.add_argument('--greet-threshold', type=float, default=0.6, help='Minimum recognition score to trigger greeting')
    args = parser.parse_args()

    device = get_device(args.device)
    # detection device can be forced to cpu to avoid MPS adaptive-pool bugs
    det_device = get_device(args.detection_device) if args.detection_device else device
    print(f"Using device: {device}")

    db = load_db(args.db) if os.path.exists(args.db) else {}
    recognizer = Recognizer(db, threshold=args.threshold)
    detector = FaceDetector(device=device, detection_device=det_device, pad_mult=args.pad_mult)
    embedder = Embedder(model_name=args.embedder, device=device)
    tracker = SimpleTracker(smoothing_alpha=args.smoothing_alpha)

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print('Cannot open camera')
        return

    frame_count = 0
    fps_deque = deque(maxlen=30)

    # Greeting cooldowns: map person name -> last greeted timestamp
    last_greet_time = {}

    # CSV logging
    csv_file = open(args.log_csv, 'a', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['timestamp', 'track_id', 'name', 'score'])

    try:
        while True:
            started = time.time()
            ret, frame = cap.read()
            if not ret:
                break

            small, scale = resize_keep_aspect(frame, width=args.width)

            labels = []
            boxes_out = []
            scores = []

            if frame_count % args.detect_every == 0:
                boxes, probs, faces = detector.detect(small)
                if boxes:
                    # If detector returned aligned face tensors, use them (fast path)
                    if faces is not None and len(faces) > 0:
                        embs = embedder.embed(faces)
                        matches = recognizer.match_batch(embs)
                    else:
                        # Fallback: detector provided boxes but no aligned faces (e.g., CPU detect after MPS issue).
                        # Crop and align on the smaller frame and build a face batch for embedding.
                        face_tensors = []
                        IMG_SIZE = 160
                        for b in boxes:
                            x1, y1, x2, y2 = b
                            # clamp coords
                            x1 = max(0, int(x1))
                            y1 = max(0, int(y1))
                            x2 = max(0, int(x2))
                            y2 = max(0, int(y2))
                            if x2 <= x1 or y2 <= y1:
                                continue
                            crop = small[y1:y2, x1:x2]
                            if crop.size == 0:
                                continue
                            # convert BGR->RGB and resize to model input
                            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                            resized = cv2.resize(crop_rgb, (IMG_SIZE, IMG_SIZE))
                            arr = resized.astype('float32') / 255.0
                            t = torch.from_numpy(arr).permute(2, 0, 1)
                            t = (t - 0.5) / 0.5
                            face_tensors.append(t)

                        if face_tensors:
                            face_batch = torch.stack(face_tensors, dim=0)
                            embs = embedder.embed(face_batch)
                            matches = recognizer.match_batch(embs)
                        else:
                            matches = []

                    # boxes correspond to small scale, scale back to original frame
                    for b, (name, score) in zip(boxes, matches):
                        b_orig = box_scale_back(b, scale)
                        boxes_out.append(b_orig)
                        labels.append(name)
                        scores.append(score)
            else:
                # no detection; use previously tracked boxes
                boxes = []

            # tracker: update with boxes_out (in original frame coords)
            tracked = tracker.update(boxes_out, labels=labels, scores=scores)

            # Draw tracked boxes
            for tid, box in tracked.items():
                # prefer track's last-known label (set in tracker.update); fallback to IoU matching
                tr = tracker.tracks.get(tid, {})
                label = tr.get('label', 'Unknown')
                score = tr.get('score', None)
                if label == 'Unknown' and boxes_out:
                    # use IoU to find best matching detection and assign label
                    best_iou = 0.0
                    best_idx = -1
                    for idx, (b, lab, sc) in enumerate(zip(boxes_out, labels, scores)):
                        v = iou(box, b)
                        if v > best_iou:
                            best_iou = v
                            best_idx = idx
                    if best_idx >= 0 and best_iou >= 0.2:
                        label = labels[best_idx]
                        score = scores[best_idx]
                        # store back into tracker
                        tr['label'] = label
                        tr['score'] = score
                draw_box_label(frame, box, f"{label}#{tid}", score)

                # TTS greeting (macOS `say`) — speak once per person per interval
                if args.greet and label != 'Unknown' and score is not None:
                    try:
                        sc_val = float(score)
                    except Exception:
                        sc_val = 0.0
                    if sc_val >= args.greet_threshold:
                        now = time.time()
                        last = last_greet_time.get(label, 0.0)
                        if now - last >= args.greet_interval:
                            # non-blocking speak
                            try:
                                text = f"Hello {label}. Welcome."
                                # debug print to confirm greeting trigger
                                print(f"Greeting triggered for {label} (score={sc_val:.3f}) -> say: {text}")
                                try:
                                    # Try the absolute say path first (more deterministic on macOS)
                                    SAY_BIN = '/usr/bin/say'
                                    p = subprocess.Popen([SAY_BIN, text], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                                    print(f"Launched say pid={p.pid} via {SAY_BIN}")
                                    last_greet_time[label] = now
                                except Exception as e:
                                    print(f"/usr/bin/say Popen failed: {e}")
                                    # Try AppleScript as a fallback
                                    try:
                                        subprocess.run(['osascript', '-e', f'say "{text}"'], check=True)
                                        print("Launched say via osascript")
                                        last_greet_time[label] = now
                                    except Exception as e2:
                                        print(f"osascript failed: {e2}, falling back to os.system")
                                        try:
                                            os.system(f"say '{text}'")
                                            last_greet_time[label] = now
                                        except Exception as e3:
                                            print(f"os.system say failed: {e3}")
                            except Exception:
                                # ignore TTS failures
                                pass

            # FPS
            elapsed = time.time() - started
            fps = 1.0 / elapsed if elapsed > 0 else 0.0
            fps_deque.append(fps)
            fps_avg = sum(fps_deque) / len(fps_deque)
            cv2.putText(frame, f"FPS: {fps_avg:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imshow('Face Recognition M4', frame)

            # Log recognitions using track's stored label if available
            for tid, box in tracked.items():
                tr = tracker.tracks.get(tid, {})
                label = tr.get('label', 'Unknown')
                sc = tr.get('score', None)
                score = f"{sc:.3f}" if sc is not None else ''
                csv_writer.writerow([time.time(), tid, label, score])

            frame_count += 1
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    finally:
        csv_file.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
