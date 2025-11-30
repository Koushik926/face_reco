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
from liveness import LivenessDetector


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
    parser.add_argument('--smoothing-alpha', type=float, default=0.0, help='Tracker smoothing alpha (0..1) where lower=more stable')
    parser.add_argument('--max-missed', type=int, default=10000, help='Max frames to keep a track without detection (higher=less blinking)')
    parser.add_argument('--log-csv', default=os.path.join(os.path.dirname(__file__), '..', 'recognitions.csv'))
    parser.add_argument('--greet', action='store_true', help='Enable TTS greeting for recognized people (macOS say)')
    parser.add_argument('--greet-unknown', action='store_true', help='Also greet unknown people with a generic message')
    parser.add_argument('--greet-interval', type=float, default=6.0, help='Minimum seconds between greetings for the same person')
    parser.add_argument('--greet-threshold', type=float, default=0.6, help='Minimum recognition score to trigger greeting')
    parser.add_argument('--min-live-frames', type=int, default=5, help='Minimum consecutive hits before considering a face live (anti-spoof heuristic)')
    parser.add_argument('--stable-k', type=int, default=3, help='Frames required to stabilize identity (majority vote)')
    parser.add_argument('--stable-min-score', type=float, default=0.65, help='Minimum score to count toward identity stabilization')
    parser.add_argument('--require-liveness', action='store_true', help='Enable landmark-based liveness detection (blink/motion) to prevent 2D photo spoofing')
    parser.add_argument('--liveness-motion-threshold', type=float, default=5.0, help='Minimum motion variance for liveness (higher=stricter)')
    parser.add_argument('--liveness-min-motion-frames', type=int, default=3, help='Minimum frames with motion required for liveness')
    parser.add_argument('--liveness-texture-threshold', type=float, default=10.0, help='Minimum texture variance for liveness (screens/photos have lower values)')
    parser.add_argument('--debug-liveness', action='store_true', help='Print liveness detection values for debugging')
    args = parser.parse_args()

    device = get_device(args.device)
    # detection device can be forced to cpu to avoid MPS adaptive-pool bugs
    det_device = get_device(args.detection_device) if args.detection_device else device
    print(f"Using device: {device}")

    db = load_db(args.db) if os.path.exists(args.db) else {}
    recognizer = Recognizer(db, threshold=args.threshold)
    detector = FaceDetector(device=device, detection_device=det_device, pad_mult=args.pad_mult)
    embedder = Embedder(model_name=args.embedder, device=device)
    tracker = SimpleTracker(smoothing_alpha=args.smoothing_alpha, max_missed=args.max_missed, stable_k=args.stable_k, stable_min_score=args.stable_min_score)
    liveness_detector = LivenessDetector(
        motion_threshold=args.liveness_motion_threshold,
        min_motion_frames=args.liveness_min_motion_frames,
        texture_threshold=args.liveness_texture_threshold
    ) if args.require_liveness else None

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
            landmarks_out = []

            if frame_count % args.detect_every == 0:
                boxes, probs, faces, landmarks = detector.detect(small)
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
                                # also push to history for stabilization
                                hist = tr.get('history')
                                if hist is not None:
                                    try:
                                        scv = float(score) if score is not None else None
                                    except Exception:
                                        scv = None
                                    hist.append((label, scv))
                        # Only process if crop_rgb and resized are valid
                        if resized is not None:
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
                    for i, (b, (name, score)) in enumerate(zip(boxes, matches)):
                        b_orig = box_scale_back(b, scale)
                        boxes_out.append(b_orig)
                        labels.append(name)
                        scores.append(score)
                        # scale landmarks back to original frame
                        if landmarks is not None and i < len(landmarks):
                            lm = np.array(landmarks[i])
                            lm_scaled = lm / scale
                            landmarks_out.append(lm_scaled)
                        else:
                            landmarks_out.append(None)
            else:
                # no detection; use previously tracked boxes
                boxes = []

            # tracker: update with boxes_out (in original frame coords)
            tracked = tracker.update(boxes_out, labels=labels, scores=scores)

            # Prioritize nearest (largest box area); draw in order
            def area(b):
                return max(0, (b[2] - b[0])) * max(0, (b[3] - b[1]))
            sorted_tracked = sorted(tracked.items(), key=lambda kv: area(kv[1]), reverse=True)

            # Draw tracked boxes only if there are tracked faces
            if sorted_tracked:
                # Compute attention score for each track
                attention_scores = {}
                for idx_sort, (tid, box) in enumerate(sorted_tracked):
                    tr = tracker.tracks.get(tid, {})
                    label = tr.get('stable_label', tr.get('label', 'Unknown'))
                    score = tr.get('stable_score', tr.get('score', None))
                    hits = int(tr.get('hits', 0))
                    # Liveness check if enabled
                    liveness_ok = True
                    lm_for_track = None
                    face_region = None
                    for i, (b, lm) in enumerate(zip(boxes_out, landmarks_out)):
                        if iou(box, b) > 0.3:
                            lm_for_track = lm
                            x1, y1, x2, y2 = [int(c) for c in box]
                            x1 = max(0, x1)
                            y1 = max(0, y1)
                            x2 = min(frame.shape[1], x2)
                            y2 = min(frame.shape[0], y2)
                            if x2 > x1 and y2 > y1:
                                face_region = frame[y1:y2, x1:x2]
                            break
                    if liveness_detector is not None and lm_for_track is not None:
                        liveness_result = liveness_detector.update(tid, lm_for_track, box, face_region)
                        liveness_ok = liveness_result['is_live']
                    # --- Attention scoring ---
                    # 1. Front-facing: use eye/nose/mouth landmarks
                    facing_score = 0.0
                    lip_score = 0.0
                    if lm_for_track is not None and isinstance(lm_for_track, np.ndarray) and lm_for_track.shape[0] >= 5:
                        # MTCNN: [left_eye, right_eye, nose, mouth_left, mouth_right]
                        left_eye, right_eye, nose, mouth_left, mouth_right = lm_for_track[:5]
                        # Front-facing: eyes horizontal, nose centered between eyes
                        eye_dx = abs(left_eye[0] - right_eye[0])
                        eye_dy = abs(left_eye[1] - right_eye[1])
                        nose_x = nose[0]
                        eye_center_x = (left_eye[0] + right_eye[0]) / 2.0
                        nose_offset = abs(nose_x - eye_center_x)
                        # Lower eye_dy and nose_offset = more front-facing
                        facing_score = max(0.0, 1.0 - (eye_dy / (eye_dx + 1e-5)) - (nose_offset / (eye_dx + 1e-5)))
                        # Lip movement: track mouth distance variance over last 10 frames
                        mouth_dist = np.linalg.norm(mouth_left - mouth_right)
                        if 'mouth_history' not in tr:
                            tr['mouth_history'] = deque(maxlen=10)
                        tr['mouth_history'].append(mouth_dist)
                        if len(tr['mouth_history']) >= 5:
                            lip_score = np.std(tr['mouth_history']) / (mouth_dist + 1e-5)
                    # Combine scores: weight front-facing 2x, lip movement 1x
                    attention_scores[tid] = 2.0 * facing_score + 1.0 * lip_score + (1.0 if liveness_ok else 0.0)
                # Find track with highest attention score
                best_tid = max(attention_scores, key=lambda k: attention_scores[k]) if attention_scores else None
                # Draw and greet only the best_tid
                for idx_sort, (tid, box) in enumerate(sorted_tracked):
                    tr = tracker.tracks.get(tid, {})
                    label = tr.get('stable_label', tr.get('label', 'Unknown'))
                    score = tr.get('stable_score', tr.get('score', None))
                    hits = int(tr.get('hits', 0))
                    # Always use last known smoothed box for drawing
                    smoothed_box = tr.get('smoothed', box)
                    liveness_ok = True
                    lm_for_track = None
                    face_region = None
                    for i, (b, lm) in enumerate(zip(boxes_out, landmarks_out)):
                        if iou(smoothed_box, b) > 0.3:
                            lm_for_track = lm
                            x1, y1, x2, y2 = [int(c) for c in smoothed_box]
                            x1 = max(0, x1)
                            y1 = max(0, y1)
                            x2 = min(frame.shape[1], x2)
                            y2 = min(frame.shape[0], y2)
                            if x2 > x1 and y2 > y1:
                                face_region = frame[y1:y2, x1:x2]
                            break
                    if liveness_detector is not None and lm_for_track is not None:
                        liveness_result = liveness_detector.update(tid, lm_for_track, smoothed_box, face_region)
                        liveness_ok = liveness_result['is_live']
                    # Draw box using smoothed_box
                    # Blue box for high attention (front-facing + lip movement)
                    facing_score = 0.0
                    lip_score = 0.0
                    if lm_for_track is not None and isinstance(lm_for_track, np.ndarray) and lm_for_track.shape[0] >= 5:
                        left_eye, right_eye, nose, mouth_left, mouth_right = lm_for_track[:5]
                        eye_dx = abs(left_eye[0] - right_eye[0])
                        eye_dy = abs(left_eye[1] - right_eye[1])
                        nose_x = nose[0]
                        eye_center_x = (left_eye[0] + right_eye[0]) / 2.0
                        nose_offset = abs(nose_x - eye_center_x)
                        facing_score = max(0.0, 1.0 - (eye_dy / (eye_dx + 1e-5)) - (nose_offset / (eye_dx + 1e-5)))
                        mouth_dist = np.linalg.norm(mouth_left - mouth_right)
                        if 'mouth_history' not in tr:
                            tr['mouth_history'] = deque(maxlen=10)
                        tr['mouth_history'].append(mouth_dist)
                        if len(tr['mouth_history']) >= 5:
                            lip_score = np.std(tr['mouth_history']) / (mouth_dist + 1e-5)
                    # Thresholds for blue box (tuned)
                    if liveness_detector is not None and not liveness_ok:
                        spoof_label = f"Spoof" if label == "Unknown" else f"{label} Spoof"
                        draw_box_label(frame, smoothed_box, f"{spoof_label}#{tid}", score, color=(0,0,255))
                    elif liveness_detector is not None and liveness_ok and facing_score > 0.4 and lip_score > 0.03:
                        draw_box_label(frame, smoothed_box, f"{label}#{tid}", score, color=(255,0,0))
                        print(f"[Attention] {label}#{tid}: facing_score={facing_score:.2f}, lip_score={lip_score:.2f} -> BLUE BOX")
                    elif liveness_detector is not None and liveness_ok:
                        draw_box_label(frame, smoothed_box, f"{label}#{tid}", score, color=(0,255,0))
                    else:
                        draw_box_label(frame, smoothed_box, f"{label}#{tid}", score)
                    # Only greet/respond to the best_tid
                    is_live = hits >= args.min_live_frames and liveness_ok
                    if args.greet and is_live and tid == best_tid:
                        if label != 'Unknown' and score is not None:
                            try:
                                sc_val = float(score)
                            except Exception:
                                sc_val = 0.0
                            if sc_val >= args.greet_threshold:
                                now = time.time()
                                last = last_greet_time.get(label, 0.0)
                                if now - last >= args.greet_interval:
                                    try:
                                        spoken_label = label.replace("Dr.", "Doctor").replace("Mr.", "Mister").replace("Ms.", "Miss").replace("Mrs.", "Missus")
                                        text = f"Hello {spoken_label}. Welcome."
                                        print(f"Greeting triggered for {label} (score={sc_val:.3f}) -> say: {text}")
                                        SAY_BIN = '/usr/bin/say'
                                        p = subprocess.Popen([SAY_BIN, text], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                                        print(f"Launched say pid={p.pid} via {SAY_BIN}")
                                        last_greet_time[label] = now
                                    except Exception as e:
                                        print(f"/usr/bin/say Popen failed: {e}")
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
                        elif args.greet_unknown:
                            now = time.time()
                            last = last_greet_time.get(f"unknown_{tid}", 0.0)
                            if now - last >= args.greet_interval:
                                try:
                                    text = "Hello there. Welcome."
                                    SAY_BIN = '/usr/bin/say'
                                    p = subprocess.Popen([SAY_BIN, text], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                                    last_greet_time[f"unknown_{tid}"] = now
                                except Exception: pass

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
