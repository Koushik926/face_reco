"""Enrollment helper: capture webcam images for a person into campus_faces/<person_id>/

Usage:
    python src/enroll.py --id john_doe --count 10

This will create `campus_faces/john_doe/` and save images img_0001.jpg ...
"""
import argparse
import os
from pathlib import Path
import cv2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', required=True, help='Person id (folder name)')
    parser.add_argument('--count', type=int, default=10, help='Number of images to capture')
    parser.add_argument('--out', default=str(Path(__file__).resolve().parent.parent / 'campus_faces'))
    parser.add_argument('--cam', type=int, default=0)
    args = parser.parse_args()

    out_dir = Path(args.out) / args.id
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print('Cannot open camera')
        return

    print(f"Capturing {args.count} images to {out_dir}. Press SPACE to capture, q to quit.")
    saved = 0
    idx = len(list(out_dir.glob('*.jpg'))) + 1
    try:
        while saved < args.count:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.putText(frame, f"Capture {saved}/{args.count} - Press SPACE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            cv2.imshow('Enroll', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == 32:  # SPACE
                fname = out_dir / f"img_{idx:04d}.jpg"
                cv2.imwrite(str(fname), frame)
                print('Saved', fname)
                saved += 1
                idx += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
