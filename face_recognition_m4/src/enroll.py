"""Enrollment helper: capture webcam images for a person into campus_faces/<person_id>/

Usage:
    python src/enroll.py --id john_doe --count 10
    python src/enroll.py --id john_doe --count 10 --auto  # Auto-capture mode

This will create `campus_faces/john_doe/` and save images img_0001.jpg ...
"""
import argparse
import os
import sys
from pathlib import Path
import cv2
import time
import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent))
from detector import FaceDetector
from liveness import LivenessDetector


def check_face_quality(box, frame, landmarks=None):
    """Check if face is suitable for enrollment.
    
    Returns:
        (bool, str): (is_good, reason)
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    
    # Face size check - should be at least 8% of frame (was 15%)
    face_area = (x2 - x1) * (y2 - y1)
    frame_area = h * w
    if face_area < 0.08 * frame_area:
        return False, "Face too small - move closer"
    
    if face_area > 0.7 * frame_area:
        return False, "Face too large - move back"
    
    # Face position check - should be centered
    face_center_x = (x1 + x2) / 2
    face_center_y = (y1 + y2) / 2
    
    if abs(face_center_x - w/2) > w * 0.3:
        return False, "Face not centered - move to center"
    
    if abs(face_center_y - h/2) > h * 0.3:
        return False, "Face not centered vertically"
    
    # Brightness check
    face_roi = frame[int(y1):int(y2), int(x1):int(x2)]
    if face_roi.size > 0:
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY) if len(face_roi.shape) == 3 else face_roi
        brightness = gray.mean()
        if brightness < 60:
            return False, "Too dark - improve lighting"
        if brightness > 200:
            return False, "Too bright - reduce lighting"
    
    return True, "Good quality"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', help='Person id (folder name). If not provided, will ask interactively.')
    parser.add_argument('--count', type=int, default=30, help='Number of images to capture (recommend 30 for all angles)')
    parser.add_argument('--out', default=str(Path(__file__).resolve().parent.parent / 'campus_faces'))
    parser.add_argument('--cam', type=int, default=0)
    parser.add_argument('--auto', action='store_true', help='Auto-capture mode (captures automatically when face is detected)')
    parser.add_argument('--delay', type=float, default=0.5, help='Delay between auto-captures in seconds')
    parser.add_argument('--require-liveness', action='store_true', help='Require liveness check (prevents screen spoofing)')
    parser.add_argument('--skip-build', action='store_true', help='Skip automatic database rebuild after enrollment')
    args = parser.parse_args()

    # Ask for person ID if not provided
    if not args.id:
        print("\n🎯 Face Recognition Enrollment")
        print("=" * 50)
        person_id = input("Enter person's name (e.g., John_Doe, Dr_Smith): ").strip()
        if not person_id:
            print("❌ Name cannot be empty!")
            return
        args.id = person_id
    
    # Validate and sanitize the ID
    args.id = args.id.replace(' ', '_')  # Replace spaces with underscores
    
    out_dir = Path(args.out) / args.id
    
    # Check if person already exists
    if out_dir.exists() and list(out_dir.glob('*.jpg')):
        print(f"\n⚠️  Person '{args.id}' already has {len(list(out_dir.glob('*.jpg')))} images enrolled.")
        response = input("Do you want to add more images? (y/n): ").strip().lower()
        if response != 'y':
            print("❌ Enrollment cancelled.")
            return
        print(f"✅ Adding more images to existing enrollment for '{args.id}'")
    else:
        print(f"\n✅ Creating new enrollment for '{args.id}'")
    
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print('Cannot open camera')
        return

    # Initialize face detector
    detector = FaceDetector()
    liveness_detector = LivenessDetector(
        motion_threshold=5.0,
        min_motion_frames=3,
        texture_threshold=10.0
    ) if args.require_liveness else None
    
    if args.auto:
        print(f"🎯 AUTO-CAPTURE MODE: Capturing {args.count} images to {out_dir}")
        print("📸 Face will be captured automatically when detected")
        print("➡️  Please slowly turn your head left, right, up, down, and tilt to show all angles!")
        print("➡️  Try to look straight, then left, right, up, down, and tilt your chin. This helps recognition in all poses.")
        print("➡️  Progress will be shown as images are captured.")
        if args.require_liveness:
            print("✅ Liveness check ENABLED - move your head naturally")
    else:
        print(f"📸 MANUAL MODE: Capturing {args.count} images to {out_dir}")
        print("Press SPACE to capture, Q to quit")
    
    saved = 0
    idx = len(list(out_dir.glob('*.jpg'))) + 1
    last_capture_time = 0
    track_id = 0
    
    # Dynamic captions for angles
    angle_captions = [
        "Look straight ahead",
        "Turn your head to the LEFT",
        "Turn your head to the RIGHT",
        "Look UP",
        "Look DOWN",
        "Tilt your head LEFT",
        "Tilt your head RIGHT",
        "Smile or change expression",
        "Blink or close eyes briefly",
        "Show your face in all angles!"
    ]
    try:
        while saved < args.count:
            ret, frame = cap.read()
            if not ret:
                break
            display = frame.copy()
            boxes, probs, faces, landmarks = detector.detect(frame)
            face_detected = False
            quality_ok = False
            liveness_ok = True
            quality_msg = ""
            if boxes and len(boxes) > 0:
                box = boxes[0]
                prob = probs[0] if probs else 0
                lm = landmarks[0] if landmarks else None
                face_detected = True
                quality_ok, quality_msg = check_face_quality(box, frame)
                if liveness_detector and lm is not None:
                    x1, y1, x2, y2 = [int(c) for c in box]
                    face_region = frame[y1:y2, x1:x2]
                    liveness_result = liveness_detector.update(track_id, lm, box, face_region)
                    liveness_ok = liveness_result['is_live']
                x1, y1, x2, y2 = [int(c) for c in box]
                if quality_ok and liveness_ok:
                    color = (0, 255, 0)
                elif not liveness_ok:
                    color = (0, 0, 255)
                else:
                    color = (0, 165, 255)
                cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                if not liveness_ok:
                    msg = "⚠️ SPOOF DETECTED - Use real face"
                    cv2.putText(display, msg, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                elif not quality_ok:
                    cv2.putText(display, quality_msg, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            # Progress bar
            bar_width = 400
            bar_height = 30
            bar_x = 10
            bar_y = display.shape[0] - 50
            cv2.rectangle(display, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
            progress = int((saved / args.count) * bar_width)
            cv2.rectangle(display, (bar_x, bar_y), (bar_x + progress, bar_y + bar_height), (0, 255, 0), -1)
            progress_text = f"{saved}/{args.count} captured"
            cv2.putText(display, progress_text, (bar_x + 10, bar_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            # Dynamic angle caption (top center, large font)
            caption_idx = min(saved // max(1, args.count // len(angle_captions)), len(angle_captions) - 1)
            angle_caption = angle_captions[caption_idx]
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.6
            thickness = 4
            text_size, _ = cv2.getTextSize(angle_caption, font, font_scale, thickness)
            text_x = (display.shape[1] - text_size[0]) // 2
            text_y = 60
            cv2.putText(display, angle_caption, (text_x, text_y), font, font_scale, (0, 255, 255), thickness)
            # Instructions
            if args.auto:
                if face_detected and quality_ok and liveness_ok:
                    status = "✅ Ready - Capturing..."
                    status_color = (0, 255, 0)
                elif not face_detected:
                    status = "👤 No face detected"
                    status_color = (0, 165, 255)
                elif not liveness_ok:
                    status = "⚠️ Spoof detected"
                    status_color = (0, 0, 255)
                else:
                    status = f"⚠️ {quality_msg}"
                    status_color = (0, 165, 255)
            else:
                status = "Press SPACE to capture, Q to quit"
                status_color = (255, 255, 255)
            cv2.putText(display, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
            cv2.imshow('Enrollment', display)
            
            # Auto-capture logic
            if args.auto and face_detected and quality_ok and liveness_ok:
                current_time = time.time()
                if current_time - last_capture_time >= args.delay:
                    fname = out_dir / f"img_{idx:04d}.jpg"
                    cv2.imwrite(str(fname), frame)
                    print(f'✅ Saved {saved + 1}/{args.count}: {fname}')
                    saved += 1
                    idx += 1
                    last_capture_time = current_time
            
            # Manual capture logic
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == 32 and not args.auto:  # SPACE
                if face_detected and quality_ok and liveness_ok:
                    fname = out_dir / f"img_{idx:04d}.jpg"
                    cv2.imwrite(str(fname), frame)
                    print(f'✅ Saved {saved + 1}/{args.count}: {fname}')
                    saved += 1
                    idx += 1
                else:
                    if not face_detected:
                        print('❌ No face detected - cannot capture')
                    elif not liveness_ok:
                        print('❌ Spoof detected - use real face')
                    else:
                        print(f'❌ {quality_msg}')
        
        print(f'\n✅ Enrollment complete! Captured {saved} images to {out_dir}')
        print(f'👤 Person: {args.id}')
        print(f'📸 Total images: {len(list(out_dir.glob("*.jpg")))}')

        # Automatically rebuild the face database unless skipped
        if not args.skip_build:
            import threading
            def rebuild_db():
                try:
                    from database import build_database
                    project_root = Path(__file__).resolve().parent.parent
                    faces_dir = Path(args.out)
                    db_path = project_root / 'face_db.pkl'
                    print('\n💾 Rebuilding face database automatically...')
                    print(f'   Faces dir: {faces_dir}')
                    print(f'   Output DB: {db_path}')
                    build_database(str(faces_dir), str(db_path))
                    print('\n✅ Database rebuilt successfully!')
                    print('🚀 You can now run recognition:')
                    print('   python src/realtime.py --db face_db.pkl --require-liveness')
                except Exception as e:
                    print(f"\n⚠️ Automatic database rebuild failed: {e}")
                    print('You can rebuild manually with:')
                    print('   python src/database.py --build --faces-dir campus_faces --out face_db.pkl')
            threading.Thread(target=rebuild_db, daemon=True).start()
        else:
            print('\n� Skipped automatic database rebuild (--skip-build provided).')
            print('To rebuild manually:')
            print('   python src/database.py --build --faces-dir campus_faces --out face_db.pkl')
        
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
