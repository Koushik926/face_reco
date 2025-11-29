"""Build and load face database.

Usage:
    python src/database.py --build --faces-dir ../campus_faces --out face_db.pkl
"""
import os
import argparse
import pickle
from typing import Dict
from pathlib import Path
from PIL import Image
import numpy as np
import sys
import torch

# Ensure local src imports work when running this file directly
sys.path.insert(0, str(Path(__file__).resolve().parent))
from detector import FaceDetector
from embedder import Embedder


def build_database(faces_dir: str, out_path: str, device=None) -> Dict:
    """Scan faces_dir and build a database mapping person_id -> {name, embedding, imgs}.

    faces_dir structure: <person_id>/<imgfiles>
    """
    # Force CPU for detection during DB build (more reliable, avoids MPS issues)
    cpu_device = torch.device('cpu')
    detector = FaceDetector(device=cpu_device, detection_device=cpu_device)
    embedder = Embedder(device=device or cpu_device)

    db = {}
    faces_dir = Path(faces_dir)
    print(f"Scanning faces directory: {faces_dir}")
    for person in sorted([d for d in faces_dir.iterdir() if d.is_dir()]):
        person_id = person.name
        print(f"\nProcessing person: {person_id}")
        imgs = []
        embeddings = []
        for img_path in sorted(person.glob('*')):
            if img_path.suffix.lower() not in ('.jpg', '.jpeg', '.png'):
                continue
            print(f"  Processing image: {img_path.name}")
            try:
                img = Image.open(img_path).convert('RGB')
                # Resize if too large (MTCNN works better with reasonable sizes)
                if max(img.size) > 1024:
                    ratio = 1024 / max(img.size)
                    new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
                    img = img.resize(new_size, Image.Resampling.LANCZOS)
                    print(f"    Resized to: {new_size}")
                
                img_cv = np.array(img)[:, :, ::-1].copy()  # PIL->OpenCV BGR
                boxes, probs, face_batch, landmarks = detector.detect(img_cv)
                
                if boxes:
                    print(f"    Detected {len(boxes)} face(s) with confidence: {probs}")
                else:
                    print(f"    No faces detected in this image")
                
                if face_batch is None or len(face_batch) == 0:
                    continue
                emb = embedder.embed(face_batch)
                # If multiple faces in the image, take first
                embeddings.append(emb[0])
                imgs.append(str(img_path))
                print(f"    ✓ Successfully extracted embedding")
            except Exception as e:
                print(f"    ✗ Error processing {img_path}: {e}")
                continue

        if not embeddings:
            print(f"No faces found for {person_id}, skipping.")
            continue

        mean_emb = np.mean(np.stack(embeddings, axis=0), axis=0)
        # ensure normalized
        mean_emb = mean_emb / np.linalg.norm(mean_emb)

        db[person_id] = {
            "name": person_id.replace('_', ' '),
            "embedding": mean_emb.astype(np.float32),
            "imgs": imgs,
        }

    # save
    out_path = Path(out_path)
    with open(out_path, 'wb') as f:
        pickle.dump(db, f)

    print(f"Database built with {len(db)} identities -> {out_path}")
    return db


def load_database(path: str) -> Dict:
    with open(path, 'rb') as f:
        return pickle.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--build', action='store_true', help='Build DB from campus_faces')
    parser.add_argument('--faces-dir', default=os.path.join(os.path.dirname(__file__), '..', 'campus_faces'), help='Faces folder')
    parser.add_argument('--out', default=os.path.join(os.path.dirname(__file__), '..', 'face_db.pkl'))
    args = parser.parse_args()

    if args.build:
        device = None
        device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
        build_database(args.faces_dir, args.out, device=device)


if __name__ == '__main__':
    main()
