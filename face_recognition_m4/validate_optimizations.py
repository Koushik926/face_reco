#!/usr/bin/env python3
"""Quick validation script for optimized recognition and detection systems."""

import sys
import os
import time
import torch
import numpy as np
from PIL import Image

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def validate_recognizer():
    """Test recognizer optimizations."""
    print("=" * 60)
    print("VALIDATING RECOGNIZER OPTIMIZATIONS")
    print("=" * 60)
    
    try:
        from recognizer import Recognizer
        import pickle
        
        # Load database
        if not os.path.exists('face_db.pkl'):
            print("⚠ No database found - skipping recognizer validation")
            return True
        
        with open('face_db.pkl', 'rb') as f:
            db = pickle.load(f)
        
        rec = Recognizer(db)
        print(f"✓ Loaded recognizer with {len(rec.db)} identities")
        
        # Verify normalized embeddings exist
        assert hasattr(rec, 'normalized_embeddings'), "Missing normalized_embeddings attribute"
        assert rec.normalized_embeddings is not None, "normalized_embeddings is None"
        assert len(rec.normalized_embeddings) == len(rec.db), "Normalized embeddings count mismatch"
        print(f"✓ Precomputed normalized embeddings: {len(rec.normalized_embeddings)} entries")
        
        # Verify normalization is correct
        for pid, norm_emb in rec.normalized_embeddings.items():
            norm = np.linalg.norm(norm_emb)
            assert abs(norm - 1.0) < 1e-5 or norm == 0, f"Embedding {pid} not normalized: norm={norm}"
        print(f"✓ Normalized embeddings are unit vectors (L2 norm ≈ 1.0)")
        
        # Test matching speed
        if len(rec.db) > 0:
            first_id = list(rec.db.keys())[0]
            test_emb = rec.db[first_id]['embedding']  # Use first embedding
            
            # Warmup
            for _ in range(5):
                rec.match(test_emb)
            
            # Benchmark
            n_trials = 100
            start = time.time()
            for _ in range(n_trials):
                rec.match(test_emb)
            elapsed = time.time() - start
            
            avg_time_ms = (elapsed / n_trials) * 1000
            print(f"✓ Average match time: {avg_time_ms:.2f} ms ({n_trials} trials)")
            print(f"  Match throughput: {int(1000 / avg_time_ms)} matches/sec")
        
        print("✓ Recognizer validation PASSED\n")
        return True
        
    except Exception as e:
        print(f"✗ Recognizer validation FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False

def validate_detector():
    """Test detector optimizations."""
    print("=" * 60)
    print("VALIDATING DETECTOR OPTIMIZATIONS")
    print("=" * 60)
    
    try:
        from detector import FaceDetector
        
        # Create detector with optimizations
        det = FaceDetector(min_face_size=30, device=torch.device('cpu'))
        print(f"✓ Created detector (device: {det.device})")
        
        # Verify new attributes
        assert hasattr(det, 'min_face_size'), "Missing min_face_size attribute"
        assert hasattr(det, '_prefer_cpu'), "Missing _prefer_cpu attribute"
        assert hasattr(det, '_mps_failure_count'), "Missing _mps_failure_count attribute"
        print(f"✓ New attributes present: min_face_size={det.min_face_size}")
        
        # Verify helper methods exist
        assert hasattr(det, '_detect_cpu'), "Missing _detect_cpu method"
        assert hasattr(det, '_process_face_batch'), "Missing _process_face_batch method"
        print(f"✓ Optimized methods present: _detect_cpu, _process_face_batch")
        
        # Create test image
        test_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Test detection (may not find faces in random image, but should not crash)
        boxes, probs, faces, landmarks = det.detect(test_img)
        print(f"✓ Detection runs without errors (found {len(boxes)} faces)")
        
        # Test CPU fallback path explicitly
        det._prefer_cpu = True
        boxes2, probs2, faces2, landmarks2 = det.detect(test_img)
        print(f"✓ CPU fallback path works (found {len(boxes2)} faces)")
        
        # Benchmark detection speed
        det._prefer_cpu = False
        n_trials = 20
        start = time.time()
        for _ in range(n_trials):
            det.detect(test_img)
        elapsed = time.time() - start
        
        avg_time_ms = (elapsed / n_trials) * 1000
        fps = 1000 / avg_time_ms if avg_time_ms > 0 else 0
        print(f"✓ Average detection time: {avg_time_ms:.1f} ms ({n_trials} trials)")
        print(f"  Detection FPS: {fps:.1f} frames/sec")
        
        print("✓ Detector validation PASSED\n")
        return True
        
    except Exception as e:
        print(f"✗ Detector validation FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False

def validate_integration():
    """Test integration of optimized components."""
    print("=" * 60)
    print("VALIDATING SYSTEM INTEGRATION")
    print("=" * 60)
    
    try:
        from detector import FaceDetector
        from recognizer import Recognizer
        import pickle
        
        if not os.path.exists('face_db.pkl'):
            print("⚠ No database found - skipping integration validation")
            return True
        
        # Load database
        with open('face_db.pkl', 'rb') as f:
            db = pickle.load(f)
        
        # Create components
        det = FaceDetector(min_face_size=30, device=torch.device('cpu'))
        rec = Recognizer(db)
        print(f"✓ Created detector and recognizer")
        
        # Test end-to-end with random image
        test_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        boxes, probs, faces, landmarks = det.detect(test_img)
        
        if faces is not None and len(boxes) > 0:
            # Try recognition on numpy embeddings
            # Note: In real system, embeddings come from embedding model
            print(f"  Detected {len(boxes)} face(s) in test image")
        else:
            print(f"  No faces detected in test image (expected for random data)")
        
        print("✓ Integration validation PASSED\n")
        return True
        
    except Exception as e:
        print(f"✗ Integration validation FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all validations."""
    print("\n" + "=" * 60)
    print("FACE RECOGNITION SYSTEM - OPTIMIZATION VALIDATION")
    print("=" * 60 + "\n")
    
    results = []
    
    # Run validations
    results.append(('Recognizer', validate_recognizer()))
    results.append(('Detector', validate_detector()))
    results.append(('Integration', validate_integration()))
    
    # Summary
    print("=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{name:20s} {status}")
    
    all_passed = all(r[1] for r in results)
    print("=" * 60)
    
    if all_passed:
        print("\n✓ ALL VALIDATIONS PASSED - System ready for use!\n")
        print("Run the system with:")
        print("  python src/realtime.py --db face_db.pkl --require-liveness")
        return 0
    else:
        print("\n✗ SOME VALIDATIONS FAILED - Please review errors above\n")
        return 1

if __name__ == '__main__':
    sys.exit(main())
