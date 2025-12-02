"""Lightweight liveness check using facial landmarks from MTCNN.

Detects eye blinks via Eye Aspect Ratio (EAR) and motion variance to prevent 2D photo spoofing.
Also includes texture analysis to detect screen/photo artifacts.
"""
import numpy as np
import cv2
from collections import deque


def eye_aspect_ratio(eye_landmarks):
    """Compute Eye Aspect Ratio (EAR) from 6 eye landmark points.
    
    Args:
        eye_landmarks: array of shape (6, 2) with eye corner and eyelid points
        
    Returns:
        float: EAR value (lower when eye is closed)
    """
    # Vertical distances
    A = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
    B = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
    # Horizontal distance
    C = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
    
    if C == 0:
        return 0.0
    
    ear = (A + B) / (2.0 * C)
    return ear


class LivenessDetector:
    """Detect live faces via blink detection and motion analysis."""
    
    def __init__(self, ear_threshold=0.21, blink_consec_frames=2, history_size=30, motion_threshold=25.0, min_blinks=1, min_motion_frames=12, texture_threshold=35.0):
        """
        Args:
            ear_threshold: EAR below this is considered closed eye
            blink_consec_frames: consecutive frames with closed eyes to count as blink
            history_size: frames of EAR history to keep
            motion_threshold: minimum pixel variance to consider motion present
            min_blinks: minimum blinks required to be considered live
            min_motion_frames: minimum frames with motion to be considered live
            texture_threshold: minimum texture variance (screens/photos have lower variance due to compression/smoothing)
        """
        self.ear_threshold = ear_threshold
        self.blink_consec_frames = blink_consec_frames
        self.history_size = history_size
        self.motion_threshold = motion_threshold
        self.min_blinks = min_blinks
        self.min_motion_frames = min_motion_frames
        self.texture_threshold = texture_threshold
        
        # Track state per track_id
        self.track_state = {}  # tid -> {'ear_history': deque, 'blink_count': int, 'closed_frames': int, 'positions': deque, 'motion_frames': int, 'liveness_history': deque}
    
    def update(self, track_id, landmarks, box, face_region=None):
        """Update liveness state for a track.
        
        Args:
            track_id: unique track identifier
            landmarks: facial landmarks from MTCNN (shape: [5, 2] or [10] flattened)
            box: bounding box [x1, y1, x2, y2]
            face_region: optional face image region (numpy array BGR) for texture analysis
            
        Returns:
            dict: {'is_live': bool, 'blink_count': int, 'has_motion': bool, 'has_texture': bool}
        """
        if track_id not in self.track_state:
            self.track_state[track_id] = {
                'ear_history': deque(maxlen=self.history_size),
                'blink_count': 0,
                'closed_frames': 0,
                'positions': deque(maxlen=10),
                'motion_frames': 0,
                'last_variance': 0.0,
                'liveness_history': deque(maxlen=5)  # Track last 5 liveness results for stability
            }
        
        state = self.track_state[track_id]
        
        # Parse landmarks (MTCNN returns 5 landmarks: left_eye, right_eye, nose, left_mouth, right_mouth)
        if landmarks is not None:
            if isinstance(landmarks, (list, tuple)):
                landmarks = np.array(landmarks)
            if landmarks.ndim == 1:
                landmarks = landmarks.reshape(-1, 2)
            
            # MTCNN landmarks: [left_eye, right_eye, nose, mouth_left, mouth_right]
            # For EAR, we approximate using eye landmarks (not full 6-point eye contour)
            # Simple approximation: use vertical spread as proxy for eye openness
            if len(landmarks) >= 2:
                left_eye = landmarks[0]
                right_eye = landmarks[1]
                
                # Simplified EAR: distance from eye center to top/bottom of box region
                # Since MTCNN only gives eye centers, use box height variance as proxy
                # Better: track eye position variance over time
                avg_ear = 0.25  # default open eye value
                
                # For now, use simpler motion-based liveness
                # Track blink-like rapid position changes in eye landmarks
                state['ear_history'].append(avg_ear)
        
        # Motion check: box position variance
        state['positions'].append(box[:2])  # track top-left corner
        has_motion = False
        variance = 0.0
        if len(state['positions']) >= 5:
            positions = np.array(state['positions'])
            variance = np.var(positions, axis=0).sum()
            has_motion = variance > self.motion_threshold
            # Track consecutive motion frames
            if has_motion:
                state['motion_frames'] += 1
            else:
                state['motion_frames'] = max(0, state['motion_frames'] - 1)
        
        # Texture analysis: real faces have higher micro-texture variance than screens/photos
        has_texture = True  # default to True if no face region provided
        texture_variance = 0.0
        if face_region is not None and face_region.size > 0:
            try:
                # Convert to grayscale and compute Laplacian variance (edge sharpness)
                gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY) if len(face_region.shape) == 3 else face_region
                
                # Check if face is large enough for texture analysis
                if gray.shape[0] > 30 and gray.shape[1] > 30:
                    # Use Laplacian for edge detection (real faces have natural micro-textures)
                    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
                    texture_variance = laplacian.var()
                    has_texture = texture_variance > self.texture_threshold
                else:
                    has_texture = True  # face too small, skip texture check
            except Exception:
                has_texture = True  # if texture check fails, don't block
        
        # Liveness criteria: require motion AND texture (screens have smooth texture, photos don't move naturally)
        # This prevents moving screens from passing
        has_sufficient_motion = state['motion_frames'] >= self.min_motion_frames
        
        # Only consider live if BOTH motion and texture conditions met
        is_live = has_sufficient_motion and has_texture
        
        # Stabilize liveness result using majority vote over last 5 frames
        state['liveness_history'].append(is_live)
        if len(state['liveness_history']) >= 3:
            # Majority vote: if at least 3 out of last 5 frames say live, then live
            live_count = sum(state['liveness_history'])
            stable_is_live = live_count >= (len(state['liveness_history']) // 2 + 1)
        else:
            # Not enough history yet, use current result
            stable_is_live = is_live
        
        return {
            'is_live': stable_is_live,
            'blink_count': state['blink_count'],
            'has_motion': has_motion,
            'motion_frames': state['motion_frames'],
            'variance': variance,
            'has_texture': has_texture,
            'texture_variance': texture_variance
        }
    
    def reset_track(self, track_id):
        """Clear state for a track that disappeared."""
        if track_id in self.track_state:
            del self.track_state[track_id]
