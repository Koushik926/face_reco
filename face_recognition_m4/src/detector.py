"""MTCNN detector wrapper optimized for MPS/Apple Silicon.

Uses facenet-pytorch MTCNN for detection and alignment. Returns cropped face
tensors ready for embedding (Nx3x160x160) and corresponding boxes + probs.
"""
from typing import List, Tuple, Optional
import time
import torch
from facenet_pytorch import MTCNN
import numpy as np
from PIL import Image
import cv2


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class FaceDetector:
    """Wrapper around facenet-pytorch MTCNN.

    Methods
    -------
    detect(frame):
        Detect faces in a BGR OpenCV frame. Returns boxes, probs, face_tensors.
    """

    def __init__(self, image_size: int = 160, margin: int = 20, keep_all: bool = True, device: Optional[torch.device] = None, detection_device: Optional[torch.device] = None, pad_mult: int = 64, min_face_size: int = 20):
        """device: preferred device for tensors/embeddings (e.g., mps)
        detection_device: device for running MTCNN. If None, uses same as device.
        min_face_size: minimum face size in pixels (smaller faces filtered out)
        Note: on macOS MPS some adaptive pooling ops in facenet-pytorch raise a
        RuntimeError when input sizes are not divisible by output sizes. In that
        case we automatically retry detection on CPU and keep embeddings on the
        preferred device to benefit from MPS for embedding.
        """
        self.device = device or get_device()
        self.detection_device = detection_device or self.device
        # padding multiple used to pad images before running MTCNN on MPS
        self.pad_mult = int(pad_mult)
        # minimum face size for quality filtering
        self.min_face_size = min_face_size
        # MTCNN will produce aligned faces sized to image_size
        self.mtcnn = MTCNN(image_size=image_size, margin=margin, keep_all=keep_all, 
                           device=self.detection_device, min_face_size=min_face_size)
        # Cache a CPU MTCNN to avoid repeated re-creation during runtime fallback
        self._cpu_mtcnn = None
        # Warning throttle for the MPS adaptive-pool issue (seconds)
        self._mps_warn_time = 0.0
        self._mps_warn_interval = 5.0
        # Track if we should prefer CPU due to repeated MPS failures
        self._prefer_cpu = False
        self._mps_failure_count = 0
        self._mps_failure_threshold = 3

    def _to_pil(self, frame: np.ndarray) -> Image.Image:
        # Convert BGR OpenCV frame to PIL RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)

    def detect(self, frame: np.ndarray) -> Tuple[List[np.ndarray], List[float], Optional[torch.Tensor], Optional[List[np.ndarray]]]:
        """Detect faces in a BGR OpenCV frame.

        Returns:
            boxes: list of [x1, y1, x2, y2]
            probs: list of detection probabilities
            face_tensors: torch.Tensor of shape (N,3,H,W) on the detector device (or None)
            landmarks: list of landmark arrays (Nx5x2) or None
        """
        if frame is None:
            return [], [], None, None

        pil = self._to_pil(frame)

        # If we've had repeated MPS failures, switch to CPU detection
        use_cpu_detection = self._prefer_cpu or (getattr(self.detection_device, 'type', None) != 'mps')
        
        if use_cpu_detection:
            # Use CPU detection directly
            return self._detect_cpu(pil)

        # Prepare padded image when running MTCNN on MPS to avoid adaptive pool errors
        pil_for_mtcnn = pil
        pad_left = pad_top = 0
        try:
            arr = np.asarray(pil)
            h, w = arr.shape[:2]
            # Larger multiple may avoid MPS adaptive pool issues for many inputs
            mult = int(self.pad_mult)
            new_h = ((h + mult - 1) // mult) * mult
            new_w = ((w + mult - 1) // mult) * mult
            pad_h = new_h - h
            pad_w = new_w - w
            if pad_h != 0 or pad_w != 0:
                pad_top = pad_h // 2
                pad_bottom = pad_h - pad_top
                pad_left = pad_w // 2
                pad_right = pad_w - pad_left
                padded = cv2.copyMakeBorder(arr, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
                pil_for_mtcnn = Image.fromarray(padded)
        except Exception:
            pil_for_mtcnn = pil

        # Try MPS detection with single detect call (includes landmarks)
        try:
            with torch.no_grad():
                boxes, probs, landmarks = self.mtcnn.detect(pil_for_mtcnn, landmarks=True)
                
            # Successful MPS detection - reset failure counter
            self._mps_failure_count = 0
            
            # Process results
            if boxes is None:
                return [], [], None, None
                
            # Remove padding offset if applicable
            if isinstance(boxes, np.ndarray) and (pad_left or pad_top):
                offset = np.array([pad_left, pad_top, pad_left, pad_top])
                boxes = boxes - offset
                if landmarks is not None:
                    landmarks = landmarks - np.array([pad_left, pad_top])
            
            # Get aligned face crops for embeddings
            try:
                with torch.no_grad():
                    face_imgs = self.mtcnn(pil_for_mtcnn, return_prob=False)
            except Exception:
                # Fallback: return boxes without face tensors
                boxes_int = boxes.astype(int).tolist()
                probs_list = probs.tolist() if probs is not None else []
                landmarks_list = landmarks.tolist() if landmarks is not None else None
                return boxes_int, probs_list, None, landmarks_list
            
            # Convert face images to tensors
            if face_imgs is None:
                boxes_int = boxes.astype(int).tolist()
                probs_list = probs.tolist() if probs is not None else []
                landmarks_list = landmarks.tolist() if landmarks is not None else None
                return boxes_int, probs_list, None, landmarks_list
            
            face_batch = self._process_face_batch(face_imgs)
            boxes_int = boxes.astype(int).tolist()
            probs_list = probs.tolist() if probs is not None else []
            landmarks_list = landmarks.tolist() if landmarks is not None else None
            
            return boxes_int, probs_list, face_batch, landmarks_list
            
        except RuntimeError as e:
            msg = str(e)
            # Known MPS adaptive pool issue: fallback to CPU for detection
            if 'Adaptive pool MPS' in msg or 'input sizes must be divisible' in msg or 'Non-divisible input sizes' in msg:
                self._mps_failure_count += 1
                now = time.time()
                
                # Throttle warning prints to avoid spamming the terminal
                if now - self._mps_warn_time >= self._mps_warn_interval:
                    print(f'Warning: MPS adaptive pool issue detected during MTCNN; using CPU fallback (failures: {self._mps_failure_count}).')
                    self._mps_warn_time = now
                
                # If we've had too many failures, switch to CPU permanently
                if self._mps_failure_count >= self._mps_failure_threshold:
                    print(f'Switching to CPU detection permanently after {self._mps_failure_count} MPS failures.')
                    self._prefer_cpu = True
                
                return self._detect_cpu(pil)
            else:
                # Other runtime error -> fallback to CPU
                self._mps_failure_count += 1
                return self._detect_cpu(pil)
        except Exception:
            # Generic fallback: CPU detection
            self._mps_failure_count += 1
            return self._detect_cpu(pil)
    
    def _detect_cpu(self, pil: Image.Image) -> Tuple[List[np.ndarray], List[float], Optional[torch.Tensor], Optional[List[np.ndarray]]]:
        """CPU fallback detection with optimized single-pass processing."""
        # Create cached CPU mtcnn if needed
        if self._cpu_mtcnn is None:
            self._cpu_mtcnn = MTCNN(
                image_size=self.mtcnn.image_size, 
                margin=self.mtcnn.margin, 
                keep_all=self.mtcnn.keep_all, 
                device=torch.device('cpu'),
                min_face_size=self.min_face_size
            )
        
        try:
            with torch.no_grad():
                # Single detect call for boxes and landmarks
                boxes, probs, landmarks = self._cpu_mtcnn.detect(pil, landmarks=True)
                
            if boxes is None:
                return [], [], None, None
            
            # Get aligned face crops
            try:
                with torch.no_grad():
                    face_imgs = self._cpu_mtcnn(pil, return_prob=False)
            except Exception:
                # Return boxes without face tensors
                boxes_int = boxes.astype(int).tolist()
                probs_list = probs.tolist() if probs is not None else []
                landmarks_list = landmarks.tolist() if landmarks is not None else None
                return boxes_int, probs_list, None, landmarks_list
            
            if face_imgs is None:
                boxes_int = boxes.astype(int).tolist()
                probs_list = probs.tolist() if probs is not None else []
                landmarks_list = landmarks.tolist() if landmarks is not None else None
                return boxes_int, probs_list, None, landmarks_list
            
            face_batch = self._process_face_batch(face_imgs)
            boxes_int = boxes.astype(int).tolist()
            probs_list = probs.tolist() if probs is not None else []
            landmarks_list = landmarks.tolist() if landmarks is not None else None
            
            return boxes_int, probs_list, face_batch, landmarks_list
            
        except Exception:
            return [], [], None, None
    
    def _process_face_batch(self, face_imgs) -> Optional[torch.Tensor]:
        """Convert face images to normalized tensor batch."""
        # Normalize face_imgs into a list for uniform processing
        face_list = []
        if face_imgs is None:
            return None
        elif isinstance(face_imgs, torch.Tensor):
            # tensor can be (N,3,H,W) or (3,H,W)
            if face_imgs.dim() == 4:
                face_list = [face_imgs[i] for i in range(face_imgs.size(0))]
            elif face_imgs.dim() == 3:
                face_list = [face_imgs]
            else:
                return None
        elif isinstance(face_imgs, Image.Image):
            face_list = [face_imgs]
        elif isinstance(face_imgs, (list, tuple)):
            face_list = list(face_imgs)
        else:
            return None

        if not face_list:
            return None

        # face_list items may be PIL.Image.Image or torch.Tensor
        face_tensors = []
        for f in face_list:
            if isinstance(f, Image.Image):
                # convert to tensor in same normalization expected by InceptionResnet
                arr = np.asarray(f).astype(np.uint8)
                # facenet models expect (3,H,W) float in [-1,1]
                t = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
                t = (t - 0.5) / 0.5
                face_tensors.append(t)
            elif isinstance(f, torch.Tensor):
                # ensure float and normalized
                t = f
                if t.dtype != torch.float32:
                    t = t.float()
                face_tensors.append(t)
            else:
                # Unknown type
                continue

        if not face_tensors:
            return None

        face_batch = torch.stack(face_tensors, dim=0).to(self.device)
        return face_batch
