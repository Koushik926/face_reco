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

    def __init__(self, image_size: int = 160, margin: int = 20, keep_all: bool = True, device: Optional[torch.device] = None, detection_device: Optional[torch.device] = None, pad_mult: int = 64):
        """device: preferred device for tensors/embeddings (e.g., mps)
        detection_device: device for running MTCNN. If None, uses same as device.
        Note: on macOS MPS some adaptive pooling ops in facenet-pytorch raise a
        RuntimeError when input sizes are not divisible by output sizes. In that
        case we automatically retry detection on CPU and keep embeddings on the
        preferred device to benefit from MPS for embedding.
        """
        self.device = device or get_device()
        self.detection_device = detection_device or self.device
        # padding multiple used to pad images before running MTCNN on MPS
        self.pad_mult = int(pad_mult)
        # MTCNN will produce aligned faces sized to image_size
        self.mtcnn = MTCNN(image_size=image_size, margin=margin, keep_all=keep_all, device=self.detection_device)
        # Cache a CPU MTCNN to avoid repeated re-creation during runtime fallback
        self._cpu_mtcnn = None
        # Warning throttle for the MPS adaptive-pool issue (seconds)
        self._mps_warn_time = 0.0
        self._mps_warn_interval = 5.0

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

        # Prepare padded image when running MTCNN on MPS to avoid adaptive pool errors
        pil_for_mtcnn = pil
        pad_left = pad_top = 0
        if getattr(self.detection_device, 'type', None) == 'mps':
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

        # Run MTCNN on (possibly padded) image
        try:
            with torch.no_grad():
                faces = self.mtcnn(pil_for_mtcnn, return_prob=True)
        except RuntimeError as e:
            msg = str(e)
            # Known MPS adaptive pool issue: fallback to CPU for detection
            if 'Adaptive pool MPS' in msg or 'input sizes must be divisible' in msg or 'Non-divisible input sizes' in msg:
                now = time.time()
                # Throttle warning prints to avoid spamming the terminal
                if now - self._mps_warn_time >= self._mps_warn_interval:
                    print('Warning: MPS adaptive pool issue detected during MTCNN; retrying detection on CPU.')
                    self._mps_warn_time = now

                # create cached CPU mtcnn if needed
                if self._cpu_mtcnn is None:
                    self._cpu_mtcnn = MTCNN(image_size=self.mtcnn.image_size, margin=self.mtcnn.margin, keep_all=self.mtcnn.keep_all, device=torch.device('cpu'))

                # Run a light .detect() on CPU and return boxes+probs (no face batch)
                try:
                    boxes, probs, landmarks = self._cpu_mtcnn.detect(pil, landmarks=True)
                except Exception:
                    return [], [], None, None
                if boxes is None:
                    return [], [], None, None
                return boxes.astype(int).tolist(), probs.tolist(), None, (landmarks.tolist() if landmarks is not None else None)
            else:
                # Other runtime error -> try detect to get boxes only
                try:
                    boxes, probs, landmarks = self.mtcnn.detect(pil_for_mtcnn, landmarks=True)
                    if boxes is None:
                        return [], [], None, None
                    # Remove padding offset if applicable
                    if isinstance(boxes, np.ndarray) and (pad_left or pad_top):
                        boxes = boxes - np.array([pad_left, pad_top, pad_left, pad_top])
                    return boxes.astype(int).tolist(), probs.tolist(), None, (landmarks.tolist() if landmarks is not None else None)
                except Exception:
                    return [], [], None, None
        except Exception:
            # Generic fallback: run detect to get boxes and probs only
            try:
                boxes, probs, landmarks = self.mtcnn.detect(pil_for_mtcnn, landmarks=True)
            except Exception:
                return [], [], None, None
            if boxes is None:
                return [], [], None, None
            if isinstance(boxes, np.ndarray) and (pad_left or pad_top):
                boxes = boxes - np.array([pad_left, pad_top, pad_left, pad_top])
            return boxes.astype(int).tolist(), probs.tolist(), None, (landmarks.tolist() if landmarks is not None else None)

        # faces can be (PIL images or torch tensors, probs)
        if faces is None:
            return [], [], None, None

        # If return is tuple (faces, probs)
        if isinstance(faces, tuple) and len(faces) == 2:
            face_imgs, probs = faces
        else:
            face_imgs = faces
            probs = None

        # Normalize face_imgs into a list for uniform processing
        face_list = []
        if face_imgs is None:
            return [], [], None, None
        elif isinstance(face_imgs, torch.Tensor):
            # tensor can be (N,3,H,W) or (3,H,W)
            if face_imgs.dim() == 4:
                face_list = [face_imgs[i] for i in range(face_imgs.size(0))]
            elif face_imgs.dim() == 3:
                face_list = [face_imgs]
            else:
                face_list = []
        elif isinstance(face_imgs, Image.Image):
            face_list = [face_imgs]
        elif isinstance(face_imgs, (list, tuple)):
            face_list = list(face_imgs)
        else:
            face_list = []

        if not face_list:
            return [], [], None, None

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
                # If tensor is on CPU, leave it; we'll move batch to device later
                face_tensors.append(t)
            else:
                # Unknown type
                continue

        if not face_tensors:
            return [], [], None, None

        face_batch = torch.stack(face_tensors, dim=0).to(self.device)

        # To get boxes and landmarks we can call detect on the PIL again
        try:
            boxes, det_probs, landmarks = self.mtcnn.detect(pil_for_mtcnn, landmarks=True)
            if boxes is None:
                boxes = []
                det_probs = []
                landmarks = None
        except Exception:
            boxes = []
            det_probs = []
            landmarks = None

        # If we padded before running MTCNN, remove the padding offset from boxes and landmarks
        if isinstance(boxes, np.ndarray) and (pad_left or pad_top):
            boxes = boxes - np.array([pad_left, pad_top, pad_left, pad_top])
            if landmarks is not None:
                landmarks = landmarks - np.array([pad_left, pad_top])

        boxes_int = boxes.astype(int).tolist() if len(boxes) else []

        return boxes_int, det_probs.tolist() if len(det_probs) else probs, face_batch, (landmarks.tolist() if landmarks is not None else None)
