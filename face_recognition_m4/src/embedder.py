"""Embedding model loader and helper functions.

Loads InceptionResnetV1 by default (vggface2 pretrained). Provides a
method to compute L2-normalized embeddings for batches of face tensors.
"""
from typing import Optional
import sys
from pathlib import Path

# Ensure local src imports work when running this file directly
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
from facenet_pytorch import InceptionResnetV1
import numpy as np


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class Embedder:
    """Load face embedding model and compute embeddings.

    Usage:
        e = Embedder(device=device)
        embeddings = e.embed(face_tensors)  # face_tensors: torch.Tensor Nx3x160x160
    """

    def __init__(self, model_name: str = 'resnet', device: Optional[torch.device] = None):
        """model_name: 'resnet' (InceptionResnetV1) or 'mobileface' (lightweight)
        """
        self.device = device or get_device()
        if model_name.lower() in ('resnet', 'inception', 'inceptionresnet', 'inception_resnet_v1'):
            # Default: InceptionResnetV1 pretrained on vggface2
            self.model = InceptionResnetV1(pretrained='vggface2').to(self.device).eval()
            self.out_dim = 512
        elif model_name.lower() in ('mobileface', 'mobilefacenet', 'mobile'):
            # Lightweight MobileFaceNet-like model
            self.model = MobileFaceNet(embedding_size=512).to(self.device).eval()
            self.out_dim = 512
        else:
            raise ValueError(f"Unknown model_name {model_name}")

    def embed(self, face_batch: torch.Tensor) -> np.ndarray:
        """Compute L2-normalized embeddings for a batch of face tensors.

        face_batch: torch.Tensor of shape (N,3,H,W) and values expected in [-1,1]
        Returns:
            numpy array of shape (N,512) float32
        """
        if face_batch is None:
            return np.zeros((0, 512), dtype=np.float32)

        with torch.no_grad():
            x = face_batch.to(self.device)
            emb = self.model(x)
            # L2 normalize
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
            emb_np = emb.cpu().numpy()
        return emb_np



class _ConvBlock(torch.nn.Module):
    def __init__(self, in_ch, out_ch, ks=3, stride=1, padding=1):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_ch, out_ch, ks, stride, padding, bias=False)
        self.bn = torch.nn.BatchNorm2d(out_ch)
        self.act = torch.nn.PReLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class MobileFaceNet(torch.nn.Module):
    """A compact MobileFaceNet-like network for quick embeddings.

    This is intentionally small and not identical to published MobileFaceNet;
    it's provided as a lightweight option for faster inference on MPS.
    Output dimension matches 512 to be compatible with DB expectations.
    """
    def __init__(self, embedding_size: int = 512):
        super().__init__()
        self.features = torch.nn.Sequential(
            _ConvBlock(3, 32, ks=3, stride=1),
            torch.nn.MaxPool2d(2),
            _ConvBlock(32, 64, ks=3, stride=1),
            torch.nn.MaxPool2d(2),
            _ConvBlock(64, 128, ks=3, stride=1),
            torch.nn.MaxPool2d(2),
            _ConvBlock(128, 256, ks=3, stride=1),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(256, 512, bias=False),
            torch.nn.BatchNorm1d(512),
            torch.nn.PReLU(),
            torch.nn.Linear(512, embedding_size, bias=False),
        )

    def forward(self, x):
        # x: (N,3,160,160)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
