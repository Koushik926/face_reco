"""Quick sanity checks for MTCNN and InceptionResnet on a sample image.

This script is not a full unit test but verifies both models can be loaded
and a forward pass completes without error. Useful when setting up MPS.
"""
import os
import sys
import numpy as np
from pathlib import Path

try:
    import torch
    from facenet_pytorch import MTCNN, InceptionResnetV1
    from PIL import Image
except Exception as e:
    print('Missing packages:', e)
    sys.exit(1)

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))


def main():
    device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
    print('Device:', device)

    mtcnn = MTCNN(image_size=160, margin=20, device=device)
    resnet = InceptionResnetV1(pretrained='vggface2').to(device).eval()

    sample = Path(__file__).parent.parent / 'campus_faces'
    # find any image
    img_path = None
    for p in sample.rglob('*'):
        if p.suffix.lower() in ('.jpg', '.jpeg', '.png'):
            img_path = p
            break

    if img_path is None:
        print('No sample images found in campus_faces. Please add images to run this test.')
        return

    img = Image.open(img_path).convert('RGB')
    with torch.no_grad():
        faces = mtcnn(img, return_prob=False)
        if faces is None:
            print('MTCNN found no faces in sample image')
        else:
            print('MTCNN returned', len(faces), 'face crops')
            # convert first to tensor if PIL
            f = faces[0]
            if not isinstance(f, torch.Tensor):
                f = torch.from_numpy(np.asarray(f)).permute(2, 0, 1).float() / 255.0
                f = (f - 0.5) / 0.5
            f = f.to(device).unsqueeze(0)
            emb = resnet(f)
            print('Embedding shape:', emb.shape)


if __name__ == '__main__':
    main()
