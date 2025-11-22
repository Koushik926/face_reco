# Face Recognition M4 (Apple Silicon optimized)

This repository provides a complete, ready-to-run real-time face recognition
demo optimized for Apple Silicon (M1/M2/M4) using PyTorch MPS backend.

Project layout

face_recognition_m4/
├─ src/
│  ├─ detector.py        # MTCNN detection + alignment (facenet-pytorch)
│  ├─ embedder.py        # Load embedding model (InceptionResnetV1) + helper
│  ├─ database.py        # Build / load face DB, save to disk (pickle)
│  ├─ recognizer.py      # Compare embeddings, threshold logic, identity match
│  ├─ tracker.py         # Simple IOU tracker to reduce detection frequency
│  ├─ realtime.py        # Main real-time loop (OpenCV capture) optimized for MPS
│  └─ utils.py           # Helpers (image conversions, visualization)
├─ campus_faces/         # Example structure with at least one sample (include .gitkeep)
├─ requirements-m4.txt   # Mac M4 specific packages + install hints
├─ run.sh                # Script to setup venv & run realtime (Mac-specific commands)
└─ README.md


Quickstart (MacBook Air M4)
----------------------------

1) Install Miniforge (recommended)

   - Download and install Miniforge (https://github.com/conda-forge/miniforge/releases).
   - Open a new terminal and run:

     conda create -n face-rec-m4 python=3.11 -y
     conda activate face-rec-m4

2) Install PyTorch with MPS support

   The PyTorch MPS backend requires a build that supports MPS. Two common methods:

   A) Conda (recommended):

     conda install pytorch torchvision -c pytorch -c conda-forge

   B) Pip (when conda is not available):

     # Find the latest MPS-enabled wheel at https://download.pytorch.org
     python -m pip install --upgrade pip
     python -m pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu

   Note: Pip wheels sometimes lag. If you have trouble, prefer Miniforge/conda.

3) Install other packages

   pip install -r requirements-m4.txt

   Notes:
   - On macOS, installing `opencv-python` via pip may be problematic. If you see
     errors, install OpenCV via Homebrew and then the headless pip wheel or use conda:

       brew install opencv
       # then (if needed) pip install opencv-python-headless

     Or with conda:

       conda install -c conda-forge opencv

4) Prepare face database (enrollment)

   - Add people folders to `campus_faces/` with structure:

       campus_faces/john_doe/1.jpg
       campus_faces/john_doe/2.jpg

   - Each folder name will be used as person_id. Filenames and number of images
     per person can vary. Aim for 3–10 images per person with frontal faces.

   - Capture images with the enrollment helper (optional):

     python src/enroll.py --id john_doe --count 10

   - Build DB:

     python src/database.py --build --faces-dir campus_faces --out face_db.pkl

     This will create `face_db.pkl` in the repo root containing averaged embeddings.

5) Run realtime demo

  python src/realtime.py --db face_db.pkl --device mps --detect-every 3 --threshold 0.6

  Optional flags:
  - `--embedder mobileface` to use the lightweight MobileFaceNet-like model (faster, lower accuracy)
  - `--width 480` to reduce detection image size and increase FPS

   Controls:
   - Press `q` to quit.


Tuning & performance tips
-------------------------

- Default settings aim to give ~8–20 FPS on MacBook Air M4 depending on face count.
- To push FPS higher:
  - Increase `--detect-every` to reduce expensive MTCNN calls (use tracker to fill gaps).
  - Reduce `--width` default 640 to 480 or 320.
  - Reduce model size (optional): use a MobileFaceNet variant if you add it.
  - Prefer `conda` installs for OpenCV and PyTorch to avoid CPU-only wheels.

Threshold guidance
------------------
- The code uses cosine similarity. The default `--threshold 0.6` is a starting point
  for InceptionResnet embeddings. Lower the threshold to be more permissive (more
  false positives), raise to be stricter.


Troubleshooting
---------------

- "torch.backends.mps is not available": Ensure you installed a PyTorch build with
  MPS support (via conda/pytorch builds). Run `python -c "import torch; print(torch.backends.mps.is_available())"`.
- OpenCV errors on import: consider installing OpenCV via Homebrew or conda-forge.
- CPU-only PyTorch: you will still run, but slower. Re-install with an MPS-enabled build.


Files of interest
-----------------
- `src/detector.py` — MTCNN wrapper optimized to keep tensors on the device.
- `src/embedder.py` — Loads InceptionResnetV1 and returns L2-normalized embeddings.
- `src/database.py` — Build and save the database file `face_db.pkl`.
- `src/realtime.py` — Main demo loop with CLI flags and CSV logging.


Checklist (commands to run on your MacBook Air M4)
--------------------------------------------------
1) Install Miniforge and create env:
   conda create -n face-rec-m4 python=3.11 -y
   conda activate face-rec-m4

2) Install PyTorch (conda recommended):
   conda install pytorch torchvision -c pytorch -c conda-forge

3) Install other dependencies:
   pip install -r requirements-m4.txt

4) Add people images into `campus_faces/`.

5) Build DB:
   python src/database.py --build --faces-dir campus_faces --out face_db.pkl

6) Run realtime demo:
   python src/realtime.py --db face_db.pkl --device mps --detect-every 3

If something fails, check the Troubleshooting section above and re-run `src/test_models.py`.
