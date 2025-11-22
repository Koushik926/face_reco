#!/usr/bin/env bash
# macOS M4 helper — create conda env (miniforge) and run realtime demo

set -euo pipefail

ENV_NAME=face-rec-m4
PYTHON=python

echo "This script helps create an environment for Mac M4 and run the demo."

if command -v conda >/dev/null 2>&1; then
  echo "Conda detected — creating environment with name $ENV_NAME"
  conda create -y -n $ENV_NAME python=3.11
  echo "Activating $ENV_NAME — please run: conda activate $ENV_NAME"
  echo "Then run:"
  echo "  pip install -r requirements-m4.txt"
  echo "Or follow README.md for MPS-specific PyTorch install instructions."
else
  echo "Conda not detected. Creating venv with python3 - please ensure python3 (3.11) is installed."
  $PYTHON -m venv .venv
  echo "Activate it with: source .venv/bin/activate"
  echo "Then install packages: pip install -r requirements-m4.txt"
fi

echo "After installing dependencies, run:"
echo "  python src/realtime.py --db face_db.pkl"
echo "To capture images for enrollment run:"
echo "  python src/enroll.py --id john_doe --count 10"
