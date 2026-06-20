#!/bin/bash
# Setup YOLOv12 environment for benchmarking

set -e

VENV_DIR="/home/ec2-user/ivana/venv_yolo"

echo "=== Creating virtual environment ==="
python3.12 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

echo "=== Upgrading pip ==="
pip install --upgrade pip

echo "=== Installing YOLOv12 (ultralytics fork) ==="
# Install the YOLOv12 fork which includes yolov12m/l weights support
pip install git+https://github.com/sunsmarterjie/yolov12.git

echo "=== Installing benchmark dependencies ==="
pip install psutil pillow opencv-python-headless

echo ""
echo "=== Setup complete ==="
echo "Activate with: source $VENV_DIR/bin/activate"
echo "Then run:      python3 benchmark_yolo.py"
