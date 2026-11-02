#!/bin/sh
# Build step for the ai-service Appwrite Function. Runs from the function root (the repo root).
# Mirrors the Dockerfile builder stage: CPU-only torch first, then the locked requirements.
set -eu

# System libraries the Dockerfile installs. python-ml-3.11 is Debian-based and often has
# them already, so this step is best-effort. The import check at the end is what fails the build.
if command -v apt-get >/dev/null 2>&1; then
  if ! { apt-get update -qq && apt-get install -y -qq --no-install-recommends libglib2.0-0 libgomp1; } >/dev/null 2>&1; then
    echo "note: could not install libglib2.0-0 / libgomp1 with apt-get; using preinstalled copies" >&2
  fi
fi

python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch torchvision
python -m pip install --no-cache-dir -r requirements.lock.txt

# Same import check as the Dockerfile builder stage.
python -c "import torch, torchvision, docling, rapidocr, onnxruntime, lightgbm, cv2; print('imports ok')"
