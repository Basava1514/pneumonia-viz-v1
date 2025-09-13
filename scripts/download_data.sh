#!/bin/bash
set -e
mkdir -p data
if ! command -v kaggle >/dev/null 2>&1; then
  echo "Please install kaggle CLI and place kaggle.json in ~/.kaggle/"
  exit 1
fi
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia -p data/ --unzip
echo "Dataset downloaded to ./data"
