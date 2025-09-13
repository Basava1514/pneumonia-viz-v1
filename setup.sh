#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "Setting up project at $PROJECT_ROOT"

# prepare folders + metadata
touch feedback_dataset.jsonl
echo "{}" > model_meta.json
mkdir -p reports

# prefer conda if available
if command -v conda >/dev/null 2>&1; then
  echo "Using conda to create environment 'pneumonia-viz'"
  conda env create -f environment.yml || conda env update -f environment.yml --prune
  echo "Activate with: conda activate pneumonia-viz"
else
  echo "Creating Python venv .venv and installing requirements"
  python3 -m venv .venv
  source .venv/bin/activate
  pip install --upgrade pip
  pip install -r requirements.txt
  echo "Activate with: source .venv/bin/activate"
fi

echo "Setup done. Run: streamlit run streamlit_app.py"
