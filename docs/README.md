# Pneumonia Detection — ViT + U-Net

A research project for pneumonia detection on chest X-rays using Vision Transformer and U-Net lung segmentation.

## Features
- ViT classifier
- U-Net lung segmentation
- Grad-CAM, ViT-CX, LIME, SHAP explainability
- Streamlit app, PDF export, feedback loop & auto-retraining

## Quickstart
1. Clone:
   git clone https://github.com/Basava1514/pneumonia-viz-v1.git
2. Setup:
   ./setup.sh
3. Run:
   streamlit run streamlit_app.py

## Dataset
Kaggle: paultimothymooney/chest-xray-pneumonia
scripts/download_data.sh fetches it (requires kaggle CLI)

## Notes
- This is a research; not for clinical use.
- Do not commit large model weights; use Hugging Face or Google Drive links.
