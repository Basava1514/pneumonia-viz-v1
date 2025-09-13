# scripts/get_model.py
import os, requests, sys

def download_file(url, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

if __name__ == "__main__":
    url = os.environ.get("MODEL_URL")
    if not url:
        print("Set MODEL_URL env var to the model download link.")
        sys.exit(1)
    out = "models/vit_pneumonia_best.pt"
    download_file(url, out)
    print(f"Downloaded model to {out}")
