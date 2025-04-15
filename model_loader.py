import os
import torch
import requests
from pathlib import Path
from tqdm import tqdm
from transformers import BlipProcessor, BlipForConditionalGeneration

# Define model paths and URLs
YOLO_MODEL_PATH = 'yolov5s.pt'
YOLO_MODEL_URL = 'https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5s.pt'
BLIP_MODEL_ID = "Salesforce/blip-image-captioning-base"

def download_file(url, path):
    """Download a file from a URL to a local path with progress bar."""
    if os.path.exists(path):
        print(f"File already exists at {path}")
        return
    
    print(f"Downloading {url} to {path}")
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    file_size = int(response.headers.get("Content-Length", 0))
    progress_bar = tqdm(total=file_size, unit="B", unit_scale=True)
    
    with open(path, "wb") as file:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                progress_bar.update(len(chunk))
                file.write(chunk)
    
    progress_bar.close()
    print(f"Download completed: {path}")

def load_yolo_model():
    """Load or download YOLOv5 model."""
    if not os.path.exists(YOLO_MODEL_PATH):
        download_file(YOLO_MODEL_URL, YOLO_MODEL_PATH)
    
    try:
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=YOLO_MODEL_PATH)
        return model
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        # Fallback to loading from hub if local file fails
        return torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

def load_blip_model():
    """Load BLIP captioning model."""
    try:
        processor = BlipProcessor.from_pretrained(BLIP_MODEL_ID)
        model = BlipForConditionalGeneration.from_pretrained(BLIP_MODEL_ID)
        return processor, model
    except Exception as e:
        print(f"Error loading BLIP model: {e}")
        raise

def ensure_models_loaded():
    """Main function to ensure all models are loaded."""
    print("Loading models...")
    yolo = load_yolo_model()
    processor, model = load_blip_model()
    print("All models loaded successfully!")
    return yolo, processor, model

if __name__ == "__main__":
    ensure_models_loaded() 