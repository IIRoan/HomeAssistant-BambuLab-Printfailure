# main.py
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import torch
from huggingface_hub import hf_hub_download
from PIL import Image, ImageDraw
import io
import uvicorn
from typing import List, Optional
from pydantic import BaseModel
import tempfile
import os
import logging
from datetime import datetime
import shutil
import hashlib
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI with metadata
app = FastAPI(
    title="3D Print Failure Detection API",
    description="AI-powered 3D printing failure detection system",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create storage directories
UPLOAD_DIR = Path("uploads")
RESULTS_DIR = Path("results")
UPLOAD_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Mount static directories
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
app.mount("/results", StaticFiles(directory="results"), name="results")

# Global model instance
model = None
model_info = {
    "loaded": False,
    "last_loaded": None,
    "inference_count": 0
}

class DetectionResult(BaseModel):
    id: str
    timestamp: str
    original_filename: str
    marked_image_path: str
    detections: List[dict]
    processing_time: float

def load_model():
    try:
        repo_id = "Javiai/3dprintfails-yolo5vs"
        filename = "model_torch.pt"
        model_path = hf_hub_download(repo_id=repo_id, filename=filename)
        return torch.hub.load('ultralytics/yolov5', 'custom', model_path, verbose=False)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

def process_image(image: Image.Image, original_filename: str):
    start_time = datetime.now()
    
    draw = ImageDraw.Draw(image)
    detections = model(image)
    
    categories = [        
        {'name': 'error', 'color': (0,0,255)},
        {'name': 'extrusor', 'color': (0,255,0)},
        {'name': 'part', 'color': (255,0,0)},
        {'name': 'spaghetti', 'color': (0,0,255)}
    ]
    
    results = []
    for detection in detections.xyxy[0]:
        x1, y1, x2, y2, confidence, category_id = detection
        x1, y1, x2, y2, category_id = int(x1), int(y1), int(x2), int(y2), int(category_id)
        
        category = categories[category_id]['name']
        results.append({
            'category': category,
            'confidence': float(confidence),
            'bbox': [x1, y1, x2, y2]
        })
        
        draw.rectangle((x1, y1, x2, y2), outline=categories[category_id]['color'], width=2)
        draw.text((x1, y1), f"{category} {float(confidence):.2f}", categories[category_id]['color'])
    
    # Generate unique ID for this detection
    detection_id = hashlib.md5(f"{original_filename}{start_time}".encode()).hexdigest()[:10]
    
    # Save marked image
    marked_image_path = RESULTS_DIR / f"marked_{detection_id}_{original_filename}"
    image.save(marked_image_path)
    
    processing_time = (datetime.now() - start_time).total_seconds()
    
    return DetectionResult(
        id=detection_id,
        timestamp=datetime.now().isoformat(),
        original_filename=original_filename,
        marked_image_path=str(marked_image_path),
        detections=results,
        processing_time=processing_time
    )

async def cleanup_old_files(background_tasks: BackgroundTasks):
    """Remove files older than 24 hours"""
    def cleanup():
        current_time = datetime.now()
        for directory in [UPLOAD_DIR, RESULTS_DIR]:
            for file_path in directory.glob("*"):
                if (current_time - datetime.fromtimestamp(file_path.stat().st_mtime)).days >= 1:
                    file_path.unlink()
    
    background_tasks.add_task(cleanup)

@app.on_event("startup")
async def startup_event():
    global model
    logger.info("Loading model...")
    model = load_model()
    model_info["loaded"] = True
    model_info["last_loaded"] = datetime.now().isoformat()
    logger.info("Model loaded successfully")

@app.post("/detect/", response_model=DetectionResult)
async def detect_failures(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    if not model_info["loaded"]:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Save uploaded file
        file_path = UPLOAD_DIR / file.filename
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process image
        image = Image.open(file_path)
        result = process_image(image, file.filename)
        
        # Update stats
        model_info["inference_count"] += 1
        
        # Schedule cleanup
        await cleanup_old_files(background_tasks)
        
        return result
    
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status/")
async def get_status():
    return {
        "model_loaded": model_info["loaded"],
        "last_loaded": model_info["last_loaded"],
        "total_inferences": model_info["inference_count"],
        "upload_dir_size": sum(f.stat().st_size for f in UPLOAD_DIR.glob('**/*')),
        "results_dir_size": sum(f.stat().st_size for f in RESULTS_DIR.glob('**/*'))
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)