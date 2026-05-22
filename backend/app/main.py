from fastapi import FastAPI, UploadFile, File, Form, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil
import uuid
from typing import List
from pathlib import Path

from app.config import DATASET_DIR, MODEL_PATH
from app.ml.model import TeachableMachineEngine
from app.ml.utils import preprocess_image_bytes

app = FastAPI(
    title="Decoupled Teachable Machine Backend",
    description="High-performance transfer learning API powered by PyTorch MobileNetV3 and Scikit-Learn.",
    version="1.0.0"
)

# Enable CORS for frontend cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual domain/IP
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instantiate the machine learning engine
engine = TeachableMachineEngine()

@app.get("/")
async def root():
    return {
        "service": "Teachable Machine API",
        "status": "online",
        "engine_device": engine.feature_extractor.device,
        "is_model_trained": engine.model_data is not None
    }

@app.post("/upload-sample", status_code=status.HTTP_201_CREATED)
async def upload_sample(
    class_name: str = Form(...),
    files: List[UploadFile] = File(...)
):
    """
    Accepts category labels and list of image files. Saves them in custom
    sub-directories named after the class, utilizing UUIDs for collision-free names.
    """
    # Clean the class name to prevent directory traversal
    clean_class_name = "".join(c for c in class_name if c.isalnum() or c in (" ", "-", "_")).strip()
    if not clean_class_name:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid class name. Names must contain alphanumeric characters, spaces, dashes, or underscores."
        )

    # Establish and create the class directory
    class_dir = DATASET_DIR / clean_class_name
    class_dir.mkdir(parents=True, exist_ok=True)

    saved_count = 0
    allowed_extensions = {".jpg", ".jpeg", ".png", ".webp"}

    for file in files:
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in allowed_extensions:
            continue  # Skip unsupported file extensions

        # Create localized, collision-free filename
        unique_filename = f"{uuid.uuid4().hex}{ext}"
        file_path = class_dir / unique_filename

        try:
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            saved_count += 1
        except Exception as e:
            print(f"Error saving file {file.filename}: {e}")

    # Count total images present in this class folder
    total_class_images = len([f for f in os.listdir(class_dir) if os.path.isfile(class_dir / f)])

    return {
        "status": "success",
        "message": f"Successfully uploaded {saved_count} samples for class '{clean_class_name}'.",
        "class_name": clean_class_name,
        "uploaded_count": saved_count,
        "total_class_samples": total_class_images
    }

@app.post("/train")
async def train_model():
    """
    Scans the dataset directory, extracts MobileNetV3 features, trains the Logistic
    Regression top-classifier, and serializes the pipeline weights.
    """
    try:
        result = engine.train(str(DATASET_DIR))
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred during training: {str(e)}"
        )

@app.post("/predict")
async def predict_sample(file: UploadFile = File(...)):
    """
    Parses a single image, pre-processes it through PyTorch, extracts MobileNetV3 features,
    and runs the prediction on our Scikit-Learn top-classifier.
    """
    # Verify a model has been trained and loaded
    if not engine.model_data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No trained model found. Please train a classification model before running inference."
        )

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in {".jpg", ".jpeg", ".png", ".webp"}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported image format. Allowed formats: PNG, JPG, JPEG, WEBP."
        )

    try:
        # Read image bytes and transform into standard preprocessed tensor
        image_bytes = await file.read()
        tensor = preprocess_image_bytes(image_bytes)
        
        # Perform prediction
        prediction_result = engine.predict(tensor)
        return prediction_result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference processing failed: {str(e)}"
        )

@app.get("/status")
async def get_status():
    """
    Provides dynamic diagnostics about the current dataset, sample distribution,
    and system status.
    """
    # Count samples per category
    sample_stats = {}
    if DATASET_DIR.exists():
        for item in os.listdir(DATASET_DIR):
            item_path = DATASET_DIR / item
            if item_path.is_dir():
                file_count = len([f for f in os.listdir(item_path) if os.path.isfile(item_path / f)])
                sample_stats[item] = file_count

    # Reload system model state
    engine.load_model()
    is_trained = engine.model_data is not None
    trained_classes = engine.model_data["classes"] if is_trained else []

    return {
        "is_model_trained": is_trained,
        "trained_classes": trained_classes,
        "sample_distribution": sample_stats,
        "total_classes_defined": len(sample_stats),
        "total_samples_collected": sum(sample_stats.values())
    }

@app.post("/clear")
async def clear_system():
    """
    Resets the machine learning system back to its factory state by clearing all datasets
    and deleting the saved model checkpoint.
    """
    # Remove dataset directories
    if DATASET_DIR.exists():
        for item in os.listdir(DATASET_DIR):
            item_path = DATASET_DIR / item
            try:
                if item_path.is_dir():
                    shutil.rmtree(item_path)
                else:
                    os.remove(item_path)
            except Exception as e:
                print(f"Error deleting path {item_path}: {e}")

    # Remove trained model weight file
    if MODEL_PATH.exists():
        try:
            os.remove(MODEL_PATH)
        except Exception as e:
            print(f"Error deleting model weight file: {e}")

    # Reload engine model
    engine.load_model()

    return {
        "status": "success",
        "message": "System reset completed. All datasets and model checkpoints have been successfully cleared."
    }
