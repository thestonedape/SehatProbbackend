import os
import gc
import logging


os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
from PIL import Image
import pickle
import io
from typing import List, Optional


tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PredictionResult(BaseModel):
    class_name: str
    confidence: float

class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    all_predictions: List[PredictionResult]

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    available_classes: List[str]

class RootResponse(BaseModel):
    message: str
    endpoints: dict

app = FastAPI(
    title="Skin Disease Classification API",
    description="API for classifying skin diseases using ensemble of EfficientNetV2 and VGG16 models",
    version="2.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


model_efficientnet: Optional[tf.keras.Model] = None
model_vgg16: Optional[tf.keras.Model] = None
class_index: Optional[dict] = None
index_class: Optional[dict] = None

@app.on_event("startup")
async def load_model_and_mappings():
    """Load both models and class mappings on startup"""
    global model_efficientnet, model_vgg16, class_index, index_class
    
    try:
        logger.info("Starting model loading...")
        
        # Load EfficientNet model
        model_efficientnet = tf.keras.models.load_model(
            'resultsskinwise/efficientnet_final.keras',
            compile=False  
        )
        model_efficientnet.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        logger.info("✅ EfficientNet model loaded successfully!")
        
        # Load VGG16 model
        try:
            model_vgg16 = tf.keras.models.load_model(
                'final_model2.keras',
                compile=False  
            )
            model_vgg16.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            logger.info("✅ VGG16 model loaded successfully!")
        except Exception as vgg_error:
            logger.warning(f"⚠️ VGG16 model not available: {str(vgg_error)}. Using EfficientNet only.")
            model_vgg16 = None

        # Load class mappings
        try:
            with open('resultsskinwise/class_index.pkl', 'rb') as f:
                class_index = pickle.load(f)
            
            index_class = {v: k for k, v in class_index.items()}
            
            logger.info(f"✅ Class mappings loaded! Available classes: {list(class_index.keys())}")
            
        except FileNotFoundError:
            logger.error("❌ class_index.pkl not found!")
            raise HTTPException(status_code=500, detail="Class index file not found")
      
        gc.collect()
        
    except Exception as e:
        logger.error(f"❌ Error loading models or mappings: {str(e)}")
        model_efficientnet = None
        model_vgg16 = None
        class_index = None
        index_class = None

def preprocess_image(image_file, model_type='efficientnet'):
    """Preprocess image for specific model type"""
    try:
        image = Image.open(io.BytesIO(image_file)).convert('RGB')
        image = image.resize((224, 224))
        img_array = np.array(image, dtype=np.float32)
        img_batch = np.expand_dims(img_array, axis=0)
        
        if model_type == 'efficientnet':
            img_preprocessed = tf.keras.applications.efficientnet_v2.preprocess_input(img_batch)
        else:  # vgg16
            img_preprocessed = tf.keras.applications.vgg16.preprocess_input(img_batch)
        
        return img_preprocessed
        
    except Exception as e:
        logger.error(f"Image preprocessing error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

@app.post("/predict", response_model=PredictionResponse)
async def predict_skin_disease(file: UploadFile = File(...)) -> PredictionResponse:
    """
    Ensemble prediction combining EfficientNet and VGG16 models
    Returns top 3 predictions with averaged confidence scores
    """
    
    if model_efficientnet is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please try again later.")
    
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    if file.size and file.size > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large. Maximum size is 10MB")
    
    try:
        image_data = await file.read()
        
        # Get predictions from EfficientNet
        preprocessed_efficient = preprocess_image(image_data, 'efficientnet')
        predictions_efficient = model_efficientnet.predict(preprocessed_efficient, verbose=0)[0]
        
        # Ensemble: Average predictions if VGG16 is available
        if model_vgg16 is not None:
            preprocessed_vgg = preprocess_image(image_data, 'vgg16')
            predictions_vgg = model_vgg16.predict(preprocessed_vgg, verbose=0)[0]
            # Average the predictions from both models
            predictions = (predictions_efficient + predictions_vgg) / 2.0
            logger.info("Using ensemble of EfficientNet + VGG16")
        else:
            predictions = predictions_efficient
            logger.info("Using EfficientNet only")
        
        # Get top prediction
        predicted_index = np.argmax(predictions)
        predicted_class = index_class[predicted_index]
        confidence = float(np.max(predictions))
  
        # Get top 3 predictions
        top_3_indices = np.argsort(predictions)[-3:][::-1]
        top_3_predictions = [
            PredictionResult(
                class_name=index_class[idx],
                confidence=float(predictions[idx])
            )
            for idx in top_3_indices
        ]
        
        logger.info(f"Prediction made: {predicted_class} with confidence {confidence:.4f}")
        
        return PredictionResponse(
            predicted_class=predicted_class,
            confidence=confidence,
            all_predictions=top_3_predictions
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint"""
    models_loaded = model_efficientnet is not None
    return HealthResponse(
        status="healthy" if models_loaded else "model_not_loaded",
        model_loaded=models_loaded,
        available_classes=list(class_index.keys()) if class_index else []
    )

@app.get("/", response_model=RootResponse)
async def root() -> RootResponse:
    """Root endpoint with API info"""
    return RootResponse(
        message="Skin Disease Classification API",
        endpoints={
            "predict": "/predict (POST) - Upload image for prediction",
            "health": "/health (GET) - Check API health",
            "docs": "/docs (GET) - API documentation"
        }
    )

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up models on shutdown"""
    global model_efficientnet, model_vgg16
    if model_efficientnet is not None:
        del model_efficientnet
        model_efficientnet = None
    if model_vgg16 is not None:
        del model_vgg16
        model_vgg16 = None
    gc.collect()
    logger.info("Application shutdown complete")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=port,
        log_level="info"
    )