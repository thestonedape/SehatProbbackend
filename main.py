from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
from PIL import Image
import pickle
import io
from typing import List

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

app = FastAPI(title="Skin Disease Classification API")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
class_index = None
index_class = None

@app.on_event("startup")
async def load_model_and_mappings():
    """Load model and class mappings on startup"""
    global model, class_index, index_class
    
    try:
       
        model = tf.keras.models.load_model('final_model2.keras')
        print("✅ Model loaded successfully!")
        
        
        with open('class_index.pkl', 'rb') as f:
            class_index = pickle.load(f)
        
    
        index_class = {v: k for k, v in class_index.items()}
        
        print(f"✅ Class mappings loaded! Available classes: {list(class_index.keys())}")
        
    except Exception as e:
        print(f"❌ Error loading model or mappings: {str(e)}")
        raise e

def preprocess_image(image_file) -> np.ndarray:
    """Preprocess uploaded image for VGG16 model"""
    try:
        
        image = Image.open(io.BytesIO(image_file)).convert('RGB')
        
    
        image = image.resize((224, 224))
        
       
        img_array = np.array(image)
        
    
        img_batch = np.expand_dims(img_array, axis=0)
        
        
        img_preprocessed = tf.keras.applications.vgg16.preprocess_input(img_batch)
        
        return img_preprocessed
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

@app.post("/predict", response_model=PredictionResponse)
async def predict_skin_disease(file: UploadFile = File(...)) -> PredictionResponse:
    
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:

        image_data = await file.read()
        preprocessed_image = preprocess_image(image_data)
        
        predictions = model.predict(preprocessed_image)
        
        predicted_index = np.argmax(predictions[0])
        predicted_class = index_class[predicted_index]
        confidence = float(np.max(predictions[0]))
        
        top_3_indices = np.argsort(predictions[0])[-3:][::-1]
        top_3_predictions = [
            PredictionResult(
                class_name=index_class[idx],
                confidence=float(predictions[0][idx])
            )
            for idx in top_3_indices
        ]
        
        return PredictionResponse(
            predicted_class=predicted_class,
            confidence=confidence,
            all_predictions=top_3_predictions
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        available_classes=list(class_index.keys()) if class_index else []
    )

@app.get("/", response_model=RootResponse)
async def root() -> RootResponse:
    """Root endpoint with API info"""
    return RootResponse(
        message="Skin Disease Classification API",
        endpoints={
            "predict": "/predict (POST) - Upload image for prediction",
            "health": "/health (GET) - Check API health"
        }
    )

if __name__ == "__main__":
    import os, uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
