import os
import json
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tensorflow as tf
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="SignBridge Backend", version="1.0.0")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and preprocessing artifacts
MODEL_PATH = os.getenv("MODEL_PATH", "../SignBridge_Final_Model(1)/signbridge_model.keras")
SCALER_MIN_PATH = os.getenv("SCALER_MIN_PATH", "../SignBridge_Final_Model(1)/scaler_min.npy")
SCALER_SCALE_PATH = os.getenv("SCALER_SCALE_PATH", "../SignBridge_Final_Model(1)/scaler_scale.npy")
LABELS_PATH = os.getenv("LABELS_PATH", "../SignBridge_Final_Model(1)/labels.json")

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    scaler_min = np.load(SCALER_MIN_PATH)
    scaler_scale = np.load(SCALER_SCALE_PATH)
    with open(LABELS_PATH, 'r') as f:
        labels = json.load(f)
    print(f"✓ Model loaded from {MODEL_PATH}")
    print(f"✓ Scaler loaded (min shape: {scaler_min.shape}, scale shape: {scaler_scale.shape})")
    print(f"✓ Labels loaded: {len(labels)} classes")
except Exception as e:
    print(f"✗ Error loading model artifacts: {e}")
    raise

class LandmarkInput(BaseModel):
    """Input: 63 landmark features (21 landmarks × 3 coordinates)"""
    landmarks: list[float]
    
class PredictionOutput(BaseModel):
    """Output: predicted class, confidence, and all class probabilities"""
    predicted_class: str
    predicted_index: int
    confidence: float
    all_predictions: dict[str, float]

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "ok", "model_loaded": True}

@app.get("/model-info")
def get_model_info():
    """Returns supported classes and model metadata"""
    return {
        "model_name": "SignBridge",
        "version": "2.0.0",
        "input_features": 63,
        "num_classes": len(labels),
        "supported_classes": sorted([labels[str(i)] for i in range(len(labels))]),
        "test_accuracy": 0.8994,
        "macro_f1": 0.8719,
    }

@app.post("/predict", response_model=PredictionOutput)
def predict(input_data: LandmarkInput):
    """Predict ISL sign from 63 landmark features"""
    try:
        # Validate input
        if len(input_data.landmarks) != 63:
            raise HTTPException(status_code=400, detail=f"Expected 63 features, got {len(input_data.landmarks)}")
        
        # Convert to numpy array
        landmarks_array = np.array(input_data.landmarks, dtype=np.float32).reshape(1, 63)
        
        # Apply MinMaxScaler (same as training)
        scaled_landmarks = (landmarks_array - scaler_min) / scaler_scale
        
        # Make prediction
        predictions = model.predict(scaled_landmarks, verbose=0)
        predicted_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_idx])
        predicted_label = labels[str(predicted_idx)]
        
        # Build all predictions dict
        all_preds = {labels[str(i)]: float(predictions[0][i]) for i in range(len(labels))}
        
        return PredictionOutput(
            predicted_class=predicted_label,
            predicted_index=predicted_idx,
            confidence=confidence,
            all_predictions=all_preds,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch-predict")
def batch_predict(inputs: list[LandmarkInput]):
    """Batch predict multiple landmark sets"""
    results = []
    for inp in inputs:
        result = predict(inp)
        results.append(result)
    return {"predictions": results}

if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host=host, port=port)
