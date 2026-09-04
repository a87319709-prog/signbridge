# SignBridge Backend

FastAPI server for ISL hand sign recognition using the trained SignBridge model.

## Setup

### 1. Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Configure Environment

Copy `.env.example` to `.env` and verify paths:

```bash
cp .env.example .env
```

Update paths in `.env` if needed. The default paths assume the model is in `SignBridge_Final_Model(1)/` at the repository root.

### 3. Run the Server

```bash
python app.py
```

The API will be available at `http://localhost:8000`.

## API Endpoints

### Health Check

```bash
GET /health
```

Response:
```json
{"status": "ok", "model_loaded": true}
```

### Model Info

```bash
GET /model-info
```

Returns supported classes and model metadata.

### Predict (Single)

```bash
POST /predict
Content-Type: application/json

{
  "landmarks": [x1, y1, z1, x2, y2, z2, ..., x21, y21, z21]
}
```

Response:
```json
{
  "predicted_class": "a",
  "predicted_index": 10,
  "confidence": 0.95,
  "all_predictions": {
    "0": 0.001,
    "1": 0.002,
    ...,
    "a": 0.95
  }
}
```

### Predict (Batch)

```bash
POST /batch-predict
Content-Type: application/json

[
  {"landmarks": [...]},
  {"landmarks": [...]}
]
```

## Preprocessing Notes

- Input: 63 features = 21 landmarks × 3 coordinates (x, y, z)
- Flatten order: x1, y1, z1, x2, y2, z2, ..., x21, y21, z21
- **Critical:** Landmarks are normalized using the fitted MinMaxScaler (scaler_min.npy, scaler_scale.npy)
- Do NOT refit the scaler; use the provided artifacts

## Testing the Backend

Use the provided test script:

```bash
python test_backend.py
```
