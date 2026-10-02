"""
Phase 6: Real-time fraud detection API.

Run from inside the api/ folder with:
    uvicorn main:app --reload --port 8000

Then open http://127.0.0.1:8000/docs for interactive Swagger UI to test it,
or POST a transaction to http://127.0.0.1:8000/predict directly.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from schemas import TransactionRequest, PredictionResponse
from predictor import FraudPredictor

app = FastAPI(
    title="Banking Fraud Detection API",
    description="Real-time fraud scoring with explainable risk reasons",
    version="1.0.0",
)

# Allow a local dashboard (Streamlit, etc.) running on a different port to
# call this API from the browser during development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Loaded once at startup, not per-request, so the model and SHAP explainer
# aren't rebuilt on every single API call.
predictor: FraudPredictor | None = None


@app.on_event("startup")
def load_model():
    global predictor
    predictor = FraudPredictor()
    print(f"Loaded model: {predictor.model_name} | threshold: {predictor.threshold}")


@app.get("/")
def root():
    return {
        "service": "Banking Fraud Detection API",
        "status": "running",
        "model_loaded": predictor is not None,
        "docs": "/docs",
    }


@app.get("/health")
def health():
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model": predictor.model_name, "threshold": predictor.threshold}


@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: TransactionRequest):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    try:
        result = predictor.predict(transaction.model_dump())
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")