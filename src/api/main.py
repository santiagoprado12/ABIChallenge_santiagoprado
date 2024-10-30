from fastapi import FastAPI

from src.api.app.batch_predictor import BatchPredictor
from src.api.app.models import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    PredictionRequest,
    PredictionResponse,
)
from src.api.app.predictor import Predictor

app = FastAPI(
    docs_url="/",
    title="Titanic Predictions",
    description="Inference endpoint for model trained on Titanic dataset.",
    version="1.0.0",
)


@app.post("/v1/prediction")
def predict(request: PredictionRequest):
    return PredictionResponse(Survived=Predictor()(request))


@app.post("/v1/batch_prediction")
def batch_predict(request: BatchPredictionRequest):
    return BatchPredictionResponse(Survived=BatchPredictor()(request))
