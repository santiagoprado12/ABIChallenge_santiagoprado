"""Module with API endpoints for Titanic Predictions."""

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
    description="Inference endpoint for a model trained on the Titanic dataset.",
    version="1.0.0",
)


@app.post("/v1/prediction")
def predict(request: PredictionRequest) -> PredictionResponse:
    """Endpoint for predicting the survival of a single Titanic passenger.

    Args:
        request (PredictionRequest): Data for a single prediction request.

    Returns:
        PredictionResponse: A response object with the survival prediction.
    """
    return PredictionResponse(Survived=Predictor()(request))


@app.post("/v1/batch_prediction")
def batch_predict(request: BatchPredictionRequest) -> BatchPredictionResponse:
    """Endpoint for predicting the survival of multiple Titanic passengers.

    Args:
        request (BatchPredictionRequest): Data for a batch prediction request.

    Returns:
        BatchPredictionResponse: A response object containing survival predictions.
    """
    return BatchPredictionResponse(Survived=BatchPredictor()(request))
