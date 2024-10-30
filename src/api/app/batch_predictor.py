from src.api.app.models import BatchPredictionRequest
from src.api.app.predictor import Predictor


class BatchPredictor(Predictor):

    def __init__(self):
        super().__init__()

    def batch_predictor(self, request: BatchPredictionRequest):

        predictions = []
        for item in request.batch_data:
            predictions.append(self.get_prediction(item))

        return predictions

    def __call__(self, *args, **kwds):
        return self.batch_predictor(*args, **kwds)
