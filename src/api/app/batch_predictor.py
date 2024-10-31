"""Module with a class to handle batch predictions for Titanic data."""

from src.api.app.models import BatchPredictionRequest
from src.api.app.predictor import Predictor


class BatchPredictor(Predictor):
    """A predictor class to handle batch predictions by extending the base Predictor class."""

    def __init__(self):
        """Initialize the BatchPredictor by inheriting from the base Predictor class."""
        super().__init__()

    def batch_predictor(self, request: BatchPredictionRequest) -> list[int]:
        """Generate survival predictions for a batch of Titanic passengers.

        Args:
            request (BatchPredictionRequest): Request object containing batch data.

        Returns:
            list[int]: List of survival predictions for each passenger in the batch.
        """
        predictions = []
        for item in request.batch_data:
            predictions.append(self.get_prediction(item))

        return predictions

    def __call__(self, *args, **kwds) -> list[int]:
        """Allow the BatchPredictor instance to be called directly for batch predictions.

        Args:
            *args: Positional arguments for the batch prediction function.
            **kwds: Keyword arguments for the batch prediction function.

        Returns:
            list[int]: List of survival predictions for each passenger in the batch.
        """
        return self.batch_predictor(*args, **kwds)
