"""Module with class predictor."""

import os
from io import BytesIO

from joblib import load
from pandas import DataFrame
from pydantic import BaseModel
from sklearn.pipeline import Pipeline

from src.db.db_manager.postgre_sql_manager import PostgreSQLManager
from src.utils.data_functions import preprocess_features

from .models import PredictionRequest


class Predictor:
    """Class responsible for making predictions with the model."""

    def __init__(self):
        """Initialize the Predictor class by loading the model."""
        self.model = self.get_model()
        self.db_manager = PostgreSQLManager()

    def get_prediction(self, request: PredictionRequest) -> float:
        """Generate a survival prediction for a Titanic passenger.

        Args:
            request (PredictionRequest): A Pydantic model with passenger information.

        Returns:
            float: The predicted survival probability, constrained to be non-negative.
        """
        # Transform request into DataFrame and preprocess features
        df_request = self._transform_to_dataframe(request)
        preprocessed_data = preprocess_features(df_request)

        # Make prediction and log it with input data to the database
        prediction = self.model.predict(preprocessed_data)[0]
        self.log_to_db(data_input=df_request, prediction=prediction)

        return max(prediction, 0)

    def get_model(self) -> Pipeline:
        """Load a machine learning model serialized as a joblib file, expected to be a scikit-learn Pipeline.

        Returns:
            Pipeline: The loaded machine learning model.
        """
        model_path = os.environ.get("MODEL_PATH", "models/best_model.pkl")
        with open(model_path, "rb") as model_file:
            model = load(BytesIO(model_file.read()))
        return model

    def __call__(self, *args, **kwds) -> float:
        """Allow direct calling of the Predictor instance to make predictions.

        Args:
            *args: Positional arguments for `get_prediction`.
            **kwds: Keyword arguments for `get_prediction`.

        Returns:
            float: Prediction result from `get_prediction`.
        """
        return self.get_prediction(*args, **kwds)

    def _transform_to_dataframe(self, class_model: BaseModel) -> DataFrame:
        """Convert a Pydantic BaseModel instance into a pandas DataFrame.

        Args:
            class_model (BaseModel): Input data in Pydantic model format.

        Returns:
            DataFrame: Transformed data in DataFrame format.
        """
        transition_dictionary = {}
        for key, value in class_model.dict().items():
            if key in ("Sex", "Embarked"):
                transition_dictionary[key] = [value.value]
            else:
                transition_dictionary[key] = [value]
        return DataFrame(transition_dictionary)

    def log_to_db(self, data_input: DataFrame, prediction: int) -> None:
        """Log the input data and the prediction result to a PostgreSQL database.

        Args:
            data_input (DataFrame): The passenger data used for prediction.
            prediction (int): The model's predicted outcome.
        """
        data_to_upload = data_input.copy()
        data_to_upload["prediction"] = [prediction]
        self.db_manager.upload_dataframe_to_postgres(
            data_to_upload, table_name="titanic"
        )
