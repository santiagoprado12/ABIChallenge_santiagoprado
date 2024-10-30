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
    """Predictor class responsible for handling model loading, input transformation,
    and making predictions based on input data. It also logs the input data
    and predictions into a PostgreSQL database.
    """

    def __init__(self):
        """Initializes the Predictor class by loading the pre-trained model
        and setting up the PostgreSQLManager instance for database operations.
        """
        self.model = self.get_model()
        self.db_manager = PostgreSQLManager()

    def get_prediction(self, request: PredictionRequest) -> float:
        """Takes a PredictionRequest object, transforms it into a DataFrame,
        preprocesses the input data, makes a prediction using the loaded model,
        logs the input and prediction to the database, and returns the prediction.

        Args:
            request (PredictionRequest): A Pydantic model representing the prediction request.

        Returns:
            float: The predicted value, constrained to be non-negative.
        """
        # Transform the request object into a pandas DataFrame.
        df_request = self._transform_to_dataframe(request)

        # Preprocess the input features before feeding them to the model.
        preprocessed_data = preprocess_features(df_request)

        # Get the prediction from the model, take the first (and only) result.
        prediction = self.model.predict(preprocessed_data)[0]

        # Log both the input data and the prediction to the database.
        self.log_to_db(data_input=df_request, prediction=prediction)

        # Ensure the prediction is non-negative, return the prediction.
        return max(prediction, 0)

    def get_model(self) -> Pipeline:
        """Loads the machine learning model from the specified file path. The model is
        expected to be a scikit-learn Pipeline object serialized with joblib.

        Returns:
            Pipeline: The loaded machine learning model.
        """
        # Get model path from environment variable or use a default path.
        model_path = os.environ.get("MODEL_PATH", "models/best_model.pkl")

        # Load the model using joblib from a binary file.
        with open(model_path, "rb") as model_file:
            model = load(BytesIO(model_file.read()))

        return model

    def __call__(self, *args, **kwds):
        """Enables the Predictor instance to be callable, invoking the `get_prediction`
        method directly when the object is called.

        Args:
            *args: Positional arguments for the `get_prediction` method.
            **kwds: Keyword arguments for the `get_prediction` method.

        Returns:
            The result from the `get_prediction` method.
        """
        return self.get_prediction(*args, **kwds)

    def _transform_to_dataframe(self, class_model: BaseModel) -> DataFrame:
        """Converts a Pydantic BaseModel object into a pandas DataFrame. This method
        handles specific transformations for fields like "Sex" and "Embarked" that
        require special handling.

        Args:
            class_model (BaseModel): A Pydantic model representing the input data.

        Returns:
            DataFrame: The transformed data in pandas DataFrame format.
        """
        # Dictionary to store the transformed data
        transition_dictionary = {}

        # Iterate through the class model's dictionary and transform fields as needed
        for key, value in class_model.dict().items():
            if key in (
                "Sex",
                "Embarked",
            ):  # Handle fields that require special value extraction
                transition_dictionary[key] = [value.value]
            else:  # For other fields, directly store the value
                transition_dictionary[key] = [value]

        # Convert the dictionary into a pandas DataFrame and return it.
        data_frame = DataFrame(transition_dictionary)
        return data_frame

    def log_to_db(self, data_input: DataFrame, prediction: int) -> None:
        """Logs the input data along with the model's prediction to the PostgreSQL database.

        Args:
            data_input (DataFrame): The preprocessed input data.
            prediction (int): The model's predicted value for the input data.
        """
        # Copy the input data and append the prediction column
        data_to_upload = data_input.copy()
        data_to_upload["prediction"] = [prediction]

        # Use the PostgreSQLManager to upload the data to the specified table.
        self.db_manager.upload_dataframe_to_postgres(
            data_to_upload, table_name="titanic"
        )
