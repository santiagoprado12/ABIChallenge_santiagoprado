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

    def __init__(self):

        self.model = self.get_model()
        self.db_manager = PostgreSQLManager()

    def get_prediction(self, request: PredictionRequest) -> float:

        df_request = self._transform_to_dataframe(request)
        preprocessed_data = preprocess_features(df_request)
        prediction = self.model.predict(preprocessed_data)[0]

        self.log_to_db(data_input=df_request, prediction=prediction)

        return max(prediction, 0)

    def get_model(self) -> Pipeline:
        model_path = os.environ.get("MODEL_PATH", "models/best_model.pkl")
        with open(model_path, "rb") as model_file:
            model = load(BytesIO(model_file.read()))
        return model

    def __call__(self, *args, **kwds):
        return self.get_prediction(*args, **kwds)

    def _transform_to_dataframe(self, class_model: BaseModel) -> DataFrame:
        transition_dictionary = {}
        for key, value in class_model.dict().items():
            if key in ("Sex", "Embarked"):
                transition_dictionary[key] = [value.value]
            else:
                transition_dictionary[key] = [value]

        data_frame = DataFrame(transition_dictionary)
        return data_frame

    def log_to_db(self, data_input: DataFrame, prediction: int) -> None:

        data_to_upload = data_input.copy()
        data_to_upload["prediction"] = [prediction]

        self.db_manager.upload_dataframe_to_postgres(
            data_to_upload, table_name="titanic"
        )
