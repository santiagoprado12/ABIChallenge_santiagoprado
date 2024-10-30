# Data processing
import numpy as np
import pandas as pd

# Pipelines
from src.ml_pipelines.pipeline_connection import PipelineBuilding

# save model
from joblib import dump


class ModelTraining(PipelineBuilding):

    def __init__(
        self, X: pd.DataFrame, y: pd.DataFrame, atributes_types: dict, models: list
    ) -> None:

        super().__init__(X, y, atributes_types)
        models_pipelines = self.build_full_pipeline()
        self.X = X
        self.y = y
        self.models = {}
        for name in models:
            self.models[name] = models_pipelines[name]

    def train_models(self) -> dict:

        for name in self.models:
            self.models[name].fit(self.X, self.y)

        return self.models

    def generate_scores(self, X_test: pd.DataFrame, y_test: pd.DataFrame) -> dict:

        scores = {}
        for model in self.models:
            scores[model] = self.models[model].score(X_test, y_test)

        return scores

    def best_model(self, X_test: pd.DataFrame, y_test: pd.DataFrame) -> str:

        scores = self.generate_scores(X_test, y_test)
        best_model = max(scores, key=scores.get)

        return best_model

    def save_model(self, model_name: str, path: str) -> None:
        """Save a model to a given path"""

        model = self.models[model_name]

        dump(model, path)
