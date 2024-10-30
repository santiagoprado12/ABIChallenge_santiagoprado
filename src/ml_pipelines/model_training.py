"""Module with class to train the model."""

# Data processing
import numpy as np
import pandas as pd

# Save model
from joblib import dump

# Pipelines
from src.ml_pipelines.pipeline_connection import PipelineBuilding


class ModelTraining(PipelineBuilding):
    """ModelTraining class for building pipelines.

    Inherits:
        PipelineBuilding: Builds full pipelines from the provided attribute types.

    Args:
        X (pd.DataFrame): The training data.
        y (pd.DataFrame): The target labels.
        atributes_types (dict): A dictionary containing the types of attributes for pipeline building.
        models (list): List of model names to be trained.
    """

    def __init__(
        self, X: pd.DataFrame, y: pd.DataFrame, atributes_types: dict, models: list
    ) -> None:
        """Initializes the ModelTraining class.

        Args:
            X (pd.DataFrame): The training feature data.
            y (pd.DataFrame): The target labels.
            atributes_types (dict): Attribute types for constructing pipelines.
            models (list): List of model names to be trained.
        """
        super().__init__(X, y, atributes_types)
        models_pipelines = self.build_full_pipeline()
        self.X = X
        self.y = y
        self.models = {}
        for name in models:
            self.models[name] = models_pipelines[name]

    def train_models(self) -> dict:
        """Trains the models specified during initialization.

        Returns:
            dict: A dictionary where the keys are model names and the values are the trained models.
        """
        for name in self.models:
            self.models[name].fit(self.X, self.y)

        return self.models

    def generate_scores(self, X_test: pd.DataFrame, y_test: pd.DataFrame) -> dict:
        """Generates performance scores for each trained model using the test dataset.

        Args:
            X_test (pd.DataFrame): The test feature data.
            y_test (pd.DataFrame): The test target labels.

        Returns:
            dict: A dictionary with model names as keys and their corresponding test scores as values.
        """
        scores = {}
        for model in self.models:
            scores[model] = self.models[model].score(X_test, y_test)

        return scores

    def best_model(self, X_test: pd.DataFrame, y_test: pd.DataFrame) -> str:
        """Identifies the best-performing model based on the test dataset.

        Args:
            X_test (pd.DataFrame): The test feature data.
            y_test (pd.DataFrame): The test target labels.

        Returns:
            str: The name of the best-performing model.
        """
        scores = self.generate_scores(X_test, y_test)
        best_model = max(scores, key=scores.get)

        return best_model

    def save_model(self, model_name: str, path: str) -> None:
        """Saves a trained model to the specified path.

        Args:
            model_name (str): The name of the model to be saved.
            path (str): The file path where the model should be saved.
        """
        model = self.models[model_name]
        dump(model, path)
