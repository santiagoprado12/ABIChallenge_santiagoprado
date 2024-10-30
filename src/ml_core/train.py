# Data processing
import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.ml_pipelines.feature_selection import *
from src.ml_pipelines.model_training import ModelTraining
from src.utils.data_functions import (
    generate_validation_report,
    load_data,
)

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s:%(name)s: %(message)s')
logger = logging.getLogger(__name__)


def train(models_to_use: list, acc_threshold: float = 0.7) -> None:
    """Train the models and save the best one locally.

    Args:
        models_to_use (list, optional): List of models to train.
        acc_threshold (float, optional): Accuracy threshold to consider the model as good. Defaults to 0.7.}

    
    Raises:
        AssertionError: If the best model is not good enough to be saved.
    """
    RANDOM_SEED = 42
    training_report = 'training_report.md'

    atributes_types = {
        'target': 'Survived',
        'ordinal_attributes': ['Sex', 'IsAlone'],
        'numeric_features': ['Parch', 'Pclass', 'SibSp', 'Fare', 'Age'],
        'categorical_features': ['Embarked', 'FamilySize']
    }

    logger.info("Loading data")

    X, y = load_data("data/train.csv")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=RANDOM_SEED)
    
    logger.info("Training models")
    
    model_train = ModelTraining(X_train, y_train, atributes_types, models_to_use)
    models = model_train.train_models()

    scores = model_train.generate_scores(X_test, y_test)
    logger.info(f"Scores: {scores}")

    best_model_name = model_train.best_model(X_test, y_test)
    logger.info(f"Best model: {best_model_name}")

    assert scores[best_model_name] > acc_threshold, "The best model is not good enough, try with different hyperparams"

    logger.info("Saving model locally")
    model_train.save_model(best_model_name, "models/best_model.pkl")

    generate_validation_report(models[best_model_name], X, y, training_report)

    logger.info('Training report generated in %s', training_report)


if __name__ == "__main__":
    train()



