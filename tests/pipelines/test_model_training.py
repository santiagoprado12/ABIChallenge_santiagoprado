"""Module with test for model training."""

import os
from io import BytesIO

import pytest
from joblib import load
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from src.ml_pipelines.model_training import *
from src.ml_pipelines.pipeline_connection import *


def mock_init(self):
    """Mock the __init__ method of ModelTraining for testing purposes."""
    pass


def test_train_models(monkeypatch):
    """Test that train_models trains each model correctly and returns a dictionary of models.

    Args:
        monkeypatch: pytest fixture to replace the __init__ method of ModelTraining with a mock.
    """
    monkeypatch.setattr(ModelTraining, "__init__", mock_init)

    model_train = ModelTraining()

    model_train.X = pd.DataFrame(
        {
            "num1": [1, 2, 3, 4, 5],
            "ord1": [1, 3, 4, 5, 6],
        }
    )

    model_train.y = pd.DataFrame({"target": [0, 1, 0, 1, 0]})

    model_train.models = {"model1": DummyClassifier()}

    models = model_train.train_models()

    try:
        models["model1"].predict(model_train.X)
    except Exception as e:
        pytest.fail(f"The model is not trained correctly: {e}")

    assert isinstance(models, dict)
    assert len(models) == 1
    assert isinstance(models["model1"], DummyClassifier)


def test_generate_scores(monkeypatch):
    """Test that generate_scores computes scores for each model and returns a dictionary.

    Args:
        monkeypatch: pytest fixture to replace the __init__ method of ModelTraining with a mock.
    """
    monkeypatch.setattr(ModelTraining, "__init__", mock_init)

    model_train = ModelTraining()

    model_train.X = pd.DataFrame(
        {
            "num1": [1, 2, 3, 4, 5],
            "ord1": [1, 3, 4, 5, 6],
        }
    )

    model_train.y = pd.DataFrame({"target": [1, 1, 1, 1, 0]})

    model_train.models = {"model1": DummyClassifier(strategy="most_frequent")}

    model_train.train_models()

    scores = model_train.generate_scores(model_train.X, model_train.y)

    assert isinstance(scores, dict)
    assert len(scores) == 1
    assert isinstance(scores["model1"], float)


def test_best_model(monkeypatch):
    """Test that best_model identifies the model with the highest score.

    Args:
        monkeypatch: pytest fixture to replace the __init__ method of ModelTraining with a mock.
    """
    monkeypatch.setattr(ModelTraining, "__init__", mock_init)
    monkeypatch.setattr(
        ModelTraining,
        "generate_scores",
        lambda self, x, y: {"model1": 0.3, "model2": 0.9},
    )

    model_train = ModelTraining()
    best_model = model_train.best_model(1, 2)

    assert isinstance(best_model, str)
    assert best_model == "model2"


def test_save_model(monkeypatch):
    """Test that save_model saves a trained model to a file and can be loaded correctly.

    Args:
        monkeypatch: pytest fixture to replace the __init__ method of ModelTraining with a mock.
    """
    monkeypatch.setattr(ModelTraining, "__init__", mock_init)
    model = DummyClassifier().fit([1, 2, 3], [1, 2, 3])

    model_train = ModelTraining()
    model_train.models = {"model1": model}

    model_train.save_model("model1", "model1.pkl")

    assert os.path.exists("model1.pkl")

    with open("model1.pkl", "rb") as f:
        model_loaded = load(BytesIO(f.read()))

    assert isinstance(model_loaded, DummyClassifier)

    os.remove("model1.pkl")
