import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier

# sklearn Pipelines
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from src.ml_pipelines.pipeline_connection import (
    PipelineBuilding,  # Replace with the appropriate import statement
)


def mock_init(self):
    # Mock the __init__() method to do nothing
    pass


def test_columns_after_processing(monkeypatch):

    sample_data = {
        "CategoricalFeature1": ["A", "B", "A", "C", "B"],
        "CategoricalFeature2": ["X", "Y", "Z", "X", "Y"],
    }

    sample_data = pd.DataFrame(
        sample_data, columns=["CategoricalFeature1", "CategoricalFeature2"]
    )

    monkeypatch.setattr(PipelineBuilding, "__init__", mock_init)

    pipeline_build = PipelineBuilding()

    categorical_features = ["CategoricalFeature1", "CategoricalFeature2"]
    result = pipeline_build._columns_after_processing(sample_data, categorical_features)

    expected_columns = [
        "CategoricalFeature10",
        "CategoricalFeature11",
        "CategoricalFeature12",
        "CategoricalFeature20",
        "CategoricalFeature21",
        "CategoricalFeature22",
    ]

    assert result == expected_columns


def test_build_data_processing_pipeline(monkeypatch):

    monkeypatch.setattr(PipelineBuilding, "__init__", mock_init)
    pipeline_build = PipelineBuilding()

    numeric_features = ["Numeric1", "Numeric2"]
    ordinal_attributes = ["Ordinal1"]
    categorical_features = ["Categorical1", "Categorical2"]

    pipeline_result = pipeline_build._build_data_processing_pipeline(
        numeric_features, ordinal_attributes, categorical_features
    )

    assert isinstance(pipeline_result, ColumnTransformer)


def test_build_data_processing_pipeline_returns_col_transformer(monkeypatch):

    numeric_features = ["num1", "num2"]
    ordinal_attributes = ["ord1", "ord2"]
    categorical_features = ["cat1", "cat2"]

    monkeypatch.setattr(PipelineBuilding, "__init__", mock_init)

    pipeline = PipelineBuilding()._build_data_processing_pipeline(
        numeric_features, ordinal_attributes, categorical_features
    )
    assert isinstance(pipeline, ColumnTransformer)


def test_build_data_processing_pipeline_process_all_columns_and_rows(monkeypatch):

    data = {
        "num1": [1, 2, 3, 4, 5],
        "ord1": [1, 0, 1, 0, 1],
        "cat1": ["a", "b", "a", "b", "a"],
    }

    data = pd.DataFrame(data, columns=["num1", "ord1", "cat1"])

    numeric_features = ["num1"]
    ordinal_attributes = ["ord1"]
    categorical_features = ["cat1"]

    monkeypatch.setattr(PipelineBuilding, "__init__", mock_init)

    pipeline = PipelineBuilding()._build_data_processing_pipeline(
        numeric_features, ordinal_attributes, categorical_features
    )

    processed_data = pipeline.fit_transform(data)

    print(processed_data)

    assert (
        processed_data.shape[1] == data.shape[1] + 1
    )  # 1 bcecause cat1 has 2 values possible (one hot encoding)
    assert processed_data.shape[0] == 5


def test_build_processing_and_feature_selection_pipeline_returns_pipeline(monkeypatch):

    numeric_features = ["num1"]
    ordinal_attributes = ["ord1"]
    categorical_features = ["cat1", "cat2"]

    total_columns = numeric_features + ordinal_attributes + categorical_features

    monkeypatch.setattr(PipelineBuilding, "__init__", mock_init)

    mock_pipeline = Pipeline([("", SimpleImputer())])

    pipeline = PipelineBuilding()._build_processing_and_feature_selection_pipeline(
        mock_pipeline, total_columns
    )

    assert isinstance(pipeline, Pipeline)


def test_build_data_processing_pipeline_returns_col_transformer(monkeypatch):

    monkeypatch.setattr(PipelineBuilding, "__init__", mock_init)

    pipeline = PipelineBuilding()

    pipeline.models = [
        {
            "name": "random_forest",
            "model": DummyClassifier(),
            "params": {"strategy": ["uniform", "constant"]},
        },
        {
            "name": "logistic_regression",
            "model": DummyClassifier(),
            "params": {"strategy": ["most_frequent", "stratified"]},
        },
    ]

    test_pipeline = Pipeline([])

    model_pipeline = pipeline._build_create_model_pipeline(test_pipeline)

    assert isinstance(model_pipeline, dict)
    assert len(model_pipeline) == 2
    assert isinstance(model_pipeline["random_forest"], Pipeline)
    assert isinstance(model_pipeline["logistic_regression"], Pipeline)
