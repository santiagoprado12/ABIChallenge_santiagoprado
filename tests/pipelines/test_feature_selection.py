"""Module with tests for feature selection."""

import numpy as np
import pandas as pd

from src.ml_pipelines.feature_selection import FeatureSelection

# Create a mock DataFrame with relevant columns for feature selection testing
data = {
    "Parch": [0, 1, 1, 0, 0, 1, 1, 0, 0, 1],
    "Pclass": [1, 0, 1, 0, 0, 1, 1, 0, 0, 1],
    "SibSp": [1, 1, 0, 0, 0, 1, 1, 0, 0, 1],
    "Fare": [1, 1, 1, 1, 0, 1, 1, 0, 0, 1],
    "Age": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "Sex": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "Embarked0": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "Embarked1": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "Embarked2": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "FamilySize0": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "FamilySize1": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "FamilySize2": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
}

# Mock target column with values for binary classification testing
target = np.array([1, 1, 1, 0, 0, 1, 1, 0, 0, 1])


def mock_train_test_split(*args, **kwargs):
    """Mock train_test_split to return the same data for train and test sets.

    Args:
        *args: Positional arguments, where the first argument is expected to be the data.
        **kwargs: Additional keyword arguments, unused here.

    Returns:
        Tuple of train and test data (X, y) for both train and test sets.
    """
    X = args[0]  # The first argument should be the data
    y = args[1]

    return X, y, X, y


def test_feature_selection(monkeypatch):
    """Test the FeatureSelection class to ensure it removes columns without data variance.

    Args:
        monkeypatch: pytest fixture to mock dependencies.
    """
    # Mock DataFrame based on `data` dictionary
    mock_df = pd.DataFrame(data)

    # Monkeypatch input to use mock_train_test_split
    monkeypatch.setattr("builtins.input", mock_train_test_split)

    # Verify that the mock train-test split returns expected values
    assert mock_train_test_split(mock_df, target) == (mock_df, target, mock_df, target)

    columns = list(data.keys())

    # Instantiate the FeatureSelection class
    feature_selector = FeatureSelection(columns=columns, verbose=True)

    # Fit and transform the data with the feature selector
    transformed_df = feature_selector.fit_transform(mock_df, target)

    # Expected columns are only those with non-zero variance
    expected_columns = ["Parch", "Pclass", "SibSp", "Fare"]

    # Assert that the transformed DataFrame has only the expected columns
    assert list(transformed_df.columns) == expected_columns
