import pandas as pd
import numpy as np
from src.ml_pipelines.feature_selection import FeatureSelection
from sklearn.model_selection import train_test_split
from unittest.mock import patch

# Create a mock DataFrame
data = {
    #columns with 1 position truncated
    'Parch': [0, 1, 1, 0, 0, 1, 1, 0, 0, 1],
    'Pclass': [1, 0, 1, 0, 0, 1, 1, 0, 0, 1],
    'SibSp': [1, 1, 0, 0, 0, 1, 1, 0, 0, 1],
    'Fare': [1, 1, 1, 1, 0, 1, 1, 0, 0, 1],

    #columns with all 0
    'Age': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'Sex': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'Embarked0': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'Embarked1': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'Embarked2': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'FamilySize0': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'FamilySize1': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'FamilySize2': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
}

# Create a mock target column with very similar values
target = np.array([1, 1, 1, 0, 0, 1, 1, 0, 0, 1])



def mock_train_test_split(*args, **kwargs):
    # Here, you can return the same data for both train and test
    X = args[0]  # Assuming the first argument is the data
    y = args[1]

    return X, y, X, y


def test_feature_selection(monkeypatch):


    mock_df = pd.DataFrame(data)

    monkeypatch.setattr('builtins.input', mock_train_test_split)

    assert mock_train_test_split(mock_df, target) == (mock_df, target, mock_df, target)

    colums = list(data.keys())

    # Instantiate the FeatureSelection class
    feature_selector = FeatureSelection(columns=colums, verbose=True)

    # Fit the feature selector with the mock DataFrame and target
    transformed_df = feature_selector.fit_transform(mock_df, target)

    # Ensure that only columns with data are left
    expected_columns = ['Parch', 'Pclass', 'SibSp', 'Fare']

    assert list(transformed_df.columns) == expected_columns

