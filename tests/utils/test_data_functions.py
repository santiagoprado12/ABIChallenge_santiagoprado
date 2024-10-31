"""Module with tests for batch predictor."""

import os
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier

from src.utils.data_functions import *


@patch("pandas.read_csv")
def test_load_data(mock_read_csv, monkeypatch):
    """Test the load_data function to verify correctly read and process CSV file.

    Args:
        mock_read_csv: Mock object for pd.read_csv.
        monkeypatch: pytest fixture to mock attributes in the src.utils.data_functions
            module.
    """
    # Mock the preprocess_data function within data_functions
    monkeypatch.setattr(
        "src.utils.data_functions.preprocess_data", lambda X, tgt: (X, 1)
    )

    # Define the mocked return value for pd.read_csv
    mock_read_csv.return_value = pd.DataFrame(
        {"column1": [1, 2, 3], "column2": ["a", "b", "c"]}
    )

    # Call the function with a dummy path
    path = "dummy.csv"
    X, _ = load_data(path)

    # Assert that pd.read_csv was called with the expected path
    mock_read_csv.assert_called_once_with(path)

    # Assert that the result is a DataFrame with the mocked data
    assert isinstance(X, pd.DataFrame)
    assert X.shape == (3, 2)
    assert list(X.columns) == ["column1", "column2"]


def test_preprocess_data():
    """Test the preprocess_data function.

    Asserts that the resulting DataFrame matches the expected output structure and values.
    """
    # Define the mocked input DataFrame
    input_data = pd.DataFrame(
        {
            "PassengerId": [1, 2, 3],
            "Name": ["John", "Jane", "Alice"],
            "Cabin": ["A123", "B456", "C789"],
            "Ticket": ["T123", "T456", "T789"],
            "SibSp": [0, 1, 2],
            "Parch": [0, 0, 1],
            "target": [0, 1, 0],
        }
    )

    # Define the expected output DataFrame after preprocessing
    expected_output_X = pd.DataFrame(
        {
            "SibSp": [0, 1, 2],
            "Parch": [0, 0, 1],
            "FamilySize": [0, 1, 3],
            "IsAlone": [1, 0, 0],
        }
    )

    expected_output_y = input_data["target"]

    # Call the function with the mocked input data
    X, y = preprocess_data(input_data, target_column="target")

    # Assert that the result matches the expected output
    pd.testing.assert_frame_equal(X, expected_output_X)
    pd.testing.assert_series_equal(y, expected_output_y)


def test_generate_validation_report(tmpdir):
    """Test the generate_validation_report function.

    Args:
        tmpdir: pytest fixture to provide a temporary directory for report generation.

    Asserts that the report file is created at the specified path.
    """
    report = f"{tmpdir}/validation_report.md"

    # Set up a dummy classifier and data
    model = DummyClassifier().fit([1, 0, 1], [1, 1, 0])
    X_train = pd.DataFrame({"column1": [1, 0, 1]})
    X_test = pd.DataFrame({"column1": [1, 1, 0]})

    # Generate the validation report
    generate_validation_report(model, X_train, X_test, report)

    # Assert that the report file exists
    assert os.path.exists(report)
