from unittest.mock import patch

import pytest

from src.front.titanic_prediction_interface import InputHandler, PredictionAPI

# Mock API base URL for testing
API_BASE_URL = "https://9k6yjdcqae.us-east-1.awsapprunner.com/"


# Test the PredictionAPI class
@pytest.fixture
def prediction_api():
    """Fixture to initialize an instance of the PredictionAPI class.

    Returns:
        PredictionAPI: An instance of the PredictionAPI class, initialized with the base API URL.
    """
    return PredictionAPI(API_BASE_URL)


def test_get_single_prediction(prediction_api):
    """Test the get_single_prediction method of PredictionAPI.

    This test mocks the requests.post method to simulate a successful API call
    and verifies that the method returns the correct prediction.

    Args:
        prediction_api (PredictionAPI): Instance of the PredictionAPI fixture.
    """
    # Mock the requests.post method to simulate API behavior
    with patch("requests.post") as mock_post:
        # Simulate a successful response from the API
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {"Survived": 1}

        # Input data to be sent to the API for prediction
        data = {
            "PassengerId": 0,
            "Pclass": 1,
            "Name": "Test",
            "Sex": "male",
            "Age": 30,
            "SibSp": 0,
            "Parch": 0,
            "Ticket": "12345",
            "Fare": 30.0,
            "Cabin": "A1",
            "Embarked": "S",
        }

        # Call the get_single_prediction method and check the result
        result = prediction_api.get_single_prediction(data)
        assert (
            result["Survived"] == 1
        )  # Ensure the prediction matches the expected value


def test_get_batch_predictions(prediction_api):
    """Test the get_batch_predictions method of PredictionAPI.

    This test mocks the requests.post method to simulate a successful API call
    and verifies that the method returns the correct batch predictions.

    Args:
        prediction_api (PredictionAPI): Instance of the PredictionAPI fixture.
    """
    # Mock the requests.post method to simulate API behavior
    with patch("requests.post") as mock_post:
        # Simulate a successful response from the API
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {"Survived": [1, 0, 1]}

        # Input data to be sent to the API for batch prediction
        data = [
            {
                "PassengerId": 0,
                "Pclass": 1,
                "Name": "Test",
                "Sex": "male",
                "Age": 30,
                "SibSp": 0,
                "Parch": 0,
                "Ticket": "12345",
                "Fare": 30.0,
                "Cabin": "A1",
                "Embarked": "S",
            },
            {
                "PassengerId": 1,
                "Pclass": 2,
                "Name": "Test 2",
                "Sex": "female",
                "Age": 40,
                "SibSp": 1,
                "Parch": 0,
                "Ticket": "54321",
                "Fare": 20.0,
                "Cabin": "B1",
                "Embarked": "C",
            },
        ]

        # Call the get_batch_predictions method and check the result
        result = prediction_api.get_batch_predictions(data)
        assert result == [
            1,
            0,
            1,
        ]  # Ensure the batch prediction matches the expected values


def test_get_manual_input():
    """Test the get_manual_input method of InputHandler.

    This test mocks Streamlit's input widgets to simulate user input
    and verifies that the method returns the correct input data as a dictionary.
    """
    # Initialize the InputHandler instance
    input_handler = InputHandler()

    # Mock the behavior of Streamlit's input widgets
    with patch("streamlit.selectbox") as mock_selectbox, patch(
        "streamlit.number_input"
    ) as mock_number_input, patch("streamlit.text_input") as mock_text_input:

        # Simulate the user's selection/input for the fields
        mock_selectbox.side_effect = [
            1,
            "male",
            "S",
        ]  # First value for Pclass, second for Sex, third for Embarked
        mock_number_input.side_effect = [30, 0, 0, 30.0]  # Age, SibSp, Parch, Fare
        mock_text_input.side_effect = ["Test", "A1", "12345"]  # Name, Cabin, Ticket

        # Call the get_manual_input method
        data = input_handler.get_manual_input()

        # Assert the correct values are returned in the data dictionary
        assert data["Sex"] == "male"
        assert data["Age"] == 30
        assert data["Cabin"] == "A1"
        assert data["Ticket"] == "12345"
