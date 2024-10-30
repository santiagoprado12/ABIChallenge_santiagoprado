from unittest.mock import patch

import pytest

from src.front.titanic_prediction_interface import InputHandler, PredictionAPI

# Mock API base URL for testing
API_BASE_URL = "https://9k6yjdcqae.us-east-1.awsapprunner.com/"


# Test the PredictionAPI class
@pytest.fixture
def prediction_api():
    return PredictionAPI(API_BASE_URL)


def test_get_single_prediction(prediction_api):
    # Mock the requests.post method
    with patch("requests.post") as mock_post:
        # Simulate a successful API response
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {"Survived": 1}

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
        result = prediction_api.get_single_prediction(data)
        assert result["Survived"] == 1


def test_get_batch_predictions(prediction_api):
    # Mock the requests.post method
    with patch("requests.post") as mock_post:
        # Simulate a successful API response
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {"Survived": [1, 0, 1]}

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
        result = prediction_api.get_batch_predictions(data)
        assert result == [1, 0, 1]


def test_get_manual_input():
    # This test can mock the behavior of user input
    input_handler = InputHandler()

    with patch("streamlit.selectbox") as mock_selectbox, patch(
        "streamlit.number_input"
    ) as mock_number_input, patch("streamlit.text_input") as mock_text_input:
        mock_selectbox.side_effect = [1, "male", "S"]
        mock_number_input.side_effect = [30, 0, 0, 30.0]
        mock_text_input.side_effect = ["Test", "A1", "12345"]

        data = input_handler.get_manual_input()
        assert data["Sex"] == "male"
        assert data["Age"] == 30
        assert data["Cabin"] == "A1"
        assert data["Ticket"] == "12345"
