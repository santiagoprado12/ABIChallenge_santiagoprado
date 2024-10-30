from unittest.mock import patch

from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


@patch("src.db.db_manager.postgre_sql_manager.psycopg2.connect")
def test_prediction(mock_connect):
    response = client.post(
        "/v1/prediction",
        json={
            "PassengerId": 0,
            "Pclass": 0,
            "Name": "string",
            "Sex": "male",
            "Age": 0,
            "SibSp": 0,
            "Parch": 0,
            "Ticket": "string",
            "Fare": 0,
            "Cabin": "string",
            "Embarked": "S",
        },
    )
    assert response.status_code == 200
    assert response.json()["Survived"] in (0, 1)


@patch("src.db.db_manager.postgre_sql_manager.psycopg2.connect")
def test_batch_prediction(mock_connect):
    response = client.post(
        "/v1/batch_prediction",
        json={
            "batch_data": [
                {
                    "PassengerId": 0,
                    "Pclass": 0,
                    "Name": "string",
                    "Sex": "male",
                    "Age": 0,
                    "SibSp": 0,
                    "Parch": 0,
                    "Ticket": "string",
                    "Fare": 0,
                    "Cabin": "string",
                    "Embarked": "S",
                }
            ]
        },
    )
    assert response.status_code == 200
    assert isinstance(response.json()["Survived"], list)
    for prediction in response.json()["Survived"]:
        assert prediction in (0, 1)
