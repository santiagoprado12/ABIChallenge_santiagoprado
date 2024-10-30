from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


def test_null_prediction():
    response = client.post("/v1/prediction",
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
                               "Embarked": "S"
                           })
    assert response.status_code == 200
    assert response.json()["Survived"] in (0, 1)