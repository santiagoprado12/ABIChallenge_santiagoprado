"""Module with data models for Titanic prediction requests and responses."""

from enum import Enum

from pydantic import BaseModel


class Sex(Enum):
    """Enumeration for passenger sex options."""

    male = "male"
    female = "female"


class Embarked(Enum):
    """Enumeration for embarkation ports on the Titanic."""

    S = "S"
    C = "C"
    Q = "Q"


class PredictionRequest(BaseModel):
    """Data model for an individual prediction request.

    Attributes:
        PassengerId (float): The unique ID of the passenger.
        Pclass (int): The passenger class (1, 2, or 3).
        Name (str): The name of the passenger.
        Sex (Sex): The sex of the passenger, as an enumerated type.
        Age (int): The age of the passenger.
        SibSp (int): Number of siblings/spouses aboard the Titanic.
        Parch (int): Number of parents/children aboard the Titanic.
        Ticket (str): The ticket number.
        Fare (float): The fare paid by the passenger.
        Cabin (str): The cabin assigned to the passenger.
        Embarked (Embarked): The embarkation port of the passenger.
    """

    PassengerId: float
    Pclass: int
    Name: str
    Sex: Sex
    Age: int
    SibSp: int
    Parch: int
    Ticket: str
    Fare: float
    Cabin: str
    Embarked: Embarked


class BatchPredictionRequest(BaseModel):
    """Data model for a batch prediction request.

    Attributes:
        batch_data (list[PredictionRequest]): List of prediction requests for multiple passengers.
    """

    batch_data: list[PredictionRequest]


class PredictionResponse(BaseModel):
    """Data model for a prediction response.

    Attributes:
        Survived (int): Predicted survival outcome (1 if survived, 0 if not).
    """

    Survived: int


class BatchPredictionResponse(BaseModel):
    """Data model for a batch prediction response.

    Attributes:
        Survived (list[int]): List of predicted survival outcomes for each passenger in the batch.
    """

    Survived: list[int]
