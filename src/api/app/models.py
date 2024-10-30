"""Module with data models."""

from enum import Enum

from pydantic import BaseModel


class Sex(Enum):
    """Data model for sex type."""

    male = "male"
    female = "female"


class Embarked(Enum):
    S = "S"
    C = "C"
    Q = "Q"


class PredictionRequest(BaseModel):
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
    batch_data: list[PredictionRequest]


class PredictionResponse(BaseModel):
    Survived: int


class BatchPredictionResponse(BaseModel):
    Survived: list[int]
