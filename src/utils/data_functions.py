from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import load
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline


def generate_validation_report(
    model, X_test, y_test, output_file="validation_report.md"
):
    """Generate a validation report for a binary classification model.

    Parameters:
        model (sklearn.base.BaseEstimator): The trained binary classification model.
        X (numpy.ndarray or pandas.DataFrame): Feature matrix.
        y (numpy.ndarray or pandas.Series): Target vector.
        output_file (str): Output file name for the Markdown report.

    Returns:
        None
    """
    y_pred = model.predict(X_test)

    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    conf_matrix_df = pd.DataFrame(
        conf_matrix,
        index=["Actual Dead", "Actual Survive"],
        columns=["Predicted Dead", "Predicted Survive"],
    )

    plt.figure(figsize=(8, 6))
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks(np.arange(2), ["Dead", "Survived"])
    plt.yticks(np.arange(2), ["Dead", "Survived"])
    plt.imshow(conf_matrix, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar()

    datetime = pd.Timestamp.now().strftime(format="%Y%m%d_%H%M%S")

    conf_matrix_plot_file = f"confusion_matrix_{datetime}.png"
    plt.savefig(conf_matrix_plot_file)
    plt.close()

    markdown_content = f"""
# Validation Report

## Confusion Matrix

![Confusion Matrix](./{conf_matrix_plot_file})

{conf_matrix_df.to_markdown()}

## Classification Report

```bash
{class_report}
```
    """

    with open(output_file, "w") as f:
        f.write(markdown_content)

    print(f"Validation report written to {output_file}")


def preprocess_features(data: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the features.

    Its importat that the data has those columns:
        - Cabin
        - PassengerId
        - Name
        - Ticket
        - SibSp
        - Parch.
    """
    titanic_data = data.copy()
    titanic_data.drop(["Cabin", "PassengerId", "Name", "Ticket"], axis=1, inplace=True)
    titanic_data["FamilySize"] = titanic_data["SibSp"] + titanic_data["Parch"]
    titanic_data["IsAlone"] = 0
    titanic_data.loc[titanic_data["FamilySize"] == 0, "IsAlone"] = 1

    return titanic_data


def preprocess_data(data: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """Preprocess the data.

    Args:
        data (pd.DataFrame): Data to preprocess.
        target_column (str): Name of the target column.

    Returns:
        pd.DataFrame: Preprocessed data.
    """
    titanic_data = preprocess_features(data)

    X = titanic_data.drop(target_column, axis=1)
    y = titanic_data[target_column]

    return X, y


def load_data(path: str) -> pd.DataFrame:
    """Load data from a CSV file.

    It is important that the data has those columns:
        - Survived.
    """
    df = pd.read_csv(path)
    X, y = preprocess_data(df, "Survived")

    return X, y


def load_model(path: str) -> Pipeline:
    """Load model from a pickle file.

    Args:
        path (str): Path to the pickle file.

    Returns:
        Pipeline: The model.
    """
    with open(path, "rb") as f:
        model = load(BytesIO(f.read()))
    return model
