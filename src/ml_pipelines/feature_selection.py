# Data processing
import pandas as pd
import numpy as np

# sklearn Pipelines
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

# models and evaluation
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Data Processing
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance


class FeatureSelection(BaseEstimator, TransformerMixin):
    """Create a selection of features Pipeline based on the permutation importance of a random forest classifier.

    Args:
        columns (list): List of column names.
        verbose (bool, optional): Whether to print the results of the feature selection. Defaults to False.
    """

    def __init__(self, columns: list, verbose=False) -> None:
        """Initialize the FeatureSelection class.

        Args:
            columns (list): List of column names.
            verbose (bool, optional): Whether to print the results of the feature selection. Defaults to False.
        """

        super().__init__()
        self.columns = columns
        self.verbose = verbose
        self.RANDOM_SEED = 42

    def fit(self, X: np.array, y: np.array = None) -> BaseEstimator:
        """Fit the pipeline on the training data.

        Args:
            X (np.array): Array of features.
            y (np.array, optional): Array of labels. Defaults to None.

        Returns:
            BaseEstimator: The fitted FeatureSelection class.
        """

        X_df = self._to_dataframe(X)
        model, X_test, y_test, X_train, y_train = self._train_feature_selection_model(
            X_df, y
        )
        perm_importance = self._calculate_permutation_importance(model, X_test, y_test)
        self.selected_features = self._select_features(perm_importance, X_train, X_test)
        self._test_selected_features(
            self.selected_features, X_train, X_test, y_train, y_test
        )

        return self

    def transform(self, X: np.array) -> pd.DataFrame:
        """Transform the data using the selected features.

        Args:
            X (np.array): Array of features.

        Returns:
            pd.DataFrame: The transformed data.
        """

        X_df = self._to_dataframe(X)
        X_df = X_df[self.selected_features]
        return X_df

    def _to_dataframe(self, X: np.array) -> pd.DataFrame:
        """Private method to convert a numpy array to a pandas DataFrame.

        Args:
            X (np.array): Array of features.

        Returns:
            pd.DataFrame: The data as a pandas DataFrame.
        """

        X_processed = pd.DataFrame(X, columns=self.columns)
        return X_processed

    def _train_feature_selection_model(self, X: pd.DataFrame, y: np.array) -> tuple:
        """Private method to train a random forest classifier to select features.

        Args:
            X (pd.DataFrame): DataFrame of features.
            y (np.array): Array of labels.

        Returns:
            tuple: A tuple containing the model, the test set, the training set, and the labels.
        """

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.33, random_state=self.RANDOM_SEED
        )

        np.random.seed(self.RANDOM_SEED)
        X_train["random_feature"] = np.random.randn(
            X_train.shape[0]
        )  # add a random feature
        X_test["random_feature"] = np.random.randn(
            X_test.shape[0]
        )  # add a random feature

        model = RandomForestClassifier(random_state=self.RANDOM_SEED)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        # Calculate the accuracy of the model
        accuracy = accuracy_score(y_test, y_pred)
        if self.verbose:
            print(f"Model Accuracy: {accuracy:.2f}")

        return model, X_test, y_test, X_train, y_train

    def _calculate_permutation_importance(
        self, model: BaseEstimator, X_test: pd.DataFrame, y_test: np.array
    ) -> np.array:
        """Private method to calculate the permutation importance of the features.

        Args:
            model (BaseEstimator): The model to calculate the permutation importance for.
            X_test (pd.DataFrame): DataFrame of features.
            y_test (np.array): Array of labels.

        Returns:
            np.array: An array of the permutation importance scores.
        """

        perm_importance = permutation_importance(
            model, X_test, y_test, n_repeats=10, random_state=self.RANDOM_SEED
        )

        if self.verbose:
            for i, feature in enumerate(X_test.columns):
                print(f"{feature}: {perm_importance.importances_mean[i]:.4f}")

        return perm_importance.importances_mean

    def _select_features(
        self, perm_importance: np.array, X_train: pd.DataFrame, X_test: pd.DataFrame
    ) -> list:
        """Private method to select features based on the permutation importance scores.

        Args:
            perm_importance (np.array): An array of the permutation importance scores.
            X_train (pd.DataFrame): DataFrame of training features.
            X_test (pd.DataFrame): DataFrame of test features.

        Returns:
            list: A list of the selected features.
        """

        # Get feature importance scores and indices
        feature_scores = perm_importance
        feature_indices = [i for i in range(len(X_test.columns))]

        # Select features with score above the random column score

        random_feature_importance = feature_scores[
            X_train.columns.get_loc("random_feature")
        ]

        selected_features = [
            X_test.columns[i]
            for i in feature_indices
            if feature_scores[i] > random_feature_importance
        ]

        # Print selected features
        if self.verbose:
            print("Selected Features:")
            for feature in selected_features:
                print(feature)

        return selected_features

    def _test_selected_features(
        self, selected_features, X_train, X_test, y_train, y_test
    ):
        """Private method to test the selected features.

        Args:
            selected_features (list): A list of the selected features.
            X_train (pd.DataFrame): DataFrame of training features.
            X_test (pd.DataFrame): DataFrame of test features.
            y_train (np.array): Array of training labels.
            y_test (np.array): Array of test labels.
        """

        # Train a new model using only the selected features
        selected_X_train = X_train[selected_features]
        selected_X_test = X_test[selected_features]

        selected_model = RandomForestClassifier(random_state=42)
        selected_model.fit(selected_X_train, y_train)

        # Make predictions on the test set using the selected model
        selected_y_pred = selected_model.predict(selected_X_test)

        # Calculate the accuracy of the selected model
        selected_accuracy = accuracy_score(y_test, selected_y_pred)
        if self.verbose:
            print(f"Selected Features Model Accuracy: {selected_accuracy:.2f}")
