# Data processing
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer

# models
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier

# sklearn Pipelines
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

# configs
from src.configs import Configs

# custom pipelines
from src.ml_pipelines.feature_selection import FeatureSelection


class PipelineBuilding(Configs):
    """Builds the full pipeline for the models to be trained.

    Args:
        X (pd.DataFrame): Dataframe with the features.
        y (pd.DataFrame): Dataframe with the target.
        atributes_types (dict): Dictionary with the types of the features.
    """

    def __init__(self, X: pd.DataFrame, y: pd.DataFrame, atributes_types: dict) -> None:
        """Initializes the class.

        Args:
            X (pd.DataFrame): Dataframe with the features.
            y (pd.DataFrame): Dataframe with the target.
            atributes_types (dict): Dictionary with the types of the features.
        """
        self.X = X
        self.y = y
        self.atributes_types = atributes_types
        cat_columns = self._columns_after_processing(
            X, atributes_types["categorical_features"]
        )
        self.all_columns = (
            atributes_types["numeric_features"]
            + atributes_types["ordinal_attributes"]
            + cat_columns
        )

    def _columns_after_processing(
        self, titanic_data: pd.DataFrame, categorical_features: list
    ) -> list:
        """After the one hot encoding, the categorical features are transformed into multiple columns.

        This function returns the names of the new columns.

        Args:
            titanic_data (pd.DataFrame): Dataframe with the features.
            categorical_features (list): List with the categorical features.
        """
        categorical_features_columns = []

        for col in categorical_features:
            for x in range(0, len(titanic_data[col].dropna().unique())):
                categorical_features_columns.extend([col + str(x)])

        return categorical_features_columns

    def build_full_pipeline(self) -> dict:
        """Builds the full pipeline for the models to be trained.

        Returns:
            dict: Dictionary with the full pipeline for each model.
        """
        atributes_types = self.atributes_types

        processing_feature_selection_pipeline = self._build_data_processing_pipeline(
            atributes_types["numeric_features"],
            atributes_types["ordinal_attributes"],
            atributes_types["categorical_features"],
        )

        proc_and_feat_select_pipeline = (
            self._build_processing_and_feature_selection_pipeline(
                processing_feature_selection_pipeline, self.all_columns
            )
        )

        models_pipelines = self._build_create_model_pipeline(
            proc_and_feat_select_pipeline
        )

        return models_pipelines

    def _build_data_processing_pipeline(
        self,
        numeric_features: list,
        ordinal_attributes: list,
        categorical_features: list,
    ) -> Pipeline:
        """Builds the pipeline for the data processing.

        Args:
            numeric_features (list): List with the numeric features.
            ordinal_attributes (list): List with the ordinal features.
            categorical_features (list): List with the categorical features.

        Returns:
            Pipeline: Pipeline for the data processing.
        """
        num_processing_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("std_scaler", StandardScaler()),
            ]
        )

        cat_processing_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("cat", OneHotEncoder()),
            ]
        )

        data_processing_pipeline = ColumnTransformer(
            [
                ("num", num_processing_pipeline, numeric_features),
                ("ord", OrdinalEncoder(), ordinal_attributes),
                ("cat", cat_processing_pipeline, categorical_features),
            ]
        )

        return data_processing_pipeline

    def _build_processing_and_feature_selection_pipeline(
        self, data_processing_pipeline: Pipeline, new_columns: list
    ) -> Pipeline:
        """Builds the pipeline for the data processing and feature selection.

        Args:
            data_processing_pipeline (Pipeline): Pipeline for the data processing.
            new_columns (list): List with the new columns after the one hot encoding.

        Returns:
            Pipeline: Pipeline for the data processing and feature selection.
        """
        processing_feature_selection_pipeline = Pipeline(
            [
                ("data_processing", data_processing_pipeline),
                ("FeatureSelection", FeatureSelection(columns=new_columns)),
            ]
        )

        return processing_feature_selection_pipeline

    def _build_create_model_pipeline(
        self, processing_feature_selection_pipeline: Pipeline
    ) -> dict:
        """Builds the pipeline for the data processing and feature selection.

        Args:
            processing_feature_selection_pipeline (Pipeline): Pipeline for the data processing and feature selection.

        Returns:
            dict: Dictionary with the full pipeline for each model.
        """
        training_pipelines = {}

        for model in self.models:

            seach_tecnique = self.search_tecnique["technique"]
            seach_params = self.search_tecnique["params"]

            name = model["name"]
            params = model["params"]
            model = model["model"]

            random_seach = seach_tecnique(model, params, **seach_params)

            full_pipeline = Pipeline(
                [
                    ("preparation", processing_feature_selection_pipeline),
                    ("model", random_seach),
                ]
            )

            training_pipelines[name] = full_pipeline

        return training_pipelines
