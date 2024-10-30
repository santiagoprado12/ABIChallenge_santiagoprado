import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


class Configs:
    """Class to store all the configurations of the project.

    Attributes:
        search_tecnique (dict): Dictionary with the search technique to use
        models (list): List of dictionaries with the models to use
    """

    search_tecnique = {
        'technique': RandomizedSearchCV,
        'params': {
            'n_iter': 15,
            'cv': 5,
            'verbose': 0,
            'n_jobs': -1
        }         
    }

    # search_tecnique = {
    #     'technique': GridSearchCV,
    #     'params': {
    #         'cv': 5,
    #         'verbose': 0,
    #         'n_jobs': -1
    #     }
    # }


    models = [
        {
            'name': 'random_forest',
            'model': RandomForestClassifier(),
            'params': {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2'],
                'bootstrap': [True, False],
                'random_state': [42]
            }
        }
        ,{
            'name': 'gradient_boosting',
            'model': GradientBoostingClassifier(),
            'params': {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 4, 5],
                'min_samples_split': [2, 3, 4],
                'min_samples_leaf': [1, 2, 3],
                'subsample': [0.8, 0.9, 1.0],
                'random_state': [42]
            }
        }
        ,{
            'name': 'knn',
            'model': KNeighborsClassifier(),
            'params': {
                'n_neighbors': np.arange(1, 21),
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan']
            }
        }
        # ,{
        #     'name': 'svm',
        #     'model': SVC(),
        #     'params': {
        #         'C': [0.1, 1, 10, 100],
        #         'gamma': [1, 0.1, 0.01, 0.001],
        #         'kernel': ['rbf', 'poly', 'sigmoid']
        #     }
        # }

    ]
