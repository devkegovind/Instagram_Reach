import os
import sys
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object
from src.utils import evaluate_models

from dataclasses import dataclass

class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, train_array, test_array):
        try:
            logging.info("Splitting Depedent & Independent Features")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-2],
                train_array[:, -2],
                test_array[:, :-2],
                test_array[:,-2]
            )

            models = {
                "Linear Regression" : LinearRegression(),
                "Lasso Regression" : Lasso(),
                "Ridge Regression" : Ridge(),
                "ElasticNet Regression" : ElasticNet()
            }

            model_report:dict = evaluate_models(X_train, X_test, y_train, y_test, models)
            print(model_report)
            print("*"*100)
            logging.info(f"Model Report : {model_report}")

            # To Get Best Model Score From Dictionary
            best_model_score = max(sorted(model_report.value()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            logging.info(f"Best Model Found, Model Name:{best_model_name}, R2 Sccore:{best_model_score}")

            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )

        except Exception as e:
            logging.info("Exception Occured at Model Training ")
            raise CustomException(e, sys)





