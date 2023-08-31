import os 
import sys
import pickle
import numpy as np
import pandas as pd


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor

from src.logger import logging
from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok = True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)



def evaluate_models(X_train, y_train, X_test, y_test, models):
   
    try:
       for i in range(len(list(models))):
            model = list(models.values())[i]
            model.fit(X_train, y_train)

            # Make Prediction
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            
                                                                                                                                                                                            'MSE_Train', 'MSE_Test', 'RMSE_Train', 'RMSE_Test', 'R2SCORE_Train', 'R2SCORE_Test', 'R2SCOREDIFF']).sort_values(by = 'R2SCORE_Test', ascending = False)
        return report
    except Exception as e:
        logging.info("Exception Occured During Model Training")
        raise CustomException(e, sys)
