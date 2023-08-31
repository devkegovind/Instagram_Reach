import os
import sys
import pickle
import numpy as np
import pandas as pd


from dataclasses import dataclass

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler


from src.components.data_ingestion import DataIngestion
from src.exception import CustomException
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_object(self):
        try:
            logging.info("Data Transformation Initiated")

            # Define which columns should be ordinal encoded and which should be scaled
            
            cat_cols = ['Username', 'Caption', 'Hashtags']

            num_cols = ['Followers']

            logging.info("Pipeline Initiated")

            # Numerical Pipeline

            num_pipeline = Pipeline(
                steps = [
                    ('imputer', SimpleImputer(strategy = 'median')),
                    ('scaler', StandardScaler())
                ]
            )

            # Categorical Pipeline

            cat_pipeline = Pipeline(
                steps = [
                    ('imputer', SimpleImputer(strategy = 'most_frequent')),
                    ('encoder', OrdinalEncoder(handle_unknown = 'use_encoded_value',unknown_value = -1))

                ]
            )

            # Preprocessor

            preprocessor = ColumnTransformer(
                transformers = [
                    ('num_pipeline', num_pipeline, num_cols),
                    ('cat_pipeline', cat_pipeline, cat_cols)
                ],
                remainder = 'drop'
            )

            # Final Pipeline
            final_pipeline = Pipeline(
                steps = [
                    ('preprocessor', preprocessor),
                ]
            )
            return final_pipeline

            logging.info("Pipeline Completed")

        except Exception as e:
            logging.info("Error In Data Transformation")
            raise CustomException(e, sys)
        

    def initiate_data_transformation(self, train_path, test_path):
        try:    
            # Reading Train and Test Data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read Train & Test Data Completed")
            logging.info(f"Train Dataframe Head:\n{train_df.head().to_string()}")
            logging.info(f"Test Dataframe Head:\n{test_df.head().to_string()}")
            logging.info(f"Shape of Train_df:{train_df.shape}")
            logging.info(f"Shape of Test_df:{test_df.shape}")
            logging.info("Obtaining Processing Object")

            preprocessing_obj = self.get_data_transformation_object()

            target_column_name = ['Time since posted (hours)', 'Likes'] 

            drop_columns = target_column_name + ['Unnamed: 0']

            input_feature_train_df = train_df.drop(columns = drop_columns, axis = 1)

            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns = drop_columns, axis = 1)

            target_feature_test_df = test_df[target_column_name]

            logging.info(f"Input_Feature_Train_df Head:\n{input_feature_train_df.head().to_string()}")
            logging.info(f"Input Feature_Test_df Head:\n{input_feature_test_df.head().to_string()}")
            logging.info(f"Target Feature_Train_df Head:\n{target_feature_train_df.head().to_string()}")
            logging.info(f"Target Feature_Test_df Head:\n{target_feature_test_df.head().to_string()}")
            logging.info(f"Input_Feature_Train_df Shape:{input_feature_train_df.shape}")
            logging.info(f"Input Feature_Test_df Shape:{input_feature_test_df.shape}")
            logging.info(f"Target Feature_Train_df Shape:{target_feature_train_df.shape}")
            logging.info(f"Target Feature_Test_df Shape:{target_feature_test_df.shape}")

            # Transforming Using Preprocessor Object
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            logging.info(f"Input Feature Train_arr:\n{input_feature_train_arr[:5]}")
            logging.info(f"Input Feature Test_arr:\n{input_feature_test_arr[:5]}")

            logging.info("Applying Preprocessing Object on Train & Test")

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Train Array:\n{train_arr[:5]}")
            logging.info(f"Test Array:\n{test_arr[:5]}")

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )

            logging.info("Preprocessor Pickle File Saved")

            return (
            train_arr,
            test_arr,
            self.data_transformation_config.preprocessor_obj_file_path            
            )
    
        except Exception as e:
            logging.info("Exception Occured in the Initiate_Datatransformation")
            raise CustomException(e,sys)
        





