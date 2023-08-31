import sys
import os
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            preprocessor_path = os.path.join('artifcats', 'preprocessor.pkl')
            model_path = os.path.join('artifacts', 'model.pkl')
            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)
            data_scaled = preprocessor.transform(features)
            pred = model.predict(data_scaled)
            return pred
        
        except Exception as e:
            logging.info("Exception Occured in Prediction")
            raise CustomException(e,sys)
        


class CustomData:
    def __init_(self,
                Username:str,
                Caption:str,
                Followers:int,
                Hashtags:str):
        self.Username = Username
        self.Caption = Caption
        self.Followers = Followers
        self.Hashtags = Hashtags

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict ={
                'Username' :[self.Username],
                'Caption' :[self.Caption],
                'Followers':[self.Followers],
                'Hashtags':[self.Hashatags]
            }

            df = pd.DataFrame(custom_data_input_dict)
            logging.info("Dataframe Gathered")
            return df
        
        except Exception as e:
            logging.info("Exception Occured in Prediction Pipeline")
            raise CustomData(e,sys)
            